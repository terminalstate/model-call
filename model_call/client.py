"""Client.structured(): ask for a JSON object that matches a schema, and get either a usable object or a
named reason why not.

What happens after the first answer, in order (the order comes from a measurement, see the README):
  1. it is checked: the API's finish reason, JSON, the schema, then your own value rules (`check`);
  2. if the shape is wrong, it is repaired locally, for nothing (code fence, misspelt key, "year" for
     "years", a number in quotes, a one-item list, a cut-off object);
  3. if it is still not usable, one more request, chosen by what went wrong:
       cut off at the output limit      -> the same request with `bigger_factor` times the limit and time;
       wrong content, wrapped, no call  -> the answer goes back with what was wrong with it ("feedback");
       empty, refused, stopped          -> the same request again;
     and the new answer goes through 1 and 2;
  4. otherwise the result says what is wrong. With allow_partial, an answer with unreadable fields comes
     back with those fields listed in `dropped`.
An HTTP failure is retried by the transport (see transport.send) and then reported; a 400 is reported at once,
because sending the same request again or arguing with the model will not change it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field

from . import providers as P
from .budget import Budget, BudgetExceeded, Price
from .log import CallLog
from .pacing import Pacer
from .repair import extract_json, repair
from .schema import check_schema, validate
from .transport import RETRYABLE, ModelCallError, RetryPolicy, Sent, scrub, send

WRAPPED = "the JSON object came with a code fence or other text around it; send the JSON object alone"
FEEDBACK_ON = {"wrapped", "invalid_json", "schema", "value", "no_tool_call"}  # empty, refusal, stopped: a plain retry


@dataclass
class Result:
    ok: bool  # value is usable: complete, or partial with allow_partial
    value: dict | None
    complete: bool
    category: str  # what was wrong with the first answer: ok, wrapped, invalid_json, schema, value, truncated,
    #                no_tool_call, empty, refusal, stopped, rejected (HTTP 4xx), http_failed
    recovered_by: str | None = None  # local, feedback, bigger, retry (+local); None if not needed or not found
    problems: list = field(default_factory=list)  # what is still wrong
    notes: list = field(default_factory=list)  # local repairs that were made
    dropped: list = field(default_factory=list)  # fields that could not be read
    requests: int = 0
    attempts: list = field(default_factory=list)
    usage: dict = field(default_factory=lambda: {"input": 0, "cached_input": 0, "output": 0, "reasoning": 0})
    cost_usd: float = 0.0  # at your prices; an attempt that timed out counts at its worst case
    seconds: float = 0.0


def examine(reply: P.Reply, schema: dict, check=None):
    """-> (category, problems, object or None)"""
    fin = reply.finish or ""
    if fin in P.STOPPED:
        return "stopped", [f"finish reason {fin}"], None
    if fin == "refusal":
        return "refusal", ["the model refused"], None
    if fin in P.TRUNCATED:
        return "truncated", ["cut off at the output limit"], None
    obj = reply.obj
    if obj is None:
        text = reply.text or ""
        if not text.strip():
            return "empty", ["empty answer"], None
        if reply.tool_called is False:
            return "no_tool_call", ["no tool call, answered in text"], None
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            if extract_json(text) is not None:
                return "wrapped", [WRAPPED], None
            return "invalid_json", [f"not valid JSON: {e.msg} at character {e.pos}"], None
    problems = validate(schema, obj) if isinstance(obj, dict) else [f"expected a JSON object, got {type(obj).__name__}"]
    if problems:
        return "schema", problems, None
    if check:
        problems = list(check(obj) or [])
        if problems:
            return "value", problems, None
    return "ok", [], obj


class Client:
    def __init__(
        self,
        provider: str,
        model: str,
        *,
        mechanism: str = "tool",
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 1024,
        retry: RetryPolicy | None = None,
        price: Price | None = None,
        budget: Budget | None = None,
        pacer: Pacer | None = None,
        log: CallLog | str | None = None,
        extra_body: dict | None = None,
        tool_name: str = "submit",
        tool_description: str = "Submit the answer.",
        bigger_factor: int = 4,
        allow_partial: bool = False,
    ):
        P.check_mechanism(provider, mechanism)
        if budget is not None and price is None:
            raise ValueError("a budget needs a price to hold")
        self.provider, self.model, self.mechanism = provider, model, mechanism
        self.api_key, self.base_url = api_key, base_url
        self.max_tokens = max_tokens
        self.retry = retry or RetryPolicy()
        self.price, self.budget, self.pacer = price, budget, pacer
        self.extra_body = dict(extra_body or {})
        self.tool_name, self.tool_description = tool_name, tool_description
        self.bigger_factor = bigger_factor
        self.allow_partial = allow_partial
        if isinstance(log, str):
            log = CallLog(log)
        self.log = log

    # ------------------------------------------------------------------ one request

    def _request(self, ask: P.Ask, res: Result, trail: list, extra=(), scale: int = 1):
        """-> (reply, sent); reply is None on an HTTP failure. Every cent reserved is settled on every path."""
        url, headers = P.endpoint(self.provider, ask, self.base_url, self.api_key)
        limit = ask.max_tokens * scale
        body = P.build(self.provider, self.model, ask, extra, limit)
        worst = self.price.worst_case(body, limit) if self.price is not None else 0.0
        held = [self.budget.reserve(worst)] if self.budget is not None else []
        if self.pacer is not None:  # after the budget, so a refused request takes no rate slot
            try:
                self.pacer.acquire(math.ceil(len(json.dumps(body, ensure_ascii=False)) / 3) + limit)
            except BaseException:
                if held:
                    self.budget.settle(sum(held), 0.0)
                raise

        def before_resend() -> bool:
            # The attempt before may have been billed: another one needs room for its own worst case.
            if self.budget is None:
                return True
            try:
                held.append(self.budget.reserve(worst))
            except BudgetExceeded:
                return False
            return True

        res.requests += 1
        entry = {"max_tokens": limit, "attempts": [], "body": body}
        trail.append(entry)
        sent, reply, spent = None, None, None
        try:
            try:
                sent = send(url, headers, body, self.retry, scale, before_resend)
            except ModelCallError as e:
                sent = getattr(e, "sent", None) or Sent(None)
                spent = sent.maybe_billed * worst  # earlier attempts may have been billed before the refusal
                raise
            if sent.data is not None:
                try:
                    reply = P.parse(self.provider, ask, sent.data)
                except (AttributeError, TypeError, KeyError, IndexError, ValueError) as e:
                    sent.detail = f"unexpected response shape: {e!r}"[:300]
                    sent.attempts[-1]["maybe_billed"] = True
            spent = sent.maybe_billed * worst + (self.price.cost(reply.usage) if reply and self.price else 0.0)
        finally:
            if spent is None:  # interrupted: whatever was reserved may have been billed
                spent = sum(held) if held else worst
            if self.budget is not None:
                self.budget.settle(sum(held), spent)
            res.cost_usd += spent
            if sent is not None:
                res.attempts += sent.attempts
                entry["attempts"] = sent.attempts
        if reply is None:
            entry["error"] = sent.detail
            return None, sent
        for k, v in reply.usage.items():
            res.usage[k] = res.usage.get(k, 0) + v
        entry["response"] = sent.data
        entry["finish"] = reply.finish
        return reply, sent

    # ------------------------------------------------------------------ the call

    def structured(self, system: str, user: str, schema: dict, check=None, allow_partial: bool | None = None) -> Result:
        """`check(obj) -> list of problems` holds your value rules (a date that is not a real date, an amount
        that does not add up). It runs on complete answers; its problems go back to the model as feedback."""
        check_schema(schema)
        if schema.get("type") != "object":
            raise ValueError("the schema's root must be an object: tools and strict modes require it")
        partial_ok = self.allow_partial if allow_partial is None else allow_partial
        ask = P.Ask(system, user, schema, self.mechanism, self.tool_name, self.tool_description, self.max_tokens, self.extra_body)
        t0 = time.monotonic()
        res = Result(ok=False, value=None, complete=False, category="")
        trail = []
        try:
            self._call(ask, schema, check, partial_ok, res, trail)
        finally:
            res.seconds = round(time.monotonic() - t0, 3)
            res.cost_usd = round(res.cost_usd, 8)
            if self.log is not None and trail:
                self.log.write(self._record(res, trail))
        return res

    def _call(self, ask, schema, check, partial_ok, res, trail):
        reply, sent = self._request(ask, res, trail)
        if reply is None:
            refused = sent.status is not None and 400 <= sent.status < 500 and sent.status not in RETRYABLE
            res.category = "rejected" if refused else "http_failed"
            res.problems = [self._http_problem(sent)]
            return
        category, problems, obj = examine(reply, schema, check)
        res.category = category
        if category == "ok":
            res.ok, res.complete, res.value = True, True, obj
            return
        done, partial, value_problems = self._try_local(reply, category, schema, check, res, "local")
        if done:
            return
        problems = problems + value_problems
        policy = "bigger" if category == "truncated" else "feedback" if category in FEEDBACK_ON else "retry"
        extra = P.follow_up(self.provider, ask, sent.data, problems, category) if policy == "feedback" else ()
        try:
            reply2, sent2 = self._request(ask, res, trail, extra, self.bigger_factor if policy == "bigger" else 1)
        except BudgetExceeded as e:
            reply2, sent2 = None, None
            problems = problems + [f"no second request: {e}"]
        if reply2 is not None:
            category2, problems2, obj2 = examine(reply2, schema, check)
            if category2 == "ok":
                res.ok, res.complete, res.value, res.recovered_by = True, True, obj2, policy
                res.problems = []
                return
            done, partial2, value_problems = self._try_local(reply2, category2, schema, check, res, policy + "+local")
            if done:
                return
            partial = partial2 or partial
            problems = problems2 + value_problems
        elif sent2 is not None:
            problems = problems + [self._http_problem(sent2)]
        res.problems = problems
        if partial is not None:
            res.dropped, res.notes = partial.dropped, partial.notes
            if partial_ok:  # otherwise value stays None: a partial answer is handed out only when asked for
                res.ok, res.value, res.recovered_by = True, partial.value, "partial"
                res.problems = [f"could not read: {', '.join(partial.dropped)}"] if partial.dropped else problems

    def _try_local(self, reply, category, schema, check, res, label):
        """-> (done, partial repair or None, value problems of a repaired answer). done: res holds a complete
        answer. A repaired answer that breaks a value rule is not used; its problems go into the feedback."""
        answer = reply.obj if reply.obj is not None else reply.text
        if answer is None or (isinstance(answer, str) and not answer.strip()):
            return False, None, []
        rep = repair(answer, schema, cut_off=category in ("truncated", "stopped"))
        if rep.value is None:
            return False, None, []
        if not rep.complete:
            return False, rep, []
        value_problems = list(check(rep.value) or []) if check and category != "value" else []
        if category == "value" or value_problems:
            return False, None, value_problems
        res.ok, res.complete, res.value = True, True, rep.value
        res.notes, res.dropped, res.recovered_by, res.problems = rep.notes, [], label, []
        return True, None, []

    def _http_problem(self, sent) -> str:
        last = sent.attempts[-1] if sent.attempts else {}
        return scrub(f"{last.get('kind', 'http')} {sent.status}: {sent.detail}"[:400])

    def _record(self, res: Result, trail: list) -> dict:
        return {
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "provider": self.provider,
            "model": self.model,
            "mechanism": self.mechanism,
            "ok": res.ok,
            "complete": res.complete,
            "category": res.category,
            "recovered_by": res.recovered_by,
            "problems": res.problems,
            "notes": res.notes,
            "dropped": res.dropped,
            "usage": res.usage,
            "cost_usd": res.cost_usd,
            "seconds": res.seconds,
            "value": res.value,
            "requests": trail,
        }
