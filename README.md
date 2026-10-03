# model-call

Structured output from model APIs that still works on a bad day.

A model API is an external dependency, like a payment provider. It times out and may still bill the request
the client gave up on; it returns JSON in a code fence, stops mid-answer, refuses, rate-limits a batch, and
costs more than the demo did. The fixes are the ones payments settled on long ago: check what came back, bound the time, retry
the right way, make side effects idempotent, cap the spend, keep a record. This library does that for one
kind of call: "give me a JSON object that matches this schema".

Standard library only. Python 3.9+. Anthropic Messages, OpenAI Chat Completions and DeepSeek (OpenAI-compatible,
with its quirks), each with four ways to ask: schema in the prompt, JSON mode, a tool call, or a strict schema.

```
pip install git+https://github.com/terminalstate/model-call
```

```python
from model_call import Budget, CallLog, Client, Price

client = Client(
    "anthropic",
    "claude-haiku-4-5",
    mechanism="tool",
    price=Price(input=1.00, output=5.00, cached_input=0.10),  # USD per million tokens, from the pricing page
    budget=Budget(max_usd=5.00),
    log=CallLog("calls.jsonl"),
)
res = client.structured(system, document_text, schema, check=my_value_rules)
if res.ok:
    save(res.value)
else:
    print(res.category, res.problems)  # what went wrong, in words
```

## Where it comes from

The order of what happens after a bad answer is not a guess. It is what worked in a measurement of 1,625
answers from three APIs on the same extraction task:
[Structured output on a bad day](https://github.com/terminalstate/extraction-bench/blob/main/reliability.md).
The short version:

| first answer was | answers | same request again | the answer + what was wrong ("feedback") | local repair | 4× output limit |
|---|---|---|---|---|---|
| wrapped in a code fence (one model, all 83 of its prompt-only answers) | 83 | 0 | 0 | 82 (+1 partial) | — |
| against the schema (15 of them `"year"` where the enum says `"years"`) | 16 | 6 | 16 | 16 | — |
| against a value rule | 4 | 1 | 4 | 0 (+4 partial) | — |
| cut off: reasoning used the whole output limit | 11 | 6 | — | 0 | 11 |

So: shape is fixed locally, for nothing; wrong content needs feedback, not a retry; a cut-off reasoning
model needs room, not a second try.

## Six places an AI feature breaks

| # | Place | What happens | What model-call does |
|---|---|---|---|
| 1 | The answer is taken on trust | Broken or wrapped JSON, a misspelt key, a value outside the enum: `json.loads` fails, or worse, succeeds | Every answer is checked: finish reason, JSON, the schema, then your value rules (`check`). Shape problems are repaired locally; the repair never invents a value and lists what it could not read |
| 2 | A slow call holds everything | A read timeout limits each wait for data, not the call: a server that sends an empty line now and then keeps it open (DeepSeek documents doing this for up to 10 minutes). With thinking on, 17–22% of one model's answers took over 10 s and carried 32–50% of the spend | `deadline_s` limits the whole attempt, `total_s` all attempts and the waits between them. Long work belongs on a queue; this gives you a call with a known worst case to put there |
| 3 | Retries that make it worse | The same request again fixes wrong content 7 times in 20; a timed-out request may still be processed and billed; a repeated tool call charges the card twice | The retry is chosen by what went wrong (table above). A request that may have been processed (timed out, or cut off after the answer started) is sent again once at most. `ToolLedger` runs a tool's side effect once per operation you name |
| 4 | Nobody capped the cost | One long input or a reasoning model with a high limit, times a batch | `Budget` reserves each request's worst case before sending and settles the real cost after, so threads cannot overspend together; a resend after a timeout needs room for its own worst case. A request that may have been processed without an answer is charged at its worst case: nobody knows what it cost |
| 5 | "It works" is a feeling | No numbers on your own data | Not this library's job: see [extraction-bench](https://github.com/terminalstate/extraction-bench). The log below is where an evaluation set starts |
| 6 | Nothing to look at afterwards | The one answer that went wrong in production cannot be found | `CallLog` writes one JSON line per call: every request, response, HTTP attempt, repair, token and cent. Keys are never written; account and request ids in error texts are masked |

And one for batches: a 429's `Retry-After` says when to try once more, not when a batch will fit. With 83
requests sent at once, 43 failed at one provider after five attempts each, while every 429 said to retry in
1–2 seconds: its tokens-per-minute limit counts each request at its output limit or its estimated size,
whichever is larger. `Pacer` sends
requests at the rate your account allows instead.

## What happens after an answer

1. **Check.** `examine()` names the problem: `ok`, `wrapped`, `invalid_json`, `schema`, `value`,
   `truncated`, `no_tool_call`, `empty`, `refusal`, `stopped`.
2. **Repair locally** when the shape is wrong: JSON out of a code fence or prose (if there is exactly one
   candidate: an example followed by the answer is ambiguous and is not guessed at), trailing commas, a
   cut-off object closed at its last whole value, a one-item list unwrapped; a key that differs from a
   schema key only in case or spacing, in singular/plural, or by a typo (in keys of five letters or more:
   one letter up to six, two above, and only if the value fits that field), when the match is unique both
   ways; `"2"` read as `2`, `"x"` as `["x"]`, `"year"` as
   `"years"` when exactly one enum value matches, and only then `"n/a"` as null where null is allowed. A
   field it cannot read is left empty and listed in `dropped`, and so is a key given twice with different
   values. After a cut-off (or when the API parsed a cut-off answer for you), a field that never came is
   unknown, not null, and the field the text may have stopped in is not trusted.
3. **One more request**, if still unusable: cut off → same request with 4× the output limit and time;
   wrong content, wrapped, no tool call → the answer goes back with what was wrong with it; empty, refused,
   stopped → the same request again. The new answer goes through 1 and 2.
4. **Report.** `Result.ok` is true only for a complete answer, and `value` is None otherwise. With
   `allow_partial=True` a partial answer comes back too, with `complete=False` and its unreadable fields in
   `dropped`.

HTTP is handled below this: 408/409/425/429/5xx, timeouts and dropped connections are retried with
exponential backoff and jitter, waiting at least `Retry-After` (a server that asks for more than
`max_retry_after`, two minutes by default, ends the call instead); a 400 is reported at once as `rejected`,
because neither a retry nor an argument with the model changes it; 401/403, 402 and 404 raise, because every
later request would fail the same way.

## Result

| field | |
|---|---|
| `ok`, `value`, `complete` | usable or not, the object, whether every field was read |
| `category` | what was wrong with the first answer (`ok` if nothing) |
| `recovered_by` | `local`, `feedback`, `bigger`, `retry` (with `+local` if the second answer was repaired too), `partial`, or None |
| `problems`, `notes`, `dropped` | what is still wrong, what was repaired, what could not be read |
| `requests`, `attempts` | model requests, and every HTTP attempt with status, timing and rate-limit headers |
| `usage`, `cost_usd`, `seconds` | tokens (input, cached, output, reasoning), money at your prices (an attempt that may have been processed without an answer counts at its worst case), wall time |

## Tool calls that run once

```python
from model_call import ToolLedger, NotApplied

ledger = ToolLedger("actions.db")  # SQLite: holds across threads, processes and restarts


def refund(order_id, args):
    key = ToolLedger.key(f"order-{order_id}/refund", "refund")  # the operation, from your side
    return ledger.once(key, lambda: gateway.refund(order_id, args["amount"]))
```

The key names the business operation and comes from your side. A model's tool-call id is new on every
answer, so it cannot tell a repeat from a new request; the model's arguments cannot either, because an
argument corrected after feedback, or 500 written as 500.0, would make a new key and a second refund. Put
arguments in the key only when different arguments really mean a different action. A repeat returns the
result stored as JSON. If the action raised, or the process died, after it may have reached the outside world,
the next `once()` with that key raises `InDoubt` instead of running it again: check what happened, then
`resolve(key, result)` or `forget(key)`. Raise `NotApplied` from the action when you know nothing happened,
and the key is free for another try. Unknown is not the same as failed.

## Options worth knowing

- `extra_body` goes into the request as is: `{"thinking": {"type": "disabled"}}` for DeepSeek,
  `{"reasoning_effort": "low"}` for OpenAI. With thinking on, DeepSeek and Anthropic refuse a forced tool
  call, so the tool is offered and the prompt asks for it.
- `RetryPolicy(max_attempts=4, timeout_s=60, deadline_s=120, total_s=None, backoff_base=1, backoff_cap=30, max_retry_after=120, resend_after_timeout=1)`.
- `Pacer(requests_per_minute=..., tokens_per_minute=...)`, shared by all clients that use one account. It
  paces requests, not the HTTP retries inside one.
- `CallLog(path, content=False)` keeps the shape of each call (statuses, timings, tokens, cost, categories)
  without prompts, answers or problem texts, which can quote the answer.

## What it does not do

- No streaming, no async (threads are fine), no multi-turn agent loop: one structured answer per call.
- JSON Schema is a subset: type, properties, required, additionalProperties, items, enum, const, anyOf,
  bounds, lengths, pattern. Anything else (`$ref`, `oneOf`, `allOf`) raises `SchemaError` instead of being
  skipped.
- Prices are yours to pass; they change. The worst case reserved before a request estimates the input at
  one token per three characters. Anthropic cache writes are priced as input.
- Combinations an API refuses (some OpenAI models refuse function tools together with `reasoning_effort` in
  Chat Completions) come back as `rejected` with the API's own message. They are not worked around.

## Tests

```
python -m unittest discover -s tests
```

102 tests against a local fake of each API; no network, no keys. `examples/smoke.py` runs nine calls against
the real APIs with your keys, under a $0.05 ceiling. On 2026-10-03 all nine came back usable, $0.0042 in all;
claude-haiku-4-5 wrapped its prompt-only answer in a code fence, as in the measurement, and it was repaired
locally. The case meant to be cut off was not (182 output tokens under a 250 limit that time), so the
larger-limit path is covered by the tests only.

## License

MIT
