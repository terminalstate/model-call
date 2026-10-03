"""HTTP for model APIs: one POST with a deadline on the whole call, and the retries around it.

Two things most HTTP clients leave out:
  * `timeout` limits each wait for data, not the call. A server that sends a byte now and then keeps the call
    open as long as it likes (DeepSeek documents sending empty lines while a request waits in its queue, up
    to 10 minutes). `deadline_s` limits the whole call.
  * A request that timed out may still be processed and billed. It is sent again once at most.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import random
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
KEEP_HEADERS = (
    "retry-after",
    "retry-after-ms",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
    "anthropic-ratelimit-requests-remaining",
    "anthropic-ratelimit-tokens-remaining",
    "anthropic-ratelimit-input-tokens-remaining",
    "anthropic-ratelimit-output-tokens-remaining",
)
_IDS = (
    (re.compile(r"\borg-[A-Za-z0-9]+"), "org-…"),
    (re.compile(r"\breq_[A-Za-z0-9]+"), "req_…"),
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"), "sk-…"),
)
sleep = time.sleep  # tests replace it


class ModelCallError(Exception):
    """A failure that retrying will not fix: fix the configuration and run again."""


class AuthError(ModelCallError):
    pass


class BalanceError(ModelCallError):
    pass


class NotFoundError(ModelCallError):
    pass


def scrub(text: str) -> str:
    """Account and request identifiers out of an error text (an OpenAI 429 names the organization)."""
    for rx, repl in _IDS:
        text = rx.sub(repl, text)
    return text


@dataclass
class RetryPolicy:
    max_attempts: int = 4
    timeout_s: float = 60.0  # longest wait for any data
    deadline_s: float = 120.0  # longest single attempt, start to end
    total_s: float | None = None  # longest time for all attempts and the waits between them
    backoff_base: float = 1.0
    backoff_cap: float = 30.0  # longest wait of our own choosing; a server's Retry-After is honoured above it
    max_retry_after: float = 120.0  # a server that asks to wait longer than this ends the call instead
    resend_after_timeout: int = 1  # attempts allowed after one that may have been processed


@dataclass
class Sent:
    data: dict | None
    attempts: list = field(default_factory=list)
    status: int | None = None
    detail: str = ""

    @property
    def maybe_billed(self) -> int:
        """Attempts that ended without a usable answer after the request reached the server (a timeout, a
        connection lost after the answer started, a 200 with a broken body): the provider may bill them."""
        return sum(1 for a in self.attempts if a.get("maybe_billed"))


def _lower(headers) -> dict:
    return {k.lower(): v for k, v in (headers.items() if headers else [])}


def post(url: str, headers: dict, body: dict, timeout: float, deadline: float = 0.0):
    """POST JSON -> (status, headers, raw, info); info: latency_s, headers_s, kind, keepalive_bytes.
    `deadline` is checked between reads, so a call ends within about twice the deadline at worst."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST", headers={**headers, "content-type": "application/json"}
    )
    t0 = time.monotonic()
    wait = min(timeout, deadline) if deadline else timeout
    info = {"headers_s": None, "keepalive_bytes": 0}

    def done(status, hdrs, raw, kind):
        info.update(latency_s=round(time.monotonic() - t0, 3), kind=kind)
        return status, hdrs, raw, info

    try:
        r = urllib.request.urlopen(req, timeout=wait)
    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
        except Exception:
            raw = b""
        return done(e.code, _lower(e.headers), raw, "http")
    except (socket.timeout, TimeoutError) as e:
        return done(None, {}, str(e).encode(), "timeout")
    except urllib.error.URLError as e:
        kind = "timeout" if isinstance(e.reason, (socket.timeout, TimeoutError)) else "network"
        return done(None, {}, str(e.reason).encode(), kind)
    except (ConnectionError, http.client.HTTPException, OSError) as e:
        return done(None, {}, repr(e).encode(), "network")
    info["headers_s"] = round(time.monotonic() - t0, 3)
    hdrs = _lower(r.headers)
    chunks = []
    try:
        with r:
            while True:
                if deadline and time.monotonic() - t0 > deadline:
                    return done(None, hdrs, b"".join(chunks), "deadline")
                chunk = r.read1(65536)
                if not chunk:
                    break
                chunks.append(chunk)
    except (socket.timeout, TimeoutError):
        late = deadline and time.monotonic() - t0 >= deadline
        return done(None, hdrs, b"".join(chunks), "deadline" if late else "timeout")
    except (ConnectionError, http.client.HTTPException, OSError) as e:
        return done(None, hdrs, repr(e).encode(), "network")
    raw = b"".join(chunks)
    info["keepalive_bytes"] = len(raw) - len(raw.lstrip())
    return done(r.status, hdrs, raw, "http")


def retry_after(headers: dict):
    """Seconds the server asked to wait, or None. Reads retry-after-ms, retry-after in seconds or as a date."""
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    ra = headers.get("retry-after")
    if not ra:
        return None
    try:
        return max(0.0, float(ra))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(ra)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def send(url: str, headers: dict, body: dict, policy: RetryPolicy, scale: float = 1.0, before_resend=None) -> Sent:
    """POST with retries. Retries 408/409/425/429/5xx, timeouts and dropped connections with exponential
    backoff and jitter, waiting at least Retry-After. 400 and other 4xx end the call (the request itself is
    wrong). 401/403/402/404 raise, because every later request would fail the same way; the exception carries
    the attempts made so far as `.sent`.

    After an attempt that may have been processed, only `resend_after_timeout` more attempts are made, and
    `before_resend()` is asked first: if it returns False (a budget, say), nothing more is sent.
    `scale` stretches the timeouts for a request that is allowed a larger output."""
    out = Sent(None)
    t0 = time.monotonic()
    after_billed = None  # attempts made since the first one that may have been processed
    for n in range(policy.max_attempts):
        if after_billed is not None:
            if after_billed >= policy.resend_after_timeout:
                break
            if before_resend is not None and not before_resend():
                out.detail += "; not sent again: no room left in the budget"
                break
            after_billed += 1
        deadline = policy.deadline_s * scale
        if policy.total_s is not None:
            left = policy.total_s - (time.monotonic() - t0)
            if left <= 0:
                break
            deadline = min(deadline, left)
        started = time.time()
        status, rh, raw, info = post(url, headers, body, policy.timeout_s * scale, deadline)
        rec = {"at": round(started, 3), "status": status, **info}
        kept = {k: rh[k] for k in KEEP_HEADERS if k in rh}
        if kept:
            rec["headers"] = kept
        out.attempts.append(rec)
        out.status = status
        detail = scrub(raw[:600].decode("utf-8", "replace")) if status != 200 else ""
        if status == 200:
            try:
                data = json.loads(raw)
                if not isinstance(data, dict):
                    raise ValueError(f"expected a JSON object, got {type(data).__name__}")
                out.data, out.detail = data, ""
                return out
            except (ValueError, UnicodeDecodeError) as e:
                rec["kind"] = "bad response body"
                detail = scrub(f"{e!r}: {raw[:120]!r}")
        elif status in (401, 403, 402, 404):
            cls, what = {
                401: (AuthError, "check the API key"),
                403: (AuthError, "check the API key and its permissions"),
                402: (BalanceError, "balance too low"),
                404: (NotFoundError, "unknown model or endpoint"),
            }[status]
            err = cls(f"HTTP {status}, {what}: {detail[:300]}")
            err.sent = out
            raise err
        elif status is not None and status not in RETRYABLE:
            rec["detail"] = out.detail = detail
            return out
        rec["detail"] = out.detail = detail
        if info["kind"] in ("timeout", "deadline") or status == 200 or (status is None and info.get("headers_s") is not None):
            rec["maybe_billed"] = True
            if after_billed is None:
                after_billed = 0
        if n + 1 == policy.max_attempts:
            break
        delay = min(policy.backoff_cap, policy.backoff_base * (2**n))
        delay = delay / 2 + random.uniform(0, delay / 2)
        asked = retry_after(rh)
        if asked is not None:
            if asked > policy.max_retry_after:
                out.detail += f"; the server asked to wait {asked:.0f} s, more than max_retry_after"
                break
            delay = max(delay, asked)
        if policy.total_s is not None and time.monotonic() - t0 + delay >= policy.total_s:
            break
        rec["retry_in_s"] = round(delay, 2)
        sleep(delay)
    return out
