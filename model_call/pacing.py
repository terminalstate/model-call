"""Client-side pacing for batches.

A 429's Retry-After says when to try once more, not when a batch will fit. OpenAI counts a request against
the tokens-per-minute limit by its output limit or its estimated size, whichever is larger, so a batch sent
at once can fail for as long as the client retries. Pacing sends requests at the rate the account allows.
Set the limits from your account's rate-limit page or headers; the estimate here is the same one the budget
uses (a token per three characters of the request) plus the output limit.
"""

from __future__ import annotations

import collections
import threading
import time


class Pacer:
    def __init__(
        self, requests_per_minute: int | None = None, tokens_per_minute: int | None = None, clock=time.monotonic, sleep=time.sleep
    ):
        if not requests_per_minute and not tokens_per_minute:
            raise ValueError("give requests_per_minute, tokens_per_minute or both")
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._events = collections.deque()  # (time, tokens)
        self._lock = threading.Lock()
        self._clock = clock
        self._sleep = sleep

    def acquire(self, tokens: int = 0) -> float:
        """Blocks until a request of `tokens` fits in the last minute's window. -> seconds waited.
        A single request larger than the whole per-minute limit goes through alone."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                while self._events and now - self._events[0][0] >= 60:
                    self._events.popleft()
                used = sum(t for _, t in self._events)
                fits_requests = not self.rpm or len(self._events) < self.rpm
                fits_tokens = not self.tpm or used + tokens <= self.tpm or not self._events
                if fits_requests and fits_tokens:
                    self._events.append((now, tokens))
                    return waited
                pause = max(0.05, 60 - (now - self._events[0][0]))
            self._sleep(pause)
            waited += pause
