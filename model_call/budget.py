"""Prices and a spending ceiling that holds under concurrency.

A request is authorised before it is sent and settled after, like a card payment: its worst case (the input
estimated from the request size, plus the whole output limit) is reserved, so twenty threads cannot all pass
the check at once and overspend together. A request that timed out after it went out is settled at its worst
case: the provider may have processed and billed it, and nobody knows for how much.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass

from .transport import ModelCallError


class BudgetExceeded(ModelCallError):
    """The next request could cost more than what is left. Nothing was sent."""


@dataclass
class Price:
    """USD per million tokens. Reasoning tokens are billed as output by all three APIs."""

    input: float
    output: float
    cached_input: float | None = None

    def cost(self, usage: dict) -> float:
        cached = usage.get("cached_input", 0) or 0
        fresh = max(0, (usage.get("input", 0) or 0) - cached)
        cached_rate = self.input if self.cached_input is None else self.cached_input
        return (fresh * self.input + cached * cached_rate + (usage.get("output", 0) or 0) * self.output) / 1e6

    def worst_case(self, body: dict, max_tokens: int) -> float:
        """An upper estimate: one token per three characters of the request, and the whole output limit."""
        chars = len(json.dumps(body, ensure_ascii=False))
        return (math.ceil(chars / 3) * self.input + max_tokens * self.output) / 1e6


class Budget:
    def __init__(self, max_usd: float):
        if max_usd <= 0:
            raise ValueError("max_usd must be positive")
        self.max_usd = max_usd
        self._spent = 0.0
        self._held = 0.0
        self._lock = threading.Lock()

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def left(self) -> float:
        with self._lock:
            return self.max_usd - self._spent - self._held

    def reserve(self, usd: float) -> float:
        with self._lock:
            if self._spent + self._held + usd > self.max_usd:
                raise BudgetExceeded(
                    f"the next request could cost ${usd:.4f}; spent ${self._spent:.4f}, "
                    f"held for requests in flight ${self._held:.4f}, ceiling ${self.max_usd:.2f}"
                )
            self._held += usd
            return usd

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self._held = max(0.0, self._held - reserved)
            self._spent += actual
