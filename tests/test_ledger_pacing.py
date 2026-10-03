import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_call import Budget, BudgetExceeded, InDoubt, NotApplied, Pacer, Price, ToolLedger  # noqa: E402


class LedgerTest(unittest.TestCase):
    def test_runs_once(self):
        led, calls = ToolLedger(), []
        key = ToolLedger.key("order-17/refund", "refund", {"amount": 500, "currency": "EUR"})
        for _ in range(3):
            self.assertEqual(led.once(key, lambda: calls.append(1) or {"refund_id": "r1"}), {"refund_id": "r1"})
        self.assertEqual(calls, [1])
        self.assertEqual(led.state(key), "done")

    def test_key_is_ours_not_the_models(self):
        a = ToolLedger.key("order-17/refund", "refund", {"amount": 500, "currency": "EUR"})
        b = ToolLedger.key("order-17/refund", "refund", {"currency": "EUR", "amount": 500})
        c = ToolLedger.key("order-18/refund", "refund", {"amount": 500, "currency": "EUR"})
        self.assertEqual(a, b)  # argument order does not matter
        self.assertNotEqual(a, c)

    def test_key_without_arguments(self):
        led, calls = ToolLedger(), []
        k = ToolLedger.key("order-17/refund", "refund")
        led.once(k, lambda: calls.append(1))
        led.once(k, lambda: calls.append(2))  # the model "corrected" its arguments: still the same refund
        self.assertEqual(calls, [1])

    def test_result_json_cannot_hold_is_still_recorded(self):
        led = ToolLedger()
        led.once("k", lambda: {(1, 2): "tuple key"})
        self.assertEqual(led.state("k"), "done")

    def test_failure_after_start_is_in_doubt(self):
        led = ToolLedger()

        def boom():
            raise TimeoutError("gateway did not answer")

        with self.assertRaises(TimeoutError):
            led.once("k", boom)
        with self.assertRaises(InDoubt) as e:
            led.once("k", lambda: "second run")
        self.assertIn("TimeoutError", str(e.exception))
        led.resolve("k", {"status": "applied"})  # checked by hand: it went through
        self.assertEqual(led.once("k", lambda: "never"), {"status": "applied"})

    def test_forget_allows_a_new_run(self):
        led = ToolLedger()
        with self.assertRaises(ValueError):
            led.once("k", lambda: (_ for _ in ()).throw(ValueError("x")))
        led.forget("k")
        self.assertEqual(led.once("k", lambda: 2), 2)

    def test_not_applied_frees_the_key(self):
        led = ToolLedger()

        def declined():
            raise NotApplied("validation failed before anything was sent")

        with self.assertRaises(NotApplied):
            led.once("k", declined)
        self.assertIsNone(led.state("k"))
        self.assertEqual(led.once("k", lambda: 3), 3)

    def test_threads_run_it_once(self):
        led, calls, results = ToolLedger(), [], []

        def slow():
            calls.append(1)
            time.sleep(0.2)
            return "done"

        def worker():
            try:
                results.append(led.once("k", slow))
            except InDoubt:
                results.append("in doubt")

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(calls, [1])
        self.assertEqual(results.count("done"), 1)
        self.assertEqual(results.count("in doubt"), 7)

    def test_holds_across_a_restart(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.db")
            ToolLedger(path).once("k", lambda: {"id": 1})
            self.assertEqual(ToolLedger(path).once("k", lambda: {"id": 2}), {"id": 1})


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class PacerTest(unittest.TestCase):
    def test_requests_per_minute(self):
        clock = FakeClock()
        p = Pacer(requests_per_minute=2, clock=clock, sleep=clock.sleep)
        self.assertEqual(p.acquire(), 0)
        clock.t = 10
        self.assertEqual(p.acquire(), 0)
        clock.t = 20
        self.assertAlmostEqual(p.acquire(), 40)  # until the first one leaves the window

    def test_tokens_per_minute(self):
        clock = FakeClock()
        p = Pacer(tokens_per_minute=10_000, clock=clock, sleep=clock.sleep)
        p.acquire(6000)
        self.assertGreater(p.acquire(6000), 0)

    def test_a_request_bigger_than_the_limit_goes_alone(self):
        clock = FakeClock()
        p = Pacer(tokens_per_minute=1000, clock=clock, sleep=clock.sleep)
        self.assertEqual(p.acquire(5000), 0)

    def test_needs_a_limit(self):
        with self.assertRaises(ValueError):
            Pacer()


class BudgetTest(unittest.TestCase):
    def test_reserve_holds_under_concurrency(self):
        b = Budget(1.0)
        held = [b.reserve(0.3) for _ in range(3)]
        with self.assertRaises(BudgetExceeded):
            b.reserve(0.3)
        b.settle(held[0], 0.01)
        b.reserve(0.3)

    def test_price(self):
        p = Price(input=1.0, output=5.0, cached_input=0.1)
        self.assertAlmostEqual(p.cost({"input": 1000, "cached_input": 400, "output": 100}), (600 + 40 + 500) / 1e6)
        self.assertAlmostEqual(Price(1.0, 5.0).cost({"input": 1000, "cached_input": 400, "output": 0}), 1000 / 1e6)


if __name__ == "__main__":
    unittest.main()
