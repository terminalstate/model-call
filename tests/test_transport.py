import sys
import time
import unittest
from email.utils import formatdate
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_api import FakeAPI  # noqa: E402

from model_call import transport  # noqa: E402
from model_call.transport import AuthError, BalanceError, NotFoundError, RetryPolicy, retry_after, send  # noqa: E402

OK = {"status": 200, "body": {"ok": True}}


class TransportTest(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        self.url = self.api.url + "/x"
        self.slept = []
        self._sleep = transport.sleep
        transport.sleep = self.slept.append

    def tearDown(self):
        transport.sleep = self._sleep
        self.api.close()

    def test_ok(self):
        self.api.push("/x", OK)
        out = send(self.url, {}, {}, RetryPolicy())
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(len(out.attempts), 1)

    def test_429_waits_at_least_retry_after(self):
        self.api.push("/x", {"status": 429, "headers": {"retry-after": "7"}, "body": {}}, OK)
        out = send(self.url, {}, {}, RetryPolicy(backoff_base=0.1))
        self.assertEqual(out.data, {"ok": True})
        self.assertGreaterEqual(self.slept[0], 7)
        self.assertEqual(out.attempts[0]["headers"]["retry-after"], "7")

    def test_retry_after_forms(self):
        self.assertEqual(retry_after({"retry-after-ms": "1500"}), 1.5)
        self.assertEqual(retry_after({"retry-after": "2"}), 2.0)
        self.assertAlmostEqual(retry_after({"retry-after": formatdate(time.time() + 30, usegmt=True)}), 30, delta=2)
        self.assertIsNone(retry_after({"retry-after": "soon"}))
        self.assertIsNone(retry_after({}))

    def test_backoff_without_retry_after_and_give_up(self):
        self.api.push("/x", *[{"status": 503, "body": {}}] * 4)
        out = send(self.url, {}, {}, RetryPolicy(max_attempts=4, backoff_base=1, backoff_cap=3))
        self.assertIsNone(out.data)
        self.assertEqual(len(out.attempts), 4)
        self.assertEqual(len(self.slept), 3)
        self.assertTrue(all(0.5 <= s <= 3 for s in self.slept))

    def test_400_is_not_retried(self):
        self.api.push("/x", {"status": 400, "body": {"error": {"message": "bad tool_choice"}}}, OK)
        out = send(self.url, {}, {}, RetryPolicy())
        self.assertIsNone(out.data)
        self.assertEqual(out.status, 400)
        self.assertIn("bad tool_choice", out.detail)
        self.assertEqual(len(out.attempts), 1)

    def test_fatal_statuses_raise(self):
        for status, exc in ((401, AuthError), (403, AuthError), (402, BalanceError), (404, NotFoundError)):
            self.api.push("/x", {"status": status, "body": {}})
            with self.assertRaises(exc):
                send(self.url, {}, {}, RetryPolicy())

    def test_ids_are_masked(self):
        msg = "Rate limit reached for gpt in organization org-AbC123xyz on tokens per min. req_9f8e7d"
        self.api.push("/x", {"status": 400, "body": {"error": {"message": msg}}})
        out = send(self.url, {}, {}, RetryPolicy())
        self.assertNotIn("org-AbC123xyz", out.detail)
        self.assertNotIn("req_9f8e7d", out.detail)
        self.assertIn("org-…", out.detail)

    def test_timed_out_request_is_sent_again_once(self):
        self.api.push("/x", *[{"delay": 0.6, **OK}] * 4)
        out = send(self.url, {}, {}, RetryPolicy(timeout_s=0.2, deadline_s=0.2, max_attempts=4))
        self.assertIsNone(out.data)
        self.assertEqual(len(out.attempts), 2)
        self.assertEqual(out.maybe_billed, 2)

    def test_deadline_beats_keepalive_bytes(self):
        self.api.push("/x", {"status": 200, "body": {"ok": True}, "trickle": {"every": 0.05, "for": 3}})
        t0 = time.monotonic()
        out = send(self.url, {}, {}, RetryPolicy(timeout_s=1, deadline_s=0.4, max_attempts=1))
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertIsNone(out.data)
        self.assertEqual(out.attempts[0]["kind"], "deadline")

    def test_keepalive_bytes_then_body_is_fine(self):
        self.api.push("/x", {"status": 200, "body": {"ok": True}, "trickle": {"every": 0.05, "for": 0.2}})
        out = send(self.url, {}, {}, RetryPolicy(timeout_s=1, deadline_s=5))
        self.assertEqual(out.data, {"ok": True})
        self.assertGreater(out.attempts[0]["keepalive_bytes"], 0)

    def test_total_time_stops_retries(self):
        self.api.push("/x", *[{"status": 429, "headers": {"retry-after": "20"}, "body": {}}] * 3)
        out = send(self.url, {}, {}, RetryPolicy(total_s=10, backoff_cap=30))
        self.assertEqual(len(out.attempts), 1)  # waiting 20 s would pass the 10 s allowed
        self.assertEqual(self.slept, [])

    def test_retry_after_above_our_cap_is_honoured(self):
        self.api.push("/x", {"status": 429, "headers": {"retry-after": "90"}, "body": {}}, OK)
        out = send(self.url, {}, {}, RetryPolicy(backoff_cap=30))
        self.assertEqual(out.data, {"ok": True})
        self.assertGreaterEqual(self.slept[0], 90)

    def test_retry_after_beyond_reason_ends_the_call(self):
        self.api.push("/x", {"status": 503, "headers": {"retry-after": "3600"}, "body": {}}, OK)
        out = send(self.url, {}, {}, RetryPolicy())
        self.assertIsNone(out.data)
        self.assertEqual(self.slept, [])
        self.assertIn("max_retry_after", out.detail)

    def test_after_a_timeout_one_more_attempt_whatever_it_returns(self):
        self.api.push("/x", {"delay": 0.6, **OK}, {"status": 503, "body": {}}, {"delay": 0.6, **OK}, OK)
        out = send(self.url, {}, {}, RetryPolicy(timeout_s=0.2, deadline_s=0.2, max_attempts=4))
        self.assertEqual([a["status"] for a in out.attempts], [None, 503])

    def test_broken_200_may_be_billed(self):
        self.api.push("/x", *[{"status": 200, "raw": '{"choices": [{"mess'}] * 3)
        out = send(self.url, {}, {}, RetryPolicy(max_attempts=4))
        self.assertEqual(len(out.attempts), 2)  # processed once, sent again once
        self.assertEqual(out.maybe_billed, 2)

    def test_resend_needs_permission(self):
        self.api.push("/x", {"delay": 0.6, **OK}, OK)
        out = send(self.url, {}, {}, RetryPolicy(timeout_s=0.2, deadline_s=0.2), before_resend=lambda: False)
        self.assertEqual(len(out.attempts), 1)
        self.assertIn("budget", out.detail)

    def test_non_object_body_is_bad(self):
        self.api.push("/x", {"status": 200, "raw": "[]"}, OK)
        out = send(self.url, {}, {}, RetryPolicy())
        self.assertEqual(out.data, {"ok": True})

    def test_fatal_error_carries_attempts(self):
        self.api.push("/x", {"delay": 0.6, **OK}, {"status": 401, "body": {}})
        with self.assertRaises(AuthError) as e:
            send(self.url, {}, {}, RetryPolicy(timeout_s=0.2, deadline_s=0.2))
        self.assertEqual(e.exception.sent.maybe_billed, 1)

    def test_bad_body_is_retried(self):
        self.api.push("/x", {"status": 200, "raw": "<html>gateway</html>"}, OK)
        out = send(self.url, {}, {}, RetryPolicy())
        self.assertEqual(out.data, {"ok": True})
        self.assertEqual(out.attempts[0]["kind"], "bad response body")


if __name__ == "__main__":
    unittest.main()
