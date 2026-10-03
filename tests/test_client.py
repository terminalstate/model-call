import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fake_api import FakeAPI, anthropic_text, anthropic_tool, openai_text, openai_tool  # noqa: E402
from test_schema_repair import GOOD, SCHEMA  # noqa: E402

from model_call import AuthError, Budget, BudgetExceeded, CallLog, Client, Price, RetryPolicy, transport  # noqa: E402

CHAT, BETA, MESSAGES = "/chat/completions", "/beta/chat/completions", "/v1/messages"
KEY = "sk-test-0123456789abcdef"
FAST = RetryPolicy(max_attempts=2, backoff_base=0.01, backoff_cap=0.02, timeout_s=5, deadline_s=5)


def real_date(obj):
    d = obj.get("effective_date")
    return [] if d is None or d[5:7] <= "12" else [f"effective_date {d} is not a real date"]


class ClientTest(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        self._sleep = transport.sleep
        transport.sleep = lambda s: None

    def tearDown(self):
        transport.sleep = self._sleep
        self.api.close()

    def client(self, provider="openai", **kw):
        kw.setdefault("retry", FAST)
        return Client(provider, kw.pop("model", "m"), base_url=self.api.url, api_key=KEY, **kw)

    # -------------------------------------------------------------- first answer fine

    def test_openai_tool_ok(self):
        self.api.push(CHAT, openai_tool(GOOD))
        res = self.client(price=Price(1.0, 5.0, 0.1)).structured("sys", "doc", SCHEMA)
        self.assertTrue(res.ok and res.complete)
        self.assertEqual((res.category, res.requests, res.recovered_by), ("ok", 1, None))
        self.assertEqual(res.value, GOOD)
        self.assertAlmostEqual(res.cost_usd, (800 * 1.0 + 200 * 0.1 + 100 * 5.0) / 1e6)
        body = self.api.bodies(CHAT)[0]
        self.assertEqual(body["tool_choice"], {"type": "function", "function": {"name": "submit"}})
        self.assertEqual(body["tools"][0]["function"]["parameters"], SCHEMA)
        self.assertEqual(body["max_completion_tokens"], 1024)

    def test_openai_strict_and_json_mode_bodies(self):
        self.api.push(CHAT, openai_text(json.dumps(GOOD)), openai_text(json.dumps(GOOD)))
        self.assertTrue(self.client(mechanism="strict").structured("sys", "doc", SCHEMA).ok)
        self.assertTrue(self.client(mechanism="json_mode").structured("sys", "doc", SCHEMA).ok)
        strict, jm = self.api.bodies(CHAT)
        self.assertEqual(strict["response_format"]["json_schema"]["strict"], True)
        self.assertEqual(jm["response_format"], {"type": "json_object"})
        self.assertIn("JSON Schema", jm["messages"][0]["content"])  # the schema is in the prompt

    def test_anthropic_strict_uses_output_config(self):
        self.api.push(MESSAGES, anthropic_text(json.dumps(GOOD)))
        res = self.client("anthropic", mechanism="strict").structured("sys", "doc", SCHEMA)
        self.assertTrue(res.ok)
        body = self.api.bodies(MESSAGES)[0]
        self.assertEqual(body["output_config"]["format"]["schema"], SCHEMA)
        self.assertEqual(res.usage["cached_input"], 500)

    def test_anthropic_has_no_json_mode(self):
        with self.assertRaises(ValueError):
            self.client("anthropic", mechanism="json_mode")

    # -------------------------------------------------------------- repaired locally, no second request

    def test_fenced_answer_is_repaired_locally(self):
        self.api.push(MESSAGES, anthropic_text("```json\n" + json.dumps(GOOD) + "\n```"))
        res = self.client("anthropic", mechanism="prompt").structured("sys", "doc", SCHEMA)
        self.assertTrue(res.ok)
        self.assertEqual((res.category, res.recovered_by, res.requests), ("wrapped", "local", 1))

    def test_year_is_repaired_locally(self):
        self.api.push(CHAT, openai_text(json.dumps({**GOOD, "term": {"number": 1, "unit": "year"}})))
        res = self.client(mechanism="json_mode").structured("sys", "doc", SCHEMA)
        self.assertTrue(res.ok)
        self.assertEqual((res.category, res.recovered_by, res.requests), ("schema", "local", 1))
        self.assertEqual(res.value["term"]["unit"], "years")

    # -------------------------------------------------------------- one more request

    def test_wrong_content_gets_feedback(self):
        self.api.push(CHAT, openai_tool({**GOOD, "parties": 42}), openai_tool(GOOD, call_id="call_2"))
        res = self.client().structured("sys", "doc", SCHEMA)
        self.assertTrue(res.ok)
        self.assertEqual((res.category, res.recovered_by, res.requests), ("schema", "feedback", 2))
        second = self.api.bodies(CHAT)[1]["messages"]
        self.assertEqual(second[2]["tool_calls"][0]["id"], "call_1")  # the answer goes back
        self.assertEqual(second[3]["role"], "tool")
        self.assertIn("parties: expected array", second[3]["content"])

    def test_value_rule_gets_feedback(self):
        self.api.push(CHAT, openai_tool({**GOOD, "effective_date": "2001-18-04"}), openai_tool(GOOD))
        res = self.client().structured("sys", "doc", SCHEMA, check=real_date)
        self.assertTrue(res.ok)
        self.assertEqual((res.category, res.recovered_by), ("value", "feedback"))
        self.assertIn("not a real date", self.api.bodies(CHAT)[1]["messages"][3]["content"])

    def test_repaired_answer_still_checked_by_value_rules(self):
        bad = {**GOOD, "term": {"number": 1, "unit": "year"}, "effective_date": "2001-18-04"}
        self.api.push(CHAT, openai_tool(bad), openai_tool(GOOD))
        res = self.client().structured("sys", "doc", SCHEMA, check=real_date)
        self.assertEqual((res.category, res.recovered_by, res.requests), ("schema", "feedback", 2))
        feedback = self.api.bodies(CHAT)[1]["messages"][3]["content"]
        self.assertIn("'year' is not one of", feedback)
        self.assertIn("not a real date", feedback)

    def test_retryable_4xx_to_the_end_is_not_rejected(self):
        self.api.push(CHAT, *[{"status": 429, "body": {}}] * 2)
        self.assertEqual(self.client().structured("s", "d", SCHEMA).category, "http_failed")

    def test_anthropic_tool_feedback_is_a_tool_result(self):
        self.api.push(MESSAGES, anthropic_tool({**GOOD, "jurisdiction": 7}), anthropic_tool(GOOD))
        res = self.client("anthropic").structured("sys", "doc", SCHEMA)
        self.assertEqual(res.recovered_by, "feedback")
        msgs = self.api.bodies(MESSAGES)[1]["messages"]
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[2]["content"][0]["type"], "tool_result")
        self.assertTrue(msgs[2]["content"][0]["is_error"])
        self.assertEqual(msgs[2]["content"][0]["tool_use_id"], "toolu_1")

    def test_cut_off_gets_a_bigger_limit(self):
        self.api.push(CHAT, openai_text("", finish="length", reasoning_content="thinking..."), openai_text(json.dumps(GOOD)))
        res = self.client("deepseek", mechanism="json_mode", max_tokens=1000).structured("sys", "doc", SCHEMA)
        self.assertTrue(res.ok)
        self.assertEqual((res.category, res.recovered_by), ("truncated", "bigger"))
        self.assertEqual([b["max_tokens"] for b in self.api.bodies(CHAT)], [1000, 4000])

    def test_empty_answer_is_asked_again(self):
        self.api.push(CHAT, openai_text(""), openai_text(json.dumps(GOOD)))
        res = self.client(mechanism="json_mode").structured("sys", "doc", SCHEMA)
        self.assertEqual((res.category, res.recovered_by, res.requests), ("empty", "retry", 2))
        self.assertEqual(len(self.api.bodies(CHAT)[1]["messages"]), 2)  # the same request, not a conversation

    def test_deepseek_feedback_keeps_reasoning(self):
        self.api.push(CHAT, openai_text("not json at all", reasoning_content="r1"), openai_text(json.dumps(GOOD)))
        res = self.client("deepseek", mechanism="json_mode").structured("sys", "doc", SCHEMA)
        self.assertEqual((res.category, res.recovered_by), ("invalid_json", "feedback"))
        echo = self.api.bodies(CHAT)[1]["messages"][2]
        self.assertEqual(echo["reasoning_content"], "r1")

    def test_deepseek_tool_with_thinking_is_offered_not_forced(self):
        self.api.push(CHAT, openai_tool(GOOD), openai_tool(GOOD))
        self.client("deepseek").structured("sys", "doc", SCHEMA)
        self.client("deepseek", extra_body={"thinking": {"type": "disabled"}}).structured("sys", "doc", SCHEMA)
        thinking, plain = self.api.bodies(CHAT)
        self.assertEqual(thinking["tool_choice"], "auto")
        self.assertIn("calling submit", thinking["messages"][1]["content"])
        self.assertEqual(plain["tool_choice"]["function"]["name"], "submit")
        self.assertEqual(plain["thinking"], {"type": "disabled"})

    def test_deepseek_strict_goes_to_beta(self):
        self.api.push(BETA, openai_tool(GOOD))
        res = self.client("deepseek", mechanism="strict", extra_body={"thinking": {"type": "disabled"}}).structured("s", "d", SCHEMA)
        self.assertTrue(res.ok)
        self.assertTrue(self.api.bodies(BETA)[0]["tools"][0]["function"]["strict"])

    # -------------------------------------------------------------- failures

    def test_still_wrong_after_feedback(self):
        bad = {**GOOD, "parties": 42}
        self.api.push(CHAT, openai_tool(bad), openai_tool(bad))
        res = self.client().structured("sys", "doc", SCHEMA)
        self.assertFalse(res.ok)
        self.assertEqual(res.requests, 2)
        self.assertIn("parties: expected array", res.problems[0])

    def test_partial_answer_only_when_allowed(self):
        bad = {**GOOD, "effective_date": "April 18", "parties": 42}
        for allowed in (False, True):
            self.api.push(CHAT, openai_tool(bad), openai_tool({**GOOD, "effective_date": "April 18"}))
            res = self.client().structured("sys", "doc", SCHEMA, allow_partial=allowed)
            self.assertEqual(res.ok, allowed)
            self.assertFalse(res.complete)
            self.assertEqual(res.dropped, ["effective_date"])
            if allowed:
                self.assertIsNone(res.value["effective_date"])
                self.assertEqual(res.value["parties"], GOOD["parties"])
            else:
                self.assertIsNone(res.value)  # not handed out unless asked for

    def test_400_is_reported_without_a_second_request(self):
        self.api.push(CHAT, {"status": 400, "body": {"error": {"message": "tools not supported with reasoning_effort"}}})
        res = self.client().structured("sys", "doc", SCHEMA)
        self.assertEqual((res.ok, res.category, res.requests), (False, "rejected", 1))
        self.assertIn("reasoning_effort", res.problems[0])

    def test_server_errors_to_the_end(self):
        self.api.push(CHAT, *[{"status": 503, "body": {}}] * 2)
        res = self.client().structured("sys", "doc", SCHEMA)
        self.assertEqual((res.ok, res.category, len(res.attempts)), (False, "http_failed", 2))

    # -------------------------------------------------------------- money, log

    def test_budget_refuses_before_sending(self):
        c = self.client(price=Price(1.0, 5.0), budget=Budget(0.001), max_tokens=1024)
        with self.assertRaises(BudgetExceeded):
            c.structured("sys", "doc", SCHEMA)  # 1024 output tokens alone could cost $0.005
        self.assertEqual(self.api.requests, [])

    def test_budget_is_settled_at_the_real_cost(self):
        self.api.push(CHAT, openai_tool(GOOD))
        b = Budget(1.0)
        res = self.client(price=Price(1.0, 5.0, 0.1), budget=b).structured("sys", "doc", SCHEMA)
        self.assertAlmostEqual(b.spent, res.cost_usd)
        self.assertAlmostEqual(b.left, 1.0 - res.cost_usd)

    def test_timed_out_request_is_charged_at_worst_case(self):
        self.api.push(CHAT, {"delay": 0.5, **openai_tool(GOOD)}, {"delay": 0.5, **openai_tool(GOOD)})
        b = Budget(1.0)
        p = Price(1.0, 5.0)
        res = self.client(price=p, budget=b, retry=RetryPolicy(timeout_s=0.1, deadline_s=0.1, max_attempts=2)).structured("s", "d", SCHEMA)
        self.assertEqual(res.category, "http_failed")
        self.assertGreater(b.spent, 2 * 1024 * 5.0 / 1e6)  # two attempts, each may have been billed

    def test_resend_after_timeout_stays_under_the_ceiling(self):
        self.api.push(CHAT, {"delay": 0.5, **openai_tool(GOOD)}, {"delay": 0.5, **openai_tool(GOOD)})
        p = Price(1.0, 5.0)
        c = self.client(price=p, retry=RetryPolicy(timeout_s=0.1, deadline_s=0.1, max_attempts=3))
        body_worst = 1024 * 5.0 / 1e6
        b = Budget(body_worst * 1.6)  # room for one worst case, not two
        c.budget = b
        res = c.structured("s", "d", SCHEMA)
        self.assertEqual(len(res.attempts), 1)  # the resend was not allowed
        self.assertLessEqual(b.spent, b.max_usd)
        self.assertAlmostEqual(b.left, b.max_usd - b.spent)

    def test_no_hold_is_left_behind(self):
        b = Budget(1.0)
        c = self.client(price=Price(1.0, 5.0), budget=b)
        self.api.push(CHAT, {"status": 200, "body": {"choices": [None]}}, openai_tool(GOOD))
        res = c.structured("s", "d", SCHEMA)
        self.assertEqual((res.category, res.recovered_by), ("empty", "retry"))
        self.api.push(CHAT, {"delay": 0.3, **openai_tool(GOOD)}, {"status": 401, "body": {}})
        c.retry = RetryPolicy(timeout_s=0.1, deadline_s=0.1)
        with self.assertRaises(AuthError):
            c.structured("s", "d", SCHEMA)
        self.assertAlmostEqual(b.left, 1.0 - b.spent)  # nothing still held
        self.assertGreater(b.spent, 1024 * 5.0 / 1e6)  # the timed-out attempt counts

    def test_anthropic_empty_tool_answer_is_asked_again_as_is(self):
        self.api.push(MESSAGES, {"status": 200, "body": {"content": [], "stop_reason": "end_turn", "usage": {}}}, anthropic_tool(GOOD))
        res = self.client("anthropic", extra_body={"thinking": {"type": "enabled", "budget_tokens": 1024}}, max_tokens=2048).structured(
            "s", "d", SCHEMA
        )
        self.assertEqual((res.category, res.recovered_by), ("empty", "retry"))
        second = self.api.bodies(MESSAGES)[1]
        self.assertEqual(len(second["messages"]), 1)
        self.assertEqual(second["tool_choice"], {"type": "auto"})

    def test_refused_request_takes_no_rate_slot(self):
        from model_call import Pacer

        p = Pacer(requests_per_minute=1, sleep=lambda s: self.fail("should not wait"))
        c = self.client(price=Price(1.0, 5.0), budget=Budget(0.001), pacer=p)
        with self.assertRaises(BudgetExceeded):
            c.structured("s", "d", SCHEMA)
        self.assertEqual(p.acquire(), 0)

    def test_token_counts_as_strings(self):
        self.api.push(CHAT, openai_tool(GOOD, usage={"prompt_tokens": "1000", "completion_tokens": "100"}))
        res = self.client(price=Price(1.0, 5.0)).structured("s", "d", SCHEMA)
        self.assertTrue(res.ok)
        self.assertEqual(res.usage["input"], 1000)

    def test_log_has_the_call_and_no_key(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "calls.jsonl")
            self.api.push(CHAT, openai_tool({**GOOD, "parties": 1}), openai_tool(GOOD))
            self.client(log=CallLog(path, secrets=[KEY])).structured("sys", "doc", SCHEMA)
            text = Path(path).read_text()
            rec = json.loads(text)
            self.assertNotIn(KEY, text)
            self.assertEqual(len(rec["requests"]), 2)
            self.assertIn("body", rec["requests"][0])
            self.assertEqual(rec["recovered_by"], "feedback")
            self.api.push(CHAT, openai_tool(GOOD))
            self.client(log=CallLog(path, content=False)).structured("sys", "doc", SCHEMA)
            lean = json.loads(Path(path).read_text().splitlines()[1])
            self.assertNotIn("body", lean["requests"][0])
            self.assertNotIn("value", lean)
            self.api.push(CHAT, openai_tool({**GOOD, "jurisdiction": 5}), openai_tool({**GOOD, "jurisdiction": 5}))
            self.client(log=CallLog(path, content=False)).structured("sys", "doc", SCHEMA)
            third = Path(path).read_text().splitlines()[2]
            self.assertNotIn("problems", json.loads(third))  # problem texts can quote the answer

    def test_schema_root_must_be_an_object(self):
        with self.assertRaises(ValueError):
            self.client().structured("s", "d", {"type": "array"})


if __name__ == "__main__":
    unittest.main()
