"""Request bodies, response parsing and follow-up messages for three APIs: Anthropic Messages, OpenAI Chat
Completions, DeepSeek (OpenAI-compatible, with its own quirks). Plain dicts in, plain dicts out.

Four ways to ask for structured output ("mechanisms"):
  prompt     the schema is in the system prompt, the answer is text;
  json_mode  as prompt, plus the API's JSON mode (OpenAI, DeepSeek; Anthropic has none);
  tool       a tool whose parameters are the schema, the model is made to call it;
  strict     the API enforces the schema (OpenAI json_schema strict, Anthropic output_config.format,
             DeepSeek strict tools on its beta endpoint).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

PROVIDERS = ("anthropic", "openai", "deepseek")
MECHANISMS = ("prompt", "json_mode", "tool", "strict")
DEFAULT_BASE = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
}
KEY_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}
SCHEMA_NOTE = "\n\nReply with only a JSON object that matches this JSON Schema:\n{schema}"
ASK_TOOL = "Give the answer by calling {tool}."
FEEDBACK_TEXT = "That answer could not be used: {problems}. Reply with only the corrected JSON object."
FEEDBACK_TOOL = "That call could not be used: {problems}. Call {tool} again with corrected arguments."
STOPPED = {"insufficient_system_resource", "aborted", "content_filter", "pause_turn"}
TRUNCATED = {"length", "max_tokens", "model_context_window_exceeded"}


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _l(x) -> list:
    return x if isinstance(x, list) else []


def _n(x) -> int:
    """A token count as an int: 0 for anything that is not a number."""
    if isinstance(x, bool):
        return 0
    if isinstance(x, (int, float)):
        return int(x)
    try:
        return int(str(x).strip())
    except ValueError:
        return 0


@dataclass
class Ask:
    """One structured request, independent of the API."""

    system: str
    user: str
    schema: dict
    mechanism: str = "tool"
    tool_name: str = "submit"
    tool_description: str = "Submit the answer."
    max_tokens: int = 1024
    extra_body: dict = field(default_factory=dict)


@dataclass
class Reply:
    text: str | None
    obj: dict | None  # input the API already parsed (an Anthropic tool call)
    finish: str
    tool_called: bool | None  # None when no tool was asked for
    usage: dict


def check_mechanism(provider: str, mechanism: str) -> None:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider {provider!r}; one of {', '.join(PROVIDERS)}")
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}; one of {', '.join(MECHANISMS)}")
    if provider == "anthropic" and mechanism == "json_mode":
        raise ValueError("the Anthropic API has no JSON mode; use tool or strict")


def uses_tool(provider: str, ask: Ask) -> bool:
    return ask.mechanism == "tool" or (ask.mechanism == "strict" and provider == "deepseek")


def thinking_on(provider: str, ask: Ask) -> bool:
    t = (ask.extra_body.get("thinking") or {}).get("type")
    if provider == "deepseek":
        return t != "disabled"  # on by default
    return provider == "anthropic" and t in ("enabled", "adaptive")


def tool_is_optional(provider: str, ask: Ask) -> bool:
    """With thinking on, DeepSeek and Anthropic refuse a forced tool choice (HTTP 400). There the tool is
    offered and the user message asks for it."""
    return uses_tool(provider, ask) and thinking_on(provider, ask)


def endpoint(provider: str, ask: Ask, base_url: str | None = None, api_key: str | None = None):
    """-> (url, headers). The key is read from the environment if not given."""
    base = (base_url or os.environ.get(f"{provider.upper()}_BASE_URL") or DEFAULT_BASE[provider]).rstrip("/")
    key = api_key if api_key is not None else os.environ.get(KEY_ENV[provider], "")
    if provider == "anthropic":
        return f"{base}/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}
    if provider == "deepseek" and ask.mechanism == "strict":
        return f"{base}/beta/chat/completions", {"authorization": f"Bearer {key}"}
    return f"{base}/chat/completions", {"authorization": f"Bearer {key}"}


def build(provider: str, model: str, ask: Ask, extra=(), max_tokens: int | None = None) -> dict:
    """The request body. `extra` continues the conversation (a follow-up after an unusable answer)."""
    check_mechanism(provider, ask.mechanism)
    limit = max_tokens or ask.max_tokens
    m = ask.mechanism
    system = ask.system
    if m in ("prompt", "json_mode"):
        system += SCHEMA_NOTE.format(schema=json.dumps(ask.schema))
    user = ask.user
    if tool_is_optional(provider, ask):
        user += "\n\n" + ASK_TOOL.format(tool=ask.tool_name)
    if provider == "anthropic":
        body = {
            "model": model,
            "max_tokens": limit,
            "system": system,
            "messages": [{"role": "user", "content": user}, *extra],
        }
        if m == "tool":
            body["tools"] = [{"name": ask.tool_name, "description": ask.tool_description, "input_schema": ask.schema}]
            forced = {"type": "tool", "name": ask.tool_name}
            body["tool_choice"] = {"type": "auto"} if tool_is_optional(provider, ask) else forced
        elif m == "strict":
            body["output_config"] = {"format": {"type": "json_schema", "schema": ask.schema}}
        body.update(ask.extra_body)
        return body
    body = {
        "model": model,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}, *extra],
    }
    body["max_completion_tokens" if provider == "openai" else "max_tokens"] = limit
    if m == "json_mode":
        body["response_format"] = {"type": "json_object"}
    elif m == "strict" and provider == "openai":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": ask.tool_name, "strict": True, "schema": ask.schema},
        }
    elif uses_tool(provider, ask):
        fn = {"name": ask.tool_name, "description": ask.tool_description, "parameters": ask.schema}
        if m == "strict":
            fn["strict"] = True
        body["tools"] = [{"type": "function", "function": fn}]
        forced = {"type": "function", "function": {"name": ask.tool_name}}
        body["tool_choice"] = "auto" if tool_is_optional(provider, ask) else forced
    body.update(ask.extra_body)
    return body


def parse(provider: str, ask: Ask, data: dict) -> Reply:
    if provider == "anthropic":
        blocks = [_d(b) for b in _l(data.get("content"))]
        text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text" and isinstance(b.get("text"), str))
        obj, called = None, None
        if ask.mechanism == "tool":
            called = False
            for b in blocks:
                if b.get("type") == "tool_use" and b.get("name") == ask.tool_name:
                    obj, called = b.get("input"), True
                    break
        u = _d(data.get("usage"))
        read = _n(u.get("cache_read_input_tokens"))
        written = _n(u.get("cache_creation_input_tokens"))
        usage = {
            "input": _n(u.get("input_tokens")) + read + written,
            "cached_input": read,
            "output": _n(u.get("output_tokens")),
            "reasoning": 0,
        }
        return Reply(text, obj, data.get("stop_reason") or "", called, usage)
    choice = _d((_l(data.get("choices")) or [{}])[0])
    msg = _d(choice.get("message"))
    finish = choice.get("finish_reason") or ""
    if msg.get("refusal"):
        finish = "refusal"
    text, called = msg.get("content"), None
    if uses_tool(provider, ask):
        called = False
        for c in _l(msg.get("tool_calls")):
            fn = _d(_d(c).get("function"))
            if fn.get("name") == ask.tool_name:
                text, called = fn.get("arguments"), True
                break
    if not isinstance(text, str):
        text = None
    u = _d(data.get("usage"))
    if provider == "deepseek":
        cached = _n(u.get("prompt_cache_hit_tokens"))
    else:
        cached = _n(_d(u.get("prompt_tokens_details")).get("cached_tokens"))
    usage = {
        "input": _n(u.get("prompt_tokens")),
        "cached_input": cached,
        "output": _n(u.get("completion_tokens")),
        "reasoning": _n(_d(u.get("completion_tokens_details")).get("reasoning_tokens")),
    }
    return Reply(text, None, finish, called, usage)


def follow_up(provider: str, ask: Ask, raw: dict, problems: list, category: str) -> list:
    """The conversation after an unusable answer: the answer itself, then what was wrong with it."""
    said = "; ".join(problems)[:600] or "the answer was not usable"
    tool = ask.tool_name
    if provider == "anthropic":
        blocks = _l(raw.get("content"))
        if not blocks:
            return []  # an empty assistant turn is refused by the API: send the request again as it was
        echo = {"role": "assistant", "content": blocks}
        uses = [b for b in blocks if b.get("type") == "tool_use"]
        if uses:
            results = [
                {
                    "type": "tool_result",
                    "tool_use_id": b.get("id"),
                    "is_error": True,
                    "content": FEEDBACK_TOOL.format(problems=said, tool=tool) if b.get("name") == tool else "ignored",
                }
                for b in uses
            ]
            return [echo, {"role": "user", "content": results}]
        ask_again = ASK_TOOL.format(tool=tool) if category == "no_tool_call" else FEEDBACK_TEXT.format(problems=said)
        return [echo, {"role": "user", "content": ask_again}]
    msg = ((raw.get("choices") or [{}])[0].get("message")) or {}
    echo = {"role": "assistant", "content": msg.get("content")}
    if provider == "deepseek" and msg.get("reasoning_content"):
        echo["reasoning_content"] = msg["reasoning_content"]  # DeepSeek answers 400 if it is left out
    calls = msg.get("tool_calls") or []
    if calls:
        echo["tool_calls"] = calls
        return [echo] + [
            {
                "role": "tool",
                "tool_call_id": c.get("id"),
                "content": FEEDBACK_TOOL.format(problems=said, tool=tool) if (c.get("function") or {}).get("name") == tool else "ignored",
            }
            for c in calls
        ]
    if echo["content"] is None:
        echo["content"] = ""
    if category == "no_tool_call":
        return [echo, {"role": "user", "content": ASK_TOOL.format(tool=tool)}]
    return [echo, {"role": "user", "content": FEEDBACK_TEXT.format(problems=said)}]
