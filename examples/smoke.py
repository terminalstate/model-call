#!/usr/bin/env python3
"""A few real calls to each API you have a key for, under a $0.05 ceiling. Not a measurement: a check that
the request bodies, the parsing and the recovery paths work against the real services, not only the fakes.

  python3 examples/smoke.py                       # keys from the environment
  python3 examples/smoke.py --keys path/to/keys.env   # lines NAME=value: OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY

Writes smoke-calls.jsonl next to where it is run (keys are never written). Model names and prices are the
ones checked on 2026-09-29; pass --models to change them.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_call import Budget, CallLog, Client, ModelCallError, Price  # noqa: E402

AGREEMENT = """MUTUAL NON-DISCLOSURE AGREEMENT

This Agreement is made on the 3rd day of March, 2025 between Acme Scheduling Ltd, a company registered in
England and Wales ("Acme"), and Jane Roe, an individual residing in Leeds ("the Recipient").

1. The Recipient shall keep confidential all information disclosed by Acme in connection with the
proposed evaluation of Acme's scheduling software.
2. The obligations in clause 1 continue for a period of three (3) years from the date of this Agreement.
3. This Agreement is governed by the laws of England and Wales.
"""
SYSTEM = "You extract fields from agreements. Use only what the text states. Dates as YYYY-MM-DD. A field the text does not state is null."
SCHEMA = {
    "type": "object",
    "properties": {
        "parties": {"type": "array", "items": {"type": "string"}},
        "effective_date": {"type": ["string", "null"]},
        "jurisdiction": {"type": ["string", "null"]},
        "term": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "number": {"type": "number"},
                        "unit": {"type": "string", "enum": ["days", "weeks", "months", "years"]},
                    },
                    "required": ["number", "unit"],
                    "additionalProperties": False,
                },
            ]
        },
    },
    "required": ["parties", "effective_date", "jurisdiction", "term"],
    "additionalProperties": False,
}
MODELS = {
    "anthropic": ("claude-haiku-4-5", Price(1.00, 5.00, 0.10)),
    "openai": ("gpt-6-luna", Price(0.10, 0.50, 0.01)),
    "deepseek": ("deepseek-flash", Price(0.30, 1.20, 0.006)),
}
NO_THINKING = {"thinking": {"type": "disabled"}}
# (provider, mechanism, extra_body, max_tokens, what it shows)
CASES = [
    ("anthropic", "prompt", {}, 600, "schema in the prompt: expect a code fence, repaired locally"),
    ("anthropic", "tool", {}, 600, "forced tool call"),
    ("anthropic", "strict", {}, 600, "output_config.format"),
    ("openai", "json_mode", {"reasoning_effort": "low"}, 2000, "JSON mode"),
    ("openai", "strict", {"reasoning_effort": "low"}, 2000, "json_schema strict"),
    ("openai", "tool", {"reasoning_effort": "none"}, 600, "forced function call"),
    ("deepseek", "json_mode", NO_THINKING, 600, "JSON mode, thinking off"),
    ("deepseek", "tool", {}, 2000, "thinking on: tool offered, not forced"),
    ("deepseek", "json_mode", {}, 250, "thinking on, 250-token limit: may be cut off, then 8x"),
]


def real_date(obj) -> list:
    """A value rule the schema cannot say: the date is a real YYYY-MM-DD date."""
    d = obj.get("effective_date")
    if d is None:
        return []
    try:
        datetime.date.fromisoformat(d)
    except ValueError:
        return [f"effective_date {d!r} is not a YYYY-MM-DD date"]
    return []


def load_keys(path: str) -> None:
    for line in Path(path).expanduser().read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY") and value and not os.environ.get(name):
            os.environ[name] = value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keys", help="file with NAME=value lines")
    ap.add_argument("--models", help='JSON: {"openai": ["model", input, output, cached_input], ...}')
    ap.add_argument("--max-usd", type=float, default=0.05)
    args = ap.parse_args()
    if args.keys:
        load_keys(args.keys)
    models = dict(MODELS)
    for provider, (name, *price) in json.loads(args.models or "{}").items():
        models[provider] = (name, Price(*price))
    budget = Budget(args.max_usd)
    keys = [os.environ.get(k, "") for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY")]
    log = CallLog("smoke-calls.jsonl", secrets=keys)
    rows, failed = [], 0
    for provider, mechanism, extra, max_tokens, what in CASES:
        key_name = f"{provider.upper()}_API_KEY"
        if not os.environ.get(key_name):
            rows.append((provider, mechanism, what, f"skipped: no {key_name}"))
            continue
        model, price = models[provider]
        client = Client(
            provider,
            model,
            mechanism=mechanism,
            price=price,
            budget=budget,
            log=log,
            extra_body=extra,
            max_tokens=max_tokens,
            bigger_factor=8 if max_tokens < 300 else 4,
        )
        try:
            res = client.structured(SYSTEM, AGREEMENT, SCHEMA, check=real_date)
        except ModelCallError as e:
            rows.append((provider, mechanism, what, f"ERROR {type(e).__name__}: {e}"[:160]))
            failed += 1
            continue
        failed += not res.ok
        outcome = (
            f"{'ok' if res.ok else 'NOT OK'}; first answer {res.category}"
            + (f", recovered by {res.recovered_by}" if res.recovered_by else "")
            + f"; {res.requests} request(s), {len(res.attempts)} attempt(s), ${res.cost_usd:.5f}, {res.seconds:.1f} s"
            + (f"; problems: {res.problems}" if res.problems else "")
        )
        rows.append((provider, mechanism, what, outcome))
        if res.ok:
            rows.append(("", "", "", "value: " + json.dumps(res.value, ensure_ascii=False)))
    for r in rows:
        print(" | ".join(x for x in r if x) if r[0] else "      " + r[3])
    print(f"\nspent ${budget.spent:.5f} of ${budget.max_usd:.2f}; log: smoke-calls.jsonl")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
