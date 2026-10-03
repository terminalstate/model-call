"""One JSON line per call: what was asked, what came back, what was done about it, how long, how much.

Without it, the one answer that went wrong in production cannot be looked at. API keys never enter the log
(they travel in headers, which are not written), and any key given in `secrets` is masked if it shows up in
a body. Account and request ids in error texts are masked too. With content=False the prompts and answers
are left out and only the shape of each call is kept.
"""

from __future__ import annotations

import json
import threading

from .transport import scrub


class CallLog:
    def __init__(self, path: str, content: bool = True, secrets=()):
        self.path = path
        self.content = content
        self._secrets = [s for s in secrets if s]
        self._lock = threading.Lock()

    CONTENT = ("body", "response", "value", "problems", "notes", "error")  # problem texts can quote the answer

    def write(self, record: dict) -> None:
        if not self.content:
            record = {k: v for k, v in record.items() if k not in self.CONTENT}
            record["requests"] = [
                {
                    **{k: v for k, v in r.items() if k not in self.CONTENT},
                    "attempts": [{k: v for k, v in a.items() if k != "detail"} for a in r.get("attempts", [])],
                }
                for r in record.get("requests", [])
            ]
        line = json.dumps(record, ensure_ascii=False, default=str)
        for s in self._secrets:
            line = line.replace(s, "[secret]")
        line = scrub(line)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
