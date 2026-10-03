"""Run a tool's side effect once, however many times the model or the retries ask for it.

A model may call the same tool twice in one answer, the same request may be sent again after a timeout, and
a feedback round asks for the call again. If the tool charges a card, sends an email or writes a row, each
of those is a second charge, email or row. Payments solved this long ago with idempotency keys, and the
same two rules apply here:

  * The key names the business operation, and it comes from your side: "refund for order 17", not the
    model's tool-call id (new on every answer) and not the model's arguments (a corrected or reformatted
    argument would make a new key and a second refund). Put arguments in the key only when different
    arguments really mean a different action.
  * "Started and never finished" is not "failed". If a run was interrupted after the side effect may have
    happened, the next attempt raises InDoubt instead of doing it again. Check the outside world, then call
    resolve() with what happened or forget() to allow a new run.

Backed by SQLite, so it holds across threads, processes and restarts when given a file path.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time

from .transport import ModelCallError


class InDoubt(ModelCallError):
    """This action started earlier and did not record an outcome."""


class NotApplied(Exception):
    """Raise it from an action to say nothing happened outside; the key is freed for another try."""


class ToolLedger:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS actions (key TEXT PRIMARY KEY, state TEXT NOT NULL, result TEXT,"
                " error TEXT, started_at REAL NOT NULL, finished_at REAL)"
            )

    @staticmethod
    def key(operation: str, tool: str, args=None) -> str:
        """`operation`: your id for the business step (an order id and the step name, say), not the model's.
        `args`: only when different arguments make a different action."""
        blob = json.dumps([operation, tool, args], sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def state(self, key: str):
        """-> None, 'started' or 'done'."""
        with self._lock:
            row = self._db.execute("SELECT state FROM actions WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def once(self, key: str, action):
        """Run `action()` if this key has never run; return the stored result if it finished before."""
        with self._lock:
            cur = self._db.execute("INSERT OR IGNORE INTO actions (key, state, started_at) VALUES (?, 'started', ?)", (key, time.time()))
            claimed = cur.rowcount == 1
            row = (
                None
                if claimed
                else self._db.execute("SELECT state, result, error, started_at FROM actions WHERE key = ?", (key,)).fetchone()
            )
        if row is not None:
            state, result, error, started = row
            if state == "done":
                return json.loads(result)
            when = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(started))
            raise InDoubt(
                f"action {key[:12]}… started {when} and has no recorded outcome"
                + (f" (it raised {error})" if error else " (still running, or interrupted)")
                + "; check whether it happened, then resolve() or forget()"
            )
        try:
            result = action()
        except NotApplied:
            self.forget(key)
            raise
        except BaseException as e:
            with self._lock:
                self._db.execute("UPDATE actions SET error = ? WHERE key = ?", (repr(e)[:500], key))
            raise
        self.resolve(key, result)
        return result

    def resolve(self, key: str, result) -> None:
        """Record the outcome (after a successful run, or after you checked an in-doubt one by hand).
        The result is stored as JSON; a value JSON cannot hold is stored as its string."""
        try:
            blob = json.dumps(result, default=str)
        except (TypeError, ValueError):
            blob = json.dumps(repr(result))
        now = time.time()
        with self._lock:
            cur = self._db.execute(
                "UPDATE actions SET state = 'done', result = ?, error = NULL, finished_at = ? WHERE key = ?", (blob, now, key)
            )
            if cur.rowcount == 0:
                self._db.execute(
                    "INSERT INTO actions (key, state, result, started_at, finished_at) VALUES (?, 'done', ?, ?, ?)", (key, blob, now, now)
                )

    def forget(self, key: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM actions WHERE key = ?", (key,))
