"""Local repair of a model's answer, without another request.

It reshapes what the model wrote and never invents a value. A field it cannot read is listed in `dropped`
(and set to null if the schema allows null, or left out if not), so a caller can always tell a complete
answer from a partial one.

What it does, in order:
  text -> JSON: as is; without trailing commas; taken out of a code fence or surrounding prose (only if
          there is exactly one candidate: an example followed by the answer is ambiguous, not repaired);
          a cut-off object closed at the last complete value;
  shape: a one-item list around the object is removed; a key is read as a schema key if it differs only in
         case or spacing, in singular/plural, or by a typo (one letter for keys up to six letters, two above),
         and only if the match is unique both ways; a value is read into the declared type when nothing is
         lost: "2" -> 2, "x" -> ["x"] for a list, "year" -> "years" or "USD" -> "usd" for an enum when exactly
         one enum value matches, and, last of all, "n/a" -> null where null is allowed;
  missing keys: read as null where null is allowed, unless the answer was cut off or had keys that could not
         be read; otherwise dropped.
  cut-off answers: a field the text may have stopped in (the last one, unless a comma or a closed string,
         list or object shows it ended) is not trusted, and a field that never came is unknown, not null.
         The same goes for an object the API parsed from a cut-off answer.
  a key given twice with different values is not trusted either.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .schema import allows_null, same, validate

EMPTY_WORDS = {"", "null", "none", "n/a", "na", "not stated", "not specified", "unknown", "not applicable"}
_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n?(.*?)```", re.S)
_INT = re.compile(r"^[+-]?\d+$")
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_DROP = object()


@dataclass
class Repaired:
    value: dict | None
    complete: bool
    notes: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    cut_off: bool = False


class _Obj(dict):
    """A parsed JSON object that keeps its key/value pairs in order, duplicates included."""

    pairs: list = []


def _hook(pairs):
    o = _Obj(pairs)
    o.pairs = pairs
    return o


def _loads(text: str):
    return json.loads(text, object_pairs_hook=_hook)


def _plain(v):
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_plain(x) for x in v]
    return v


def strip_trailing_commas(text: str) -> str:
    """Removes a comma that comes right before } or ], outside strings."""
    out, in_str, esc, i = [], False, False, 0
    while i < len(text):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == ",":
            j = i + 1
            while j < len(text) and text[j] in " \t\r\n":
                j += 1
            if j < len(text) and text[j] in "}]":
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


def matching_brace(s: str, start: int):
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            depth += 1
        elif c in "}]":
            depth -= 1
            if depth == 0:
                return i
    return None


def _candidates(text: str) -> list:
    found = []
    for m in _FENCE.finditer(text):
        try:
            v = _loads(m.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(v, (dict, list)):
            found.append(v)
    if found:
        return found
    start = text.find("{")
    while start != -1:
        end = matching_brace(text, start)
        if end is None:
            break
        try:
            found.append(_loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass  # a broken outer object: an object nested in it is not taken for the answer
        start = text.find("{", end + 1)
    return found


def extract_json(text: str):
    """The one JSON object inside a code fence or prose, or None if there is none or more than one."""
    distinct = []
    for v in _candidates(text):
        if not any(same(v, d) for d in distinct):
            distinct.append(v)
    return distinct[0] if len(distinct) == 1 else None


def close_cut_off(text: str):
    """The longest prefix of a cut-off JSON object that parses once its open brackets are closed.
    -> (object or None, whether its last top-level value is known to be whole)

    The last value is whole when the cut came after it at the top level: a closed string, list or object, a
    literal, or anything followed by a comma. A number at the very end may have lost digits, and a list or
    object still open at the cut may have lost items."""
    start = text.find("{")
    if start < 0:
        return None, False
    s = text[start:]
    stack, in_str, esc = [], False, False
    state = [(False, "")]  # state[i]: (inside a string, closing brackets) for the prefix s[:i]
    for c in s:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in "{[":
            stack.append("}" if c == "{" else "]")
        elif c in "}]" and stack:
            stack.pop()
        state.append((in_str, "".join(reversed(stack))))
    for cut in range(len(s), 0, -1):
        inside, closers = state[cut]
        if inside:
            continue
        head = s[:cut].rstrip()
        t = head.rstrip(",").rstrip()
        if not t or t.endswith(":"):
            continue
        try:
            v = _loads(t + closers)
        except json.JSONDecodeError:
            continue
        if isinstance(v, dict):
            comma_after = head.endswith(",") or s[cut:].lstrip().startswith(",")
            whole = closers == "}" and (comma_after or not t[-1].isdigit())
            return v, whole
    return None, False


def close_truncated(text: str):
    """The longest prefix of a cut-off JSON object that parses once its open brackets are closed, or None."""
    return _plain(close_cut_off(text)[0])


def parse_loose(text: str):
    """-> (object or None, notes, closed: the text was cut off and closed here, last value whole)"""
    text = (text or "").strip()
    if not text:
        return None, [], False, True
    fixed = strip_trailing_commas(text)
    steps = (
        (lambda: _loads(text), None),
        (lambda: _loads(fixed), "trailing comma removed"),
        (lambda: extract_json(text), "JSON taken out of a code fence or prose"),
        (lambda: extract_json(fixed), "JSON taken out of a code fence or prose; trailing comma removed"),
    )
    for step, note in steps:
        try:
            obj = step()
        except json.JSONDecodeError:
            continue
        if obj is not None:
            return obj, [note] if note else [], False, True
    start = fixed.find("{")
    if start < 0 or matching_brace(fixed, start) is not None:
        return None, [], False, True  # the object ends, so it was not cut off: it is broken or ambiguous
    obj, whole = close_cut_off(fixed)
    if obj is not None:
        return obj, ["cut-off JSON closed at the last complete value"], True, whole
    return None, [], False, True


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm_key(k) -> str:
    return re.sub(r"[\s\-]+", "_", str(k).strip()).lower()


def _forms(n: str) -> set:
    out = {n, n + "s", n + "es"}
    if n.endswith("ies"):
        out.add(n[:-3] + "y")
    if n.endswith("es"):
        out.add(n[:-2])
    if n.endswith("s"):
        out.add(n[:-1])
    if n.endswith("y"):
        out.add(n[:-1] + "ies")
    return out


def _same_norm(k, p) -> bool:
    return norm_key(k) == norm_key(p)


def _plural(k, p) -> bool:
    return norm_key(p) in _forms(norm_key(k))


def _typo(k, p) -> bool:
    nk, np_ = norm_key(k), norm_key(p)
    if min(len(nk), len(np_)) < 5:  # short words one letter apart are often different words: site/size, rate/date
        return False
    return edit_distance(nk, np_) <= (1 if len(np_) <= 6 else 2)


def _match(keys: list, free: list, pred) -> dict:
    """Pairs each key with a schema key when each is the other's only match."""
    out = {}
    for k in keys:
        cands = [p for p in free if pred(k, p)]
        if len(cands) == 1 and sum(1 for k2 in keys if pred(k2, cands[0])) == 1:
            out[k] = cands[0]
    return out


def _where(path: str) -> str:
    return path[2:] if path.startswith("$.") else path


def _types(schema: dict) -> list:
    t = schema.get("type")
    return t if isinstance(t, list) else [t] if t else []


def _enum_match(schema: dict, v):
    if not isinstance(v, str) or "enum" not in schema:
        return _DROP
    s = v.strip().lower()
    hits = {e for e in schema["enum"] if isinstance(e, str) and (e.lower() == s or e.lower() == s + "s" or e.lower() + "s" == s)}
    return next(iter(hits)) if len(hits) == 1 else _DROP


def _fix(schema: dict, v, path: str, ctx: dict):
    """-> a value that matches `schema` (an object may be partial, see ctx['dropped']), or _DROP."""
    if not validate(schema, v, path):
        return v
    notes = ctx["notes"]
    if "anyOf" in schema:
        for null_words in (False, True):  # a real value in some alternative before "n/a" as null
            for alt in schema["anyOf"]:
                trial = {**ctx, "notes": [], "dropped": [], "null_words": null_words}
                got = _fix(alt, v, path, trial)
                if got is not _DROP and not trial["dropped"] and not validate(schema, got, path):
                    notes.extend(trial["notes"])
                    return got
        return _DROP
    types = _types(schema)
    if isinstance(v, dict) and "object" in types:
        return _fix_object(schema, v, path, ctx)
    if isinstance(v, list) and "array" in types:
        return _fix_array(schema, v, path, ctx)
    candidates = []
    if "enum" in schema:

        def enum():
            e = _enum_match(schema, v)
            if e is not _DROP:
                notes.append(f"{_where(path)}: {v!r} read as {e!r}")
            return e

        candidates.append(enum)
    if isinstance(v, str) and _NUMBER.match(v.strip()) and ("number" in types or "integer" in types):

        def number():
            s = v.strip()
            if _INT.match(s):
                x = int(s)  # exact, however long
            else:
                x = float(s)
                if "number" not in types:
                    if not x.is_integer():
                        return _DROP
                    x = int(x)
            notes.append(f"{_where(path)}: {v!r} read as {x!r}")
            return x

        candidates.append(number)
    if ctx.get("null_words", True) and allows_null(schema) and isinstance(v, str) and v.strip().lower() in EMPTY_WORDS:

        def null():
            notes.append(f"{_where(path)}: {v!r} read as null")
            return None

        candidates.append(null)
    empty_word = isinstance(v, str) and v.strip().lower() in EMPTY_WORDS
    if not isinstance(v, list) and "array" in types and "items" in schema and not empty_word:

        def single():
            trial = {**ctx, "notes": [], "dropped": []}
            item = _fix(schema["items"], v, path + "[0]", trial)
            if item is _DROP or trial["dropped"]:
                return _DROP
            notes.extend(trial["notes"])
            notes.append(f"{_where(path)}: a single value read as a list of one")
            return [item]

        candidates.append(single)
    for make in candidates:
        got = make()
        if got is not _DROP and not validate(schema, got, path):
            return got
    return _DROP


def _fix_array(schema: dict, items: list, path: str, ctx: dict):
    out = []
    for i, item in enumerate(items):
        got = _fix(schema.get("items") or {}, item, f"{path}[{i}]", ctx)
        if got is _DROP:
            return _DROP  # one unreadable item makes the list unreadable: dropping it would change the answer
        out.append(got)
    return out


def _fix_object(schema: dict, obj: dict, path: str, ctx: dict):
    notes, dropped = ctx["notes"], ctx["dropped"]
    props = schema.get("properties") or {}
    closed = schema.get("additionalProperties") is False
    unknown_keys = bool(ctx.pop("unread", False)) if path == "$" else False
    mapping = {k: k for k in obj if k in props}
    unmatched = [k for k in obj if k not in props]
    free = [p for p in props if p not in obj]

    def typo_that_fits(k, p) -> bool:
        return _typo(k, p) and not validate(props[p], obj[k])  # a typo match must also hold a value that fits

    for pred in (_same_norm, _plural, typo_that_fits):
        found = _match(unmatched, free, pred)
        for k, p in found.items():
            mapping[k] = p
            notes.append(f"{_where(path + '.' + str(k))}: key read as {p!r}")
        unmatched = [k for k in unmatched if k not in found]
        free = [p for p in free if p not in found.values()]
    out = {}
    for k in unmatched:
        unknown_keys = True  # it may hold a field we are missing
        if closed:
            notes.append(f"{_where(path + '.' + str(k))}: unknown key left out")
        else:
            out[k] = obj[k]
    for k, p in mapping.items():
        before = len(dropped)
        got = _fix(props[p], obj[k], f"{path}.{p}", ctx)
        if got is not _DROP and len(dropped) == before and validate(props[p], got):
            got = _DROP  # still wrong, and nothing inside it was marked: the field as a whole is unreadable
        if got is _DROP:
            dropped.append(_where(f"{path}.{p}"))
            if allows_null(props[p]):
                out[p] = None
        else:
            out[p] = got
    for p in schema.get("required") or []:
        if p in mapping.values() or p in out:
            continue
        if not ctx["cut_off"] and not unknown_keys and allows_null(props.get(p, {})):
            out[p] = None
            notes.append(f"{_where(path + '.' + p)}: missing, read as null")
        else:
            dropped.append(_where(f"{path}.{p}"))
    order = {p: i for i, p in enumerate(props)}
    return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], len(order))))


def repair(answer, schema: dict, cut_off: bool = False) -> Repaired:
    """`answer`: the text the model wrote, or the object an API already parsed (a tool call's input).
    `cut_off`: the API said the answer stopped at the output limit (or was stopped)."""
    notes, closed, whole = [], False, True
    from_api = not isinstance(answer, str)
    if isinstance(answer, str):
        answer, notes, closed, whole = parse_loose(answer)
        cut_off = cut_off or closed
    if isinstance(answer, list) and len(answer) == 1 and isinstance(answer[0], dict):
        answer = answer[0]
        notes.append("object taken out of a one-item list")
    if not isinstance(answer, dict) or "object" not in _types(schema):
        return Repaired(None, False, notes, [], cut_off)
    props = schema.get("properties") or {}
    pairs = answer.pairs if isinstance(answer, _Obj) else list(answer.items())
    unread = []
    seen = {}
    for k, v in pairs:
        if k in seen and not same(seen[k], v) and k not in unread:
            unread.append(k)
            notes.append(f"{k}: given twice with different values; left unread")
        seen.setdefault(k, v)
    uncertain = closed or (cut_off and from_api)
    if uncertain and pairs and ((closed and not whole) or from_api):
        last = pairs[-1][0]
        if last not in unread:
            unread.append(last)
            notes.append(f"{last}: the answer was cut off in or after this field; left unread")
    answer = {k: v for k, v in answer.items() if k not in unread}
    ctx = {"notes": notes, "dropped": [], "cut_off": cut_off, "unread": bool(unread)}
    value = _plain(_fix_object(schema, answer, "$", ctx))
    dropped = set(ctx["dropped"])
    dropped.update(k for k in unread if k in props)
    if uncertain:
        dropped.update(p for p in props if p not in value)  # never came: unknown, not absent
    for p in dropped:
        if p in props and allows_null(props[p]):
            value[p] = None
        elif p in props:
            value.pop(p, None)
    order = {p: i for i, p in enumerate(props)}
    value = dict(sorted(value.items(), key=lambda kv: order.get(kv[0], len(order))))
    dropped = sorted(dropped)
    required = schema.get("required") or []
    if required and all(p in dropped for p in required):
        return Repaired(None, False, notes, dropped, cut_off)  # nothing that was asked for could be read
    complete = not dropped and not any(k not in props for k in unread) and not validate(schema, value)
    if not complete and not dropped:
        return Repaired(None, False, notes, [], cut_off)  # wrong in a way no field can be blamed for
    return Repaired(value, complete, notes, dropped, cut_off)
