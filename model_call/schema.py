"""A small JSON Schema validator: the subset that structured-output APIs accept.

Supported: type (one name or a list), properties, required, additionalProperties (bool), items, enum, const,
anyOf, minimum, maximum, exclusiveMinimum, exclusiveMaximum, minLength, maxLength, minItems, maxItems, pattern.
Annotations are allowed and not checked: title, description, default, examples, format, $schema.

Any other keyword (allOf, oneOf, not, $ref, $defs, ...) raises SchemaError when the schema is checked, so a
rule is never skipped without anyone noticing.
"""

from __future__ import annotations

import re

CHECKED = {
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "enum",
    "const",
    "anyOf",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "pattern",
}
ANNOTATIONS = {"title", "description", "default", "examples", "format", "$schema"}
TYPES = {"object", "array", "string", "number", "integer", "boolean", "null"}


class SchemaError(ValueError):
    """The schema uses something this validator does not check."""


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def check_schema(schema, path: str = "$") -> None:
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: a schema must be an object")
    unknown = set(schema) - CHECKED - ANNOTATIONS
    if unknown:
        raise SchemaError(f"{path}: not supported: {', '.join(sorted(unknown))}")
    t = schema.get("type")
    names = t if isinstance(t, list) else [t] if t is not None else []
    for n in names:
        if n not in TYPES:
            raise SchemaError(f"{path}: unknown type {n!r}")
    if "additionalProperties" in schema and not isinstance(schema["additionalProperties"], bool):
        raise SchemaError(f"{path}: additionalProperties must be true or false here")
    for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if k in schema and not _is_number(schema[k]):
            raise SchemaError(f"{path}: {k} must be a number (the boolean form of older drafts is not supported)")
    for k in ("minLength", "maxLength", "minItems", "maxItems"):
        if k in schema and not (isinstance(schema[k], int) and not isinstance(schema[k], bool) and schema[k] >= 0):
            raise SchemaError(f"{path}: {k} must be a non-negative integer")
    if "required" in schema and not (isinstance(schema["required"], list) and all(isinstance(r, str) for r in schema["required"])):
        raise SchemaError(f"{path}: required must be a list of names")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise SchemaError(f"{path}: enum must be a list")
    if "properties" in schema and not isinstance(schema["properties"], dict):
        raise SchemaError(f"{path}: properties must be an object")
    if "anyOf" in schema and not (isinstance(schema["anyOf"], list) and schema["anyOf"]):
        raise SchemaError(f"{path}: anyOf must be a non-empty list")
    for name, sub in (schema.get("properties") or {}).items():
        check_schema(sub, f"{path}.{name}")
    if "items" in schema:
        check_schema(schema["items"], f"{path}[]")
    for i, sub in enumerate(schema.get("anyOf") or []):
        check_schema(sub, f"{path}.anyOf[{i}]")
    if "pattern" in schema:
        if not isinstance(schema["pattern"], str):
            raise SchemaError(f"{path}: pattern must be a string")
        _regex(schema["pattern"])


_REGEX_CACHE = {}


_W = "A-Za-z0-9_"
_OUTSIDE = {
    "d": "[0-9]",
    "D": "[^0-9]",
    "w": f"[{_W}]",
    "W": f"[^{_W}]",
    "b": f"(?:(?<=[{_W}])(?![{_W}])|(?<![{_W}])(?=[{_W}]))",
    "B": f"(?:(?<=[{_W}])(?=[{_W}])|(?<![{_W}])(?![{_W}]))",
}
_INSIDE = {"d": "0-9", "w": _W}


def _regex(pattern: str):
    """A JSON Schema (ECMA-262) pattern as a Python regex: \\d, \\w and \\b are ASCII (\\s stays Unicode, as in
    ECMA), and $ matches only at the very end (Python's $ also matches before a final newline)."""
    rx = _REGEX_CACHE.get(pattern)
    if rx is None:
        out, i, in_class = [], 0, False
        while i < len(pattern):
            c = pattern[i]
            if c == "\\" and i + 1 < len(pattern):
                e = pattern[i + 1]
                if in_class and e in _INSIDE:
                    out.append(_INSIDE[e])
                elif not in_class and e in _OUTSIDE:
                    out.append(_OUTSIDE[e])
                else:
                    out.append(pattern[i : i + 2])
                i += 2
                continue
            if c == "[" and not in_class:
                in_class = True
            elif c == "]" and in_class:
                in_class = False
            elif c == "$" and not in_class:
                c = r"\Z"
            out.append(c)
            i += 1
        try:
            rx = re.compile("".join(out))
        except re.error as e:
            raise SchemaError(f"pattern {pattern!r} does not compile: {e}") from None
        _REGEX_CACHE[pattern] = rx
    return rx


def same(a, b) -> bool:
    """JSON equality: true is not 1, 1 is 1.0, lists and objects compared item by item."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) and _is_number(b):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def type_of(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def has_type(value, name: str) -> bool:
    actual = type_of(value)
    if name == "number":
        return actual in ("integer", "number")
    if name == "integer":
        return actual == "integer" or (actual == "number" and float(value).is_integer())
    return actual == name


def allows_null(schema: dict) -> bool:
    t = schema.get("type")
    if t == "null" or (isinstance(t, list) and "null" in t):
        return True
    if "enum" in schema and any(e is None for e in schema["enum"]):
        return True
    return any(allows_null(s) for s in schema.get("anyOf") or [])


def _where(path: str) -> str:
    return path[2:] if path.startswith("$.") else path


def validate(schema: dict, value, path: str = "$") -> list:
    """-> problems, one line each, empty if the value matches."""
    problems = []
    t = schema.get("type")
    if t is not None:
        names = t if isinstance(t, list) else [t]
        if not any(has_type(value, n) for n in names):
            want = " or ".join(names)
            return [f"{_where(path)}: expected {want}, got {type_of(value)}"]
    if "const" in schema and not same(value, schema["const"]):
        problems.append(f"{_where(path)}: must be {schema['const']!r}")
    if "enum" in schema and not any(same(value, e) for e in schema["enum"]):
        problems.append(f"{_where(path)}: {value!r} is not one of {schema['enum']!r}")
    if "anyOf" in schema:
        if not any(not validate(s, value, path) for s in schema["anyOf"]):
            details = "; ".join(p for s in schema["anyOf"] for p in validate(s, value, path))
            problems.append(f"{_where(path)}: matches none of the allowed forms ({details})")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            problems.append(f"{_where(path)}: {value} is less than {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            problems.append(f"{_where(path)}: {value} is more than {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            problems.append(f"{_where(path)}: {value} must be more than {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            problems.append(f"{_where(path)}: {value} must be less than {schema['exclusiveMaximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            problems.append(f"{_where(path)}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            problems.append(f"{_where(path)}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and not _regex(schema["pattern"]).search(value):
            problems.append(f"{_where(path)}: does not match {schema['pattern']!r}")
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            problems.append(f"{_where(path)}: fewer than {schema['minItems']} items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            problems.append(f"{_where(path)}: more than {schema['maxItems']} items")
        if "items" in schema:
            for i, item in enumerate(value):
                problems += validate(schema["items"], item, f"{path}[{i}]")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for name in schema.get("required") or []:
            if name not in value:
                problems.append(f"{_where(path + '.' + name)}: missing")
        for name, item in value.items():
            if name in props:
                problems += validate(props[name], item, f"{path}.{name}")
            elif schema.get("additionalProperties") is False:
                problems.append(f"{_where(path + '.' + name)}: not allowed (unknown key)")
    return problems
