import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model_call.repair import close_truncated, extract_json, repair  # noqa: E402
from model_call.schema import SchemaError, check_schema, validate  # noqa: E402

TERM = {
    "type": "object",
    "properties": {
        "number": {"type": "number", "exclusiveMinimum": 0},
        "unit": {"type": "string", "enum": ["days", "weeks", "months", "years"]},
    },
    "required": ["number", "unit"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "parties": {"type": "array", "items": {"type": "string"}},
        "effective_date": {"type": ["string", "null"], "pattern": r"^\d{4}-\d{2}-\d{2}$"},
        "jurisdiction": {"type": ["string", "null"]},
        "term": {"anyOf": [{"type": "null"}, TERM]},
    },
    "required": ["parties", "effective_date", "jurisdiction", "term"],
    "additionalProperties": False,
}
GOOD = {
    "parties": ["Acme Corp", "Jane Roe"],
    "effective_date": "2001-04-18",
    "jurisdiction": "Oregon",
    "term": {"number": 2, "unit": "years"},
}


class SchemaTest(unittest.TestCase):
    def test_good_passes(self):
        self.assertEqual(validate(SCHEMA, GOOD), [])

    def test_types(self):
        self.assertEqual(validate({"type": "integer"}, 2.0), [])
        self.assertTrue(validate({"type": "integer"}, 2.5))
        self.assertTrue(validate({"type": "integer"}, True))
        self.assertTrue(validate({"type": "number"}, "2"))
        self.assertEqual(validate({"type": ["string", "null"]}, None), [])

    def test_problems_name_the_field(self):
        bad = {**GOOD, "term": {"number": 1, "unit": "year"}, "extra": 1}
        bad.pop("jurisdiction")
        text = " | ".join(validate(SCHEMA, bad))
        self.assertIn("jurisdiction: missing", text)
        self.assertIn("extra: not allowed", text)
        self.assertIn("'year' is not one of", text)

    def test_numbers_strings_lists(self):
        self.assertTrue(validate({"type": "number", "minimum": 1}, 0))
        self.assertTrue(validate({"type": "number", "exclusiveMinimum": 0}, 0))
        self.assertTrue(validate({"type": "string", "pattern": "^a"}, "b"))
        self.assertTrue(validate({"type": "array", "minItems": 1}, []))
        self.assertTrue(validate({"const": 3}, 4))

    def test_booleans_are_not_numbers(self):
        self.assertTrue(validate({"enum": [0, 1]}, True))
        self.assertTrue(validate({"const": 1}, True))
        self.assertTrue(validate({"const": False}, 0))
        self.assertTrue(validate({"enum": [[1]]}, [True]))
        self.assertEqual(validate({"enum": [1]}, 1.0), [])

    def test_pattern_is_ecma_like(self):
        p = {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}
        self.assertEqual(validate(p, "2001-04-18"), [])
        self.assertTrue(validate(p, "2001-04-18\n"))
        self.assertTrue(validate({"type": "string", "pattern": r"^\d{4}$"}, "\u0662\u0660\u0660\u0661"))
        self.assertEqual(validate({"type": "string", "pattern": r"^[$a]+$"}, "$a"), [])

    def test_whitespace_classes_stay_unicode(self):
        self.assertTrue(validate({"type": "string", "pattern": r"^\S+$"}, "a\u00a0b"))
        self.assertTrue(validate({"type": "string", "pattern": r"^[^\s]+$"}, "a\u2003b"))
        self.assertEqual(validate({"type": "string", "pattern": r"^\w+\b"}, "abc_1"), [])

    def test_unsupported_keywords_raise(self):
        for bad in (
            {"$ref": "#/x"},
            {"oneOf": []},
            {"type": "object", "properties": {"a": {"allOf": []}}},
            {"type": "map"},
            {"type": "number", "exclusiveMinimum": True},
            {"type": "string", "pattern": "("},
        ):
            with self.assertRaises(SchemaError):
                check_schema(bad)
        check_schema({**SCHEMA, "description": "fine", "$schema": "x"})


class RepairTest(unittest.TestCase):
    def test_code_fence(self):
        text = 'Here it is:\n```json\n{"parties": ["Acme Corp"], "effective_date": null, "jurisdiction": null, "term": null}\n```\nHope this helps.'
        r = repair(text, SCHEMA)
        self.assertTrue(r.complete)
        self.assertEqual(r.value["parties"], ["Acme Corp"])
        self.assertIn("code fence", r.notes[0])

    def test_trailing_comma(self):
        r = repair('{"parties": [], "effective_date": null, "jurisdiction": null, "term": null,}', SCHEMA)
        self.assertTrue(r.complete)

    def test_year_read_as_years(self):
        r = repair({**GOOD, "term": {"number": 1, "unit": "year"}}, SCHEMA)
        self.assertTrue(r.complete)
        self.assertEqual(r.value["term"], {"number": 1, "unit": "years"})
        self.assertTrue(any("'year' read as 'years'" in n for n in r.notes))

    def test_misspelt_key_and_case(self):
        obj = {"partties": ["A"], "Effective Date": None, "jurisdiction": None, "term": None}
        r = repair(obj, SCHEMA)
        self.assertTrue(r.complete, r)
        self.assertEqual(r.value["parties"], ["A"])
        self.assertIn("effective_date", r.value)

    def test_number_in_quotes_and_single_value(self):
        obj = {**GOOD, "parties": "Acme Corp", "term": {"number": "3", "unit": "Months"}}
        r = repair(obj, SCHEMA)
        self.assertTrue(r.complete, r)
        self.assertEqual(r.value["parties"], ["Acme Corp"])
        self.assertEqual(r.value["term"], {"number": 3, "unit": "months"})

    def test_empty_words_and_missing_null(self):
        r = repair({"parties": [], "effective_date": "n/a", "term": None}, SCHEMA)
        self.assertTrue(r.complete, r)
        self.assertIsNone(r.value["effective_date"])
        self.assertIsNone(r.value["jurisdiction"])
        self.assertTrue(any("missing, read as null" in n for n in r.notes))

    def test_unreadable_value_is_dropped_not_guessed(self):
        r = repair({**GOOD, "effective_date": "April 18th", "term": {"number": 2, "unit": "fortnights"}}, SCHEMA)
        self.assertFalse(r.complete)
        self.assertEqual(r.dropped, ["effective_date", "term"])
        self.assertIsNone(r.value["effective_date"])
        self.assertIsNone(r.value["term"])
        self.assertEqual(r.value["parties"], GOOD["parties"])

    def test_missing_list_is_dropped(self):
        r = repair({"effective_date": None, "jurisdiction": None, "term": None}, SCHEMA)
        self.assertFalse(r.complete)
        self.assertEqual(r.dropped, ["parties"])

    def test_unknown_key_makes_missing_keys_unknown(self):
        r = repair({"parties": [], "governing": "Oregon", "effective_date": None, "term": None}, SCHEMA)
        self.assertEqual(r.dropped, ["jurisdiction"])  # it may have been in the key we could not read

    def test_cut_off_last_field_is_unread(self):
        text = '{"parties": ["Acme Corp", "Jane Roe"], "effective_date": "2001-04-18", "jurisdiction": "Oreg'
        r = repair(text, SCHEMA, cut_off=True)
        self.assertTrue(r.cut_off)
        self.assertFalse(r.complete)
        self.assertEqual(r.value["parties"], ["Acme Corp", "Jane Roe"])
        self.assertEqual(r.value["effective_date"], "2001-04-18")  # a comma followed it: whole
        self.assertEqual(r.dropped, ["jurisdiction", "term"])  # cut off: missing means unknown, not null

    def test_number_at_the_cut_is_unread(self):
        s = {"type": "object", "properties": {"a": {"type": "string"}, "n": {"type": "integer"}}, "required": ["a", "n"]}
        r = repair('{"a": "x", "n": 12', s)
        self.assertEqual(r.dropped, ["n"])  # 12 may have been 123
        r = repair('{"a": "x", "n": 12,', s)
        self.assertTrue(r.complete)

    def test_list_cut_mid_way_is_not_delivered(self):
        r = repair('{"jurisdiction": null, "parties": ["Acme Corp", "Jane', SCHEMA)
        self.assertNotIn("parties", r.value or {})
        self.assertIn("parties", r.dropped)

    def test_nothing_readable(self):
        self.assertIsNone(repair("I cannot help with that.", SCHEMA).value)
        self.assertIsNone(repair("", SCHEMA).value)
        self.assertIsNone(repair({"foo": 1}, SCHEMA).value)

    def test_one_item_list(self):
        r = repair([GOOD], SCHEMA)
        self.assertTrue(r.complete)

    def test_ambiguous_enum_is_not_guessed(self):
        s = {"type": "object", "properties": {"u": {"type": "string", "enum": ["Day", "day", "week"]}}, "required": ["u"]}
        r = repair({"u": "DAY"}, s)
        self.assertEqual(r.dropped, ["u"])

    # cases found in review: each one was a wrong value delivered as complete

    def test_real_enum_value_beats_empty_word(self):
        s = {
            "type": "object",
            "properties": {"severity": {"type": ["string", "null"], "enum": ["none", "mild", None]}},
            "required": ["severity"],
        }
        self.assertEqual(repair({"severity": "None"}, s).value, {"severity": "none"})
        s2 = {
            "type": "object",
            "properties": {"x": {"anyOf": [{"type": "null"}, {"type": "string", "enum": ["None", "Some"]}]}},
            "required": ["x"],
        }
        self.assertEqual(repair({"x": "none"}, s2).value, {"x": "None"})
        self.assertEqual(repair({"x": "n/a"}, s2).value, {"x": None})

    def test_cut_off_optional_field_is_not_complete(self):
        s = {"type": "object", "properties": {"name": {"type": "string"}, "notes": {"type": "string"}}, "required": ["name"]}
        r = repair('{"name": "Acme", "notes": "Do not ship before the', s, cut_off=True)
        self.assertFalse(r.complete)
        self.assertEqual(r.dropped, ["notes"])
        self.assertEqual(r.value, {"name": "Acme"})

    def test_cut_off_object_from_the_api_is_not_trusted(self):
        s = {"type": "object", "properties": {"a": {"type": "string"}, "j": {"type": "string"}}, "required": ["a", "j"]}
        r = repair({"a": "x", "j": "Oreg"}, s, cut_off=True)
        self.assertFalse(r.complete)
        self.assertEqual(r.dropped, ["j"])
        self.assertTrue(repair({"a": "x", "j": "Oregon"}, s).complete)  # not cut off: trusted

    def test_open_schema_unmatched_key_makes_missing_unknown(self):
        s = {
            "type": "object",
            "properties": {"parties": {"type": "array"}, "jurisdiction": {"type": ["string", "null"]}},
            "required": ["parties", "jurisdiction"],
        }
        r = repair({"parties": ["A"], "governing_law": "Oregon"}, s)
        self.assertFalse(r.complete)
        self.assertEqual(r.dropped, ["jurisdiction"])

    def test_short_keys_are_not_guessed(self):
        s = {
            "type": "object",
            "properties": {"title": {"type": "string"}, "time": {"type": ["string", "null"]}},
            "required": ["title", "time"],
            "additionalProperties": False,
        }
        r = repair({"title": "Standup", "type": "meeting"}, s)
        self.assertNotEqual(r.value and r.value.get("time"), "meeting")
        s2 = {"type": "object", "properties": {"city": {"type": ["string", "null"]}}, "required": ["city"]}
        self.assertNotEqual((repair({"site": "example.com"}, s2).value or {}).get("city"), "example.com")

    def test_a_key_cannot_take_another_keys_field(self):
        s = {
            "type": "object",
            "properties": {"amount": {"type": "number"}, "account": {"type": ["number", "null"]}},
            "required": ["amount", "account"],
            "additionalProperties": False,
        }
        r = repair({"amont": 1, "Amount": 2}, s)
        self.assertEqual(r.value.get("amount"), 2)
        self.assertNotEqual(r.value.get("account"), 1)
        self.assertFalse(r.complete)

    def test_null_word_on_a_nullable_list_is_null_not_a_list(self):
        s = {"type": "object", "properties": {"x": {"type": ["array", "null"], "items": {"type": "string"}}}, "required": ["x"]}
        self.assertEqual(repair({"x": "N/A"}, s).value, {"x": None})
        self.assertEqual(repair({"x": "Acme"}, s).value, {"x": ["Acme"]})

    def test_short_words_one_letter_apart_are_different_words(self):
        s = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "size": {"type": ["string", "null"]}},
            "required": ["name", "size"],
        }
        r = repair({"name": "Shirt", "site": "shop.example.com"}, s)
        self.assertIsNone(r.value["size"])
        self.assertFalse(r.complete)

    def test_typo_match_needs_a_value_that_fits(self):
        s = {"type": "object", "properties": {"count": {"type": ["integer", "null"]}}, "required": ["count"]}
        self.assertIsNone(repair({"court": "Delaware"}, s).value)  # not read as count = "Delaware"
        self.assertEqual(repair({"coumt": 3}, s).value, {"count": 3})

    def test_plural_key(self):
        r = repair({"party": ["A"], "effective_date": None, "jurisdiction": None, "term": None}, SCHEMA)
        self.assertTrue(r.complete)
        self.assertEqual(r.value["parties"], ["A"])

    def test_big_quoted_integer_is_exact(self):
        s = {"type": "object", "properties": {"id": {"type": "integer"}}, "required": ["id"]}
        self.assertEqual(repair({"id": "12345678901234567890"}, s).value, {"id": 12345678901234567890})

    def test_trailing_comma_inside_a_string_is_kept(self):
        s = {"type": "object", "properties": {"code": {"type": "string"}, "n": {"type": "integer"}}, "required": ["code", "n"]}
        r = repair('{"code": "s = {1, 2, }", "n": 1,}', s)
        self.assertEqual(r.value, {"code": "s = {1, 2, }", "n": 1})

    def test_example_then_answer_is_ambiguous(self):
        text = 'For example {"parties": ["X"], "effective_date": null, "jurisdiction": null, "term": null}; the answer: {"parties": ["Acme"], "effective_date": null, "jurisdiction": null, "term": null}'
        self.assertIsNone(repair(text, SCHEMA).value)

    def test_duplicate_key_is_not_trusted(self):
        r = repair('{"parties": ["A"], "effective_date": null, "jurisdiction": "Oregon", "term": null, "jurisdiction": "Texas"}', SCHEMA)
        self.assertFalse(r.complete)
        self.assertEqual(r.dropped, ["jurisdiction"])
        self.assertTrue(
            repair(
                '{"parties": ["A"], "effective_date": null, "jurisdiction": "Oregon", "term": null, "jurisdiction": "Oregon"}', SCHEMA
            ).complete
        )

    def test_helpers(self):
        self.assertEqual(extract_json('x {"a": {"b": 1}} y'), {"a": {"b": 1}})
        self.assertIsNone(extract_json("no json"))
        self.assertEqual(close_truncated('{"a": 1, "b": [1, 2'), {"a": 1, "b": [1, 2]})


if __name__ == "__main__":
    unittest.main()
