"""Mirror of pi-schema's validator.test.ts: `compile()` and the validators, their
errors, the unresolved-reference option and the error messages.

Not mirrored: the `bytes` cases (a Uint8Array has no JSON value), a property
set to `undefined`, properties named like `Object.prototype` members and the
detached `check` (JavaScript-only), the `schemas` and `draft` options (not
ported: tool arguments compile under draft 2020-12 with no registered
documents) and deep-frozen schemas (a Python dict cannot be frozen; the
never-modified half is mirrored). TypeBox schemas are written as the JSON
Schema they emit.
"""

import copy
import math

import pytest

from pidrei_ai.utils.schema import FORMATS, ParseResult, ValidationError, compile


READ = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "offset": {"type": "integer", "minimum": 1}},
    "required": ["path"],
    "additionalProperties": False,
}

BLOCK = {
    "anyOf": [
        {
            "type": "object",
            "properties": {"kind": {"const": "text"}, "text": {"type": "string"}},
            "required": ["kind", "text"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {"kind": {"const": "image"}, "width": {"type": "integer", "minimum": 0}},
            "required": ["kind", "width"],
            "additionalProperties": False,
        },
    ]
}


def error(path: str, keyword: str, message: str) -> ValidationError:
    return ValidationError(path=path, keyword=keyword, message=message)


# errors


def test_reports_a_missing_field_and_an_unknown_field_at_the_field():
    assert compile(READ).errors({"offset": 0, "extra": True}) == [
        error("/path", "required", "missing"),
        error("/offset", "minimum", "expected >= 1"),
        error("/extra", "additionalProperties", "unknown field"),
    ]


def test_is_empty_for_a_valid_value():
    assert compile(READ).errors({"path": "a.ts"}) == []


def test_reports_the_branch_a_tagged_unions_tag_selects():
    validator = compile({"type": "array", "items": BLOCK})
    assert validator.errors([{"kind": "text", "text": "a"}, {"kind": "image", "width": -1}]) == [
        error("/1/width", "minimum", "expected >= 0")
    ]
    assert validator.errors([{"kind": "video"}]) == [error("/0/kind", "anyOf", 'expected one of ["text","image"]')]


def test_escapes_tilde_and_slash_in_paths_as_json_pointer_does():
    validator = compile({"type": "object", "properties": {"a/b~c": {"type": "string"}}})
    assert validator.errors({"a/b~c": 1}) == [error("/a~1b~0c", "type", "expected string, got integer")]


def test_follows_ref_into_the_referenced_schema():
    validator = compile(
        {
            "$defs": {"point": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"]}},
            "type": "array",
            "items": {"$ref": "#/$defs/point"},
        }
    )
    assert validator.errors([{"x": 1}, {}]) == [error("/1/x", "required", "missing")]


# PiDrei: pi's annotating compiler gives each node its own base and reads a draft 7 tuple `items` as a prefix;
# the engine's annotations do the same.
def test_unevaluated_properties_sees_properties_evaluated_through_a_reference_inside_a_nested_id():
    validator = compile(
        {
            "allOf": [
                {"$id": "https://example.com/sub", "$defs": {"x": {"properties": {"a": True}}}, "$ref": "#/$defs/x"}
            ],
            "unevaluatedProperties": False,
        }
    )
    assert validator.check({"a": 1}) is True
    assert validator.errors({"a": 1, "b": 2}) == [error("/b", "unevaluatedProperties", "unknown field")]


def test_unevaluated_items_sees_only_the_prefix_a_tuple_items_evaluates():
    validator = compile({"items": [{"type": "number"}], "unevaluatedItems": False})
    assert validator.check([1]) is True
    assert validator.errors([1, 2]) == [error("/1", "unevaluatedItems", "unexpected item")]


# values


def test_ignores_required_entries_that_are_not_strings():
    validator = compile({"required": [42, "x"]})
    assert validator.check({"x": 1}) is True
    assert validator.errors({}) == [error("/x", "required", "missing")]


def test_does_not_accept_every_value_for_a_type_the_draft_does_not_define():
    assert compile({"type": "constructor"}).check(1) is False


# PiDrei: pi also compares each answer with TypeBox's `Check`, which has no counterpart here.
def test_rejects_nan_and_infinity_as_a_number_or_an_integer_with_or_without_bounds():
    schemas = [
        {"type": "number"},
        {"type": "integer"},
        {"type": "number", "minimum": 0},
        {"type": ["number", "string"]},
        {"anyOf": [{"type": "number"}, {"type": "null"}]},
        {"type": "object", "properties": {"a": {"type": "number"}}},
    ]
    for schema in schemas:
        check = compile(schema).check
        for v in (math.nan, math.inf, -math.inf):
            value = {"a": v} if schema.get("type") == "object" else v
            assert check(value) is False, f"{schema}: {v}"
    assert compile({"type": "number"}).errors(math.nan) == [error("", "type", "expected number, got NaN")]


# options


def test_treats_format_as_an_annotation_unless_formats_are_given():
    schema = {"type": "string", "format": "email"}
    assert compile(schema).check("not an email") is True
    assert compile(schema, formats=FORMATS).check("not an email") is False
    assert compile(schema, formats=FORMATS).check("a@example.com") is True


def test_raises_for_a_ref_that_names_nothing():
    with pytest.raises(ValueError, match=r"Unresolved \$ref"):
        compile({"$ref": "https://example.com/missing.json"})
    with pytest.raises(ValueError, match=r"Unresolved \$ref"):
        compile({"$ref": "#/$defs/missing"})


# schemas


def test_compiles_typebox_schemas_as_they_are():
    # Type.Object({ name: Type.String(), tags: Type.Array(Type.String()) })
    validator = compile(
        {
            "type": "object",
            "properties": {"name": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}},
            "required": ["name", "tags"],
        }
    )
    assert validator.check({"name": "a", "tags": []}) is True
    assert validator.errors({"name": "a", "tags": [1]}) == [error("/tags/0", "type", "expected string, got integer")]


def test_never_modifies_a_schema():
    schema = {**copy.deepcopy(BLOCK), "$defs": {"s": {"type": "string", "pattern": "^a"}}}
    before = copy.deepcopy(schema)
    validator = compile(schema)
    assert validator.check({"kind": "text", "text": "x"}) is True
    assert len(validator.errors({"kind": "image", "width": -1})) == 1
    assert schema == before


def test_compiles_a_recursive_schema():
    tree = {
        "type": "object",
        "properties": {"value": {"type": "number"}, "children": {"type": "array", "items": {"$ref": "#"}}},
        "required": ["value"],
    }
    validator = compile(tree)
    assert validator.check({"value": 1, "children": [{"value": 2, "children": [{"value": 3}]}]}) is True
    assert validator.errors({"value": 1, "children": [{"value": 2, "children": [{}]}]}) == [
        error("/children/0/children/0/value", "required", "missing")
    ]


# #11697
def test_applies_a_relative_root_id_once_when_resolving_local_references():
    schema = {
        "$id": "schemas/tool.json",
        "type": "object",
        "properties": {"count": {"$ref": "#/$defs/count"}},
        "required": ["count"],
        "$defs": {"count": {"type": "integer"}},
    }
    assert compile(schema).check({"count": 2}) is True
    validator = compile(schema, unresolved_refs="reject")
    assert validator.check({"count": 2}) is True
    assert validator.errors({"count": "a"}) == [error("/count", "type", "expected integer, got string")]
    assert compile(schema, coerce=True).parse({"count": "2"}) == ParseResult(ok=True, value={"count": 2})


def test_applies_a_relative_root_id_once_in_schemas_with_unevaluated_keywords():
    props = {
        "$id": "schemas/tool.json",
        "type": "object",
        "properties": {"count": {"$ref": "#/$defs/count"}},
        "$defs": {"count": {"type": "integer"}},
        "unevaluatedProperties": False,
    }
    validator = compile(props, unresolved_refs="reject")
    assert validator.check({"count": 2}) is True
    assert validator.errors({"count": 2, "extra": 1}) == [error("/extra", "unevaluatedProperties", "unknown field")]
    items = {
        "$id": "schemas/list.json",
        "type": "array",
        "prefixItems": [{"$ref": "#/$defs/count"}],
        "$defs": {"count": {"type": "integer"}},
        "unevaluatedItems": False,
    }
    assert compile(items, unresolved_refs="reject").check([2]) is True
    assert compile(items).check([2, 3]) is False


def test_explains_errors_under_nested_resources_with_a_relative_id():
    cloned = {
        "$id": "https://example.com/root.json",
        "allOf": [{"$ref": "#/$defs/child"}],
        "$defs": {
            "child": {
                "$id": "schemas/tool.json",
                "$ref": "#/$defs/props",
                "unevaluatedProperties": False,
                "$defs": {"props": {"properties": {"count": {"type": "integer"}}}},
            },
        },
    }
    assert compile(cloned).errors({"count": 2, "extra": 1}) == [
        error("/extra", "unevaluatedProperties", "unknown field")
    ]
    branch = {
        "$id": "https://example.com/root.json",
        "anyOf": [{"$id": "a/b.json", "$ref": "#/$defs/n", "$defs": {"n": {"type": "integer"}}}, {"type": "boolean"}],
    }
    assert compile(branch).errors({}) == [error("", "type", "expected integer, got object")]


def test_raises_for_a_malformed_schema():
    with pytest.raises(ValueError, match="must be an object or a boolean"):
        compile({"properties": {"a": 1}})


# unresolved references

DANGLING = {"type": "object", "properties": {"t": {"$ref": "#/definitions/Missing"}}}
EXTERNAL = {"type": "object", "properties": {"t": {"$ref": "https://example.com/schemas/t.json"}}}


def test_unresolved_references_raise_when_the_schema_compiles_by_default():
    with pytest.raises(ValueError, match=r"Unresolved \$ref #/definitions/Missing"):
        compile(DANGLING)
    with pytest.raises(ValueError, match=r"Unresolved \$ref https://example.com/schemas/t.json"):
        compile(EXTERNAL)


def test_unresolved_references_reject_only_values_at_their_place():
    for schema in (DANGLING, EXTERNAL):
        validator = compile(schema, unresolved_refs="reject")
        assert validator.check({}) is True
        assert validator.check({"t": 1}) is False
    assert compile(DANGLING, unresolved_refs="reject").errors({"t": 1}) == [
        error("/t", "$ref", "unresolved reference #/definitions/Missing")
    ]


def test_unresolved_references_leave_values_at_their_place_uncoerced():
    validator = compile(DANGLING, unresolved_refs="reject", coerce=True)
    result = validator.parse({"t": "1"})
    assert result.ok is False
    assert result.errors == (error("/t", "$ref", "unresolved reference #/definitions/Missing"),)


def test_unresolved_references_reject_a_typebox_ref_without_its_definition():
    # Type.Object({ a: Type.Ref("A") })
    validator = compile(
        {"type": "object", "properties": {"a": {"$ref": "A"}}, "required": ["a"]}, unresolved_refs="reject"
    )
    assert validator.errors({"a": {"x": "1"}}) == [error("/a", "$ref", "unresolved reference A")]


# string lengths


def test_string_lengths_count_an_unpaired_surrogate_as_one_code_point_and_a_pair_as_one():
    assert compile({"type": "string", "minLength": 2}).check("\ud800A") is True
    assert compile({"type": "string", "maxLength": 1}).check("\ud800A") is False
    assert compile({"type": "string", "maxLength": 1}).check("\ud800") is True
    assert compile({"type": "string", "maxLength": 1}).check("😀") is True
    assert compile({"type": "string", "minLength": 2}).check("😀") is False
    assert compile({"type": "string", "minLength": 2}).check("\udc00\ud800") is True


# error messages


def test_error_messages_list_at_most_20_enum_values():
    values = [f"v{i}" for i in range(5000)]
    validator = compile({"type": "object", "properties": {"e": {"type": "string", "enum": values}}})
    assert validator.check({"e": "v4999"}) is True
    [first] = validator.errors({"e": "v5000"})
    listed = ",".join(f'"{value}"' for value in values[:20])
    assert first.message == f"expected one of [{listed}, ... and 4980 more]"
    assert compile({"enum": ["a", "b"]}).errors("c") == [error("", "enum", 'expected one of ["a","b"]')]


def test_error_messages_accept_a_10000_value_enum():
    values = [f"member{i}" for i in range(10_000)]
    assert compile({"enum": values}).check("member9999") is True


def test_error_messages_describe_contains_bounds_in_words():
    def at(schema: dict) -> str:
        return compile({"type": "array", **schema}).errors(["a"])[0].message

    assert at({"contains": {"type": "number"}}) == "expected at least 1 matching item"
    assert at({"contains": {"type": "number"}, "minContains": 2}) == "expected at least 2 matching items"
    assert at({"contains": {"type": "string"}, "maxContains": 0, "minContains": 0}) == (
        "expected at most 0 matching items"
    )
    assert at({"contains": {"type": "number"}, "minContains": 1, "maxContains": 3}) == (
        "expected between 1 and 3 matching items"
    )


def test_error_messages_blame_a_one_of_failure_on_the_branch_that_declares_most_of_the_objects_properties():
    filter_schema = {
        "type": "object",
        "properties": {"where": {"$ref": "#/$defs/F"}},
        "$defs": {
            "F": {
                "oneOf": [
                    {
                        "type": "object",
                        "properties": {"field": {"type": "string"}, "eq": {"type": "string"}},
                        "required": ["field", "eq"],
                        "additionalProperties": False,
                    },
                    {
                        "type": "object",
                        "properties": {"and": {"type": "array", "items": {"$ref": "#/$defs/F"}}},
                        "required": ["and"],
                        "additionalProperties": False,
                    },
                ]
            }
        },
    }
    assert compile(filter_schema).errors({"where": {"and": [{"field": "a", "eq": 1, "x": 2}]}}) == [
        error("/where/and/0/eq", "type", "expected string, got integer"),
        error("/where/and/0/x", "additionalProperties", "unknown field"),
    ]


def test_error_messages_explain_one_object_reached_through_the_same_reference_at_two_paths_at_each_path():
    schema = {
        "type": "object",
        "properties": {"a": {"$ref": "#/$defs/P"}, "b": {"$ref": "#/$defs/P"}},
        "$defs": {"P": {"type": "object", "properties": {"n": {"type": "integer"}}}},
    }
    shared = {"n": "x"}
    assert compile(schema).errors({"a": shared, "b": shared}) == [
        error("/a/n", "type", "expected integer, got string"),
        error("/b/n", "type", "expected integer, got string"),
    ]


def test_error_messages_write_numbers_as_javascript_does():
    # PiDrei: Python writes 1.0 and 1e-07 where the messages pi sends say 1 and 1e-7.
    assert compile({"type": "number", "minimum": 1.0}).errors(0) == [error("", "minimum", "expected >= 1")]
    assert compile({"type": "number", "maximum": 1e-7}).errors(1) == [error("", "maximum", "expected <= 1e-7")]
    assert compile({"const": [1.0, 0.5]}).errors(2) == [error("", "const", "expected [1,0.5]")]


# TypeBox's extra formats


def test_checks_url_and_json_pointer_uri_fragment():
    url = compile({"type": "string", "format": "url"}, formats=FORMATS)
    assert url.check("https://example.com/a?b=1") is True
    assert url.check("not a url") is False
    fragment = compile({"type": "string", "format": "json-pointer-uri-fragment"}, formats=FORMATS)
    assert fragment.check("#/a/b~1c") is True
    assert fragment.check("#/a b") is False
    assert fragment.check("/a") is False


def test_checks_iri_and_iri_reference():
    # PiDrei: jsonschema checks IRIs only with rfc3987-syntax, which is not installed; formats.py checks them.
    iri = compile({"type": "string", "format": "iri"}, formats=FORMATS)
    assert iri.check("https://例え.jp/パス?q=値") is True
    assert iri.check("relative/パス") is False
    assert iri.check("https://例え.jp/a b") is False
    reference = compile({"type": "string", "format": "iri-reference"}, formats=FORMATS)
    assert reference.check("relative/パス#片") is True
    assert reference.check("a b") is False


# TypeBox's JavaScript types
#
# PiDrei: a bigint is an `int`, so the number `5` passes `bigint` here (pi tells `5` from `5n`) and the
# `expected bigint, got integer` error has no value to fail on. Not mirrored: `Type.Undefined` and a required
# property set to undefined (a Python dict has no key set to undefined), and a bigint against a fractional
# `multipleOf` (an `int` there is a JSON number, a multiple of 0.5 when whole).


def test_checks_type_bigint_with_its_bounds_as_typebox_does():
    validator = compile({"anyOf": [{"type": "bigint", "minimum": 0}, {"type": "null"}]})
    for value in (5, 0, None):
        assert validator.check(value) is True
    # pi's `5` (a number, not a bigint) is `5.5` here: a whole `int` is a bigint.
    for value in (-1, "5", 5.5):
        assert validator.check(value) is False
    assert compile({"type": "bigint", "maximum": 10, "multipleOf": 5}).check(10) is True
    assert compile({"type": "bigint", "multipleOf": 5}).check(7) is False
    assert compile({"type": "bigint", "exclusiveMaximum": 3}).check(3) is False
    assert compile({"type": "bigint", "minimum": 0}).errors(-2) == [error("", "minimum", "expected >= 0")]
    # pi: errors(2) is `expected bigint, got integer`; a fractional number is the non-bigint number here.
    assert compile({"type": "bigint"}).errors(2.5) == [error("", "type", "expected bigint, got number")]


def test_checks_multiple_of_of_bigints_exactly_however_large():
    five = compile({"type": "bigint", "multipleOf": 5})
    assert five.check(10**400) is True
    assert five.check(10**400 + 1) is False
    # A bigint is no multiple of 0.
    assert compile({"multipleOf": 0}).check(4) is False
