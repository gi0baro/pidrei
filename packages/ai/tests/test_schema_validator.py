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

import pytest

from pidrei_ai.utils.schema import FORMATS, ValidationError, compile


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


# values


def test_does_not_accept_every_value_for_a_type_the_draft_does_not_define():
    assert compile({"type": "constructor"}).check(1) is False


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
