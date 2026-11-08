"""Mirror of pi's google-shared-convert-tools.test.ts."""

import copy
import json

import pytest

from pidrei_ai.api.google_shared import (
    convert_tools,
    resolve_google_function_calling_mode,
    supports_google_strict_tool_sampling,
)
from pidrei_ai.types import JsonSchemaConstrainedSampling, Tool


def make_tool(parameters: dict) -> Tool:
    return Tool(name="test_tool", description="A test tool", parameters=parameters)


def test_strips_json_schema_meta_keys_from_parameters_when_use_parameters_true():
    tools = [
        make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$id": "urn:bash-tool",
                "$comment": "A bash tool for demonstration",
                "$defs": {"commandDef": {"type": "string"}},
                "definitions": {"legacyDef": {"type": "number"}},
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }
        )
    ]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl["parameters"] == {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    assert "$schema" not in decl["parameters"]
    assert "$id" not in decl["parameters"]
    assert "$comment" not in decl["parameters"]
    assert "$defs" not in decl["parameters"]
    assert "definitions" not in decl["parameters"]


def test_recursively_strips_nested_json_schema_meta_keys():
    tools = [
        make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "deep": {
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "$id": "urn:nested",
                        "type": "string",
                    }
                },
            }
        )
    ]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl["parameters"] == {"type": "object", "properties": {"deep": {"type": "string"}}}


def test_preserves_ref_while_stripping_meta_keys():
    tools = [
        make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"refProp": {"$ref": "#/$defs/someDef", "type": "string"}},
            }
        )
    ]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl["parameters"] == {
        "type": "object",
        "properties": {"refProp": {"$ref": "#/$defs/someDef", "type": "string"}},
    }


def test_does_not_mutate_the_original_tool_parameters_object():
    original_parameters = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }
    tools = [make_tool(original_parameters)]

    convert_tools(tools, True)

    assert original_parameters == {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }


def test_preserves_schema_in_parameters_json_schema_when_use_parameters_false():
    tools = [
        make_tool(
            {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }
        )
    ]

    result = convert_tools(tools, False)
    decl = result[0]["functionDeclarations"][0]

    assert decl["parametersJsonSchema"] == {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }


def test_handles_tools_without_schema_gracefully():
    tools = [make_tool({"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]})]

    result = convert_tools(tools, True)
    decl = result[0]["functionDeclarations"][0]

    assert decl["parameters"] == {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }


def test_uses_validated_function_calling_for_strict_tools_on_gemini_3():
    tool = make_tool({"type": "object", "properties": {}})
    tool.constrained_sampling = JsonSchemaConstrainedSampling(strict="require")

    assert supports_google_strict_tool_sampling("gemini-3.1-pro-preview") is True
    assert supports_google_strict_tool_sampling("gemini-2.5-pro") is False
    assert resolve_google_function_calling_mode([tool], None, True) == "VALIDATED"
    with pytest.raises(ValueError, match='Tool "test_tool" requires JSON-schema constrained sampling'):
        resolve_google_function_calling_mode([tool], None, False)


def test_returns_none_for_empty_tool_list():
    assert convert_tools([]) is None
    assert convert_tools([], True) is None


# --- const handling (regression coverage for #11308) ---------------------------


def _json_schema(parameters: dict) -> dict:
    return convert_tools([make_tool(parameters)])[0]["functionDeclarations"][0]["parametersJsonSchema"]


@pytest.mark.parametrize("key", ["default", "examples", "enum", "const", "x-custom"])
def test_preserves_literal_data_under(key):
    literal = {"const": "literal-data", "properties": {"nested": {"const": 1}}}
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "object", key: [literal] if key in ("enum", "examples") else literal}},
    }
    original = copy.deepcopy(parameters)

    assert _json_schema(parameters) == original
    assert parameters == original


@pytest.mark.parametrize(
    "key", ["properties", "patternProperties", "$defs", "definitions", "dependentSchemas", "dependencies"]
)
def test_preserves_names_while_rewriting_schemas_in(key):
    parameters = {
        "type": "object",
        key: {
            "__proto__": {"const": "x"},
            "const": {"const": "y"},
            "properties": {"const": "z"},
        },
    }
    original = copy.deepcopy(parameters)
    expected = {
        "type": "object",
        key: {
            "__proto__": {"enum": ["x"]},
            "const": {"enum": ["y"]},
            "properties": {"enum": ["z"]},
        },
    }

    result = _json_schema(parameters)
    assert result == expected
    assert json.dumps(result) == json.dumps(expected)
    assert parameters == original


def test_rewrites_only_schema_bearing_keywords():
    parameters = {
        "type": "object",
        "additionalProperties": {"const": "x"},
        "unevaluatedProperties": False,
        "propertyNames": {"const": "name"},
        "allOf": [{"if": {"const": "x"}, "then": {"const": "y"}, "else": {"not": {"const": "z"}}}],
        "properties": {
            "list": {
                "type": "array",
                "items": [{"const": 1}],
                "prefixItems": [{"const": 2}],
                "additionalItems": {"const": 3},
                "contains": {"const": 4},
                "unevaluatedItems": {"const": 5},
            },
        },
        "dependencies": {"list": ["other"]},
    }

    assert _json_schema(parameters) == {
        "type": "object",
        "additionalProperties": {"enum": ["x"]},
        "unevaluatedProperties": False,
        "propertyNames": {"enum": ["name"]},
        "allOf": [{"if": {"enum": ["x"]}, "then": {"enum": ["y"]}, "else": {"not": {"enum": ["z"]}}}],
        "properties": {
            "list": {
                "type": "array",
                "items": [{"enum": [1]}],
                "prefixItems": [{"enum": [2]}],
                "additionalItems": {"enum": [3]},
                "contains": {"enum": [4]},
                "unevaluatedItems": {"enum": [5]},
            },
        },
        "dependencies": {"list": ["other"]},
    }


def test_rewrites_string_and_number_const_values_to_single_value_enums_in_parameters_json_schema():
    parameters = {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "const": "zq-src-7"},
            "mode": {"anyOf": [{"type": "string", "const": "a"}, {"type": "string", "const": "b"}]},
            "level": {"oneOf": [{"const": 1}, {"const": 2}]},
            "flag": {"type": "boolean", "const": True},
            "list": {"type": "array", "items": {"const": "x"}},
            "const": {"type": "string"},
        },
        "required": ["origin"],
    }

    assert _json_schema(parameters) == {
        "type": "object",
        "properties": {
            "origin": {"type": "string", "enum": ["zq-src-7"]},
            "mode": {"anyOf": [{"type": "string", "enum": ["a"]}, {"type": "string", "enum": ["b"]}]},
            "level": {"oneOf": [{"enum": [1]}, {"enum": [2]}]},
            "flag": {"type": "boolean", "const": True},
            "list": {"type": "array", "items": {"enum": ["x"]}},
            "const": {"type": "string"},
        },
        "required": ["origin"],
    }
    assert parameters["properties"]["origin"] == {"type": "string", "const": "zq-src-7"}
