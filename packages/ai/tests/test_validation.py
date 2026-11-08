"""Mirror of pi's validation.test.ts.

pi's first case guards TypeBox's Compile fallback when the Function
constructor is unavailable, a TypeBox-only concern with no pidrei
equivalent; its observable assertion (string "42" coerces to 42 inside an
object schema) is covered here directly. pi's TypeBox schemas are written as
the JSON Schema they emit.
"""

import pytest

from pidrei_ai.types import Tool, ToolCall
from pidrei_ai.utils.validation import validate_tool_arguments, validate_tool_call


def create_tool_call_with_plain_schema(schema: dict, value) -> tuple[Tool, ToolCall]:
    tool = Tool(
        name="echo",
        description="Echo tool",
        parameters={"type": "object", "properties": {"value": schema}, "required": ["value"]},
    )
    tool_call = ToolCall(id="tool-1", name="echo", arguments={"value": value})
    return tool, tool_call


def test_coerces_object_property_numbers():
    tool = Tool(
        name="echo",
        description="Echo tool",
        parameters={"type": "object", "properties": {"count": {"type": "number"}}, "required": ["count"]},
    )
    tool_call = ToolCall(id="tool-1", name="echo", arguments={"count": "42"})

    assert validate_tool_arguments(tool, tool_call) == {"count": 42}


@pytest.mark.parametrize(
    ("schema", "input_value", "expected"),
    [
        ({"type": "number"}, "42", 42),
        ({"type": "number"}, True, 1),
        ({"type": "number"}, None, 0),
        ({"type": "integer"}, "42", 42),
        ({"type": "boolean"}, "true", True),
        ({"type": "boolean"}, "false", False),
        ({"type": "boolean"}, 1, True),
        ({"type": "boolean"}, 0, False),
        # TypeBox's conversion rules come first: null is "null" for a required string.
        ({"type": "string"}, None, "null"),
        ({"type": "string"}, True, "true"),
        ({"type": "null"}, "", None),
        ({"type": "null"}, 0, None),
        ({"type": "null"}, False, None),
        ({"type": ["number", "string"]}, "1", "1"),
        ({"type": ["boolean", "number"]}, "1", 1),
        ({"type": "boolean"}, "1", True),
        ({"type": "boolean"}, "0", False),
        ({"type": "null"}, "null", None),
        ({"type": "integer"}, "42.1", 42),
        ({"type": "array", "items": {"type": "number"}}, "7", [7]),
    ],
)
def test_coerces_serialized_plain_json_schemas_like_typebox_schemas(schema, input_value, expected):
    tool, tool_call = create_tool_call_with_plain_schema(schema, input_value)
    result = validate_tool_arguments(tool, tool_call)
    assert result == {"value": expected}
    assert type(result["value"]) is type(expected)


def test_treats_null_as_omission_for_optional_non_nullable_properties():
    tool = Tool(
        name="echo",
        description="Echo tool",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "number"},
                "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "metadata": {"type": "object", "properties": {"enabled": {"type": "boolean"}}},
            },
            "required": ["path", "metadata"],
        },
    )
    tool_call = ToolCall(
        id="tool-1",
        name="echo",
        arguments={"path": "file.txt", "offset": None, "nullable": None, "metadata": {"enabled": None}},
    )

    assert validate_tool_arguments(tool, tool_call) == {"path": "file.txt", "nullable": None, "metadata": {}}


def test_preserves_optional_nulls_whose_referenced_schema_is_nullable():
    tool = Tool(
        name="echo",
        description="Echo tool",
        parameters={
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "$defs": {"value": {"anyOf": [{"type": "number"}, {"type": "null"}]}},
        },
    )
    tool_call = ToolCall(id="tool-1", name="echo", arguments={"value": None})

    assert validate_tool_arguments(tool, tool_call) == {"value": None}


def test_preserves_a_value_that_already_matches_a_nullable_union_arm():
    tool, tool_call = create_tool_call_with_plain_schema({"anyOf": [{"type": "number"}, {"type": "null"}]}, None)

    assert validate_tool_arguments(tool, tool_call) == {"value": None}


def test_preserves_a_value_that_already_matches_a_one_of_nullable_union_arm():
    tool, tool_call = create_tool_call_with_plain_schema({"oneOf": [{"type": "number"}, {"type": "null"}]}, None)

    assert validate_tool_arguments(tool, tool_call) == {"value": None}


def test_still_coerces_nullable_unions_when_the_original_value_does_not_match_any_arm():
    tool, tool_call = create_tool_call_with_plain_schema({"anyOf": [{"type": "number"}, {"type": "null"}]}, "42")

    assert validate_tool_arguments(tool, tool_call) == {"value": 42}


def test_accepts_null_for_nullable_array_schemas_with_items():
    tool, tool_call = create_tool_call_with_plain_schema({"type": ["array", "null"], "items": {"type": "string"}}, None)

    assert validate_tool_arguments(tool, tool_call) == {"value": None}


@pytest.mark.parametrize(
    ("schema", "input_value"),
    [
        ({"type": "boolean"}, "yes"),
        ({"type": "number"}, "abc"),
        ({"type": "null"}, "nil"),
        ({"type": "integer"}, "4x"),
    ],
)
def test_rejects_invalid_coercions_for_serialized_plain_json_schemas(schema, input_value):
    tool, tool_call = create_tool_call_with_plain_schema(schema, input_value)
    with pytest.raises(ValueError, match="Validation failed"):
        validate_tool_arguments(tool, tool_call)


def test_lists_pi_schemas_errors_then_the_arguments_it_received():
    tool, tool_call = create_tool_call_with_plain_schema({"type": "number"}, "many")
    with pytest.raises(ValueError) as excinfo:
        validate_tool_arguments(tool, tool_call)
    assert str(excinfo.value) == (
        'Validation failed for tool "echo":\n  - value: expected number, got string\n\n'
        'Received arguments:\n{\n  "value": "many"\n}'
    )


def test_reports_a_missing_required_property_at_the_property():
    tool, tool_call = create_tool_call_with_plain_schema({"type": "string"}, "x")
    missing = ToolCall(id=tool_call.id, name=tool_call.name, arguments={})
    with pytest.raises(ValueError, match="  - value: missing"):
        validate_tool_arguments(tool, missing)


def test_returns_a_copy_of_the_arguments_also_when_nothing_is_coerced():
    tool, tool_call = create_tool_call_with_plain_schema({"type": "string"}, "x")
    args = validate_tool_arguments(tool, tool_call)
    assert args == {"value": "x"}
    assert args is not tool_call.arguments


def test_validate_tool_call_finds_tool_by_name():
    tool, tool_call = create_tool_call_with_plain_schema({"type": "string"}, "hello")
    assert validate_tool_call([tool], tool_call) == {"value": "hello"}

    missing_call = ToolCall(id="tool-2", name="unknown", arguments={})
    with pytest.raises(ValueError, match='Tool "unknown" not found'):
        validate_tool_call([tool], missing_call)


def test_reports_malformed_arguments_json_instead_of_validating_empty_arguments():
    # pi #10897
    tool = Tool(
        name="echo",
        description="Echo tool",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}},
    )
    tool_call = ToolCall(id="tool-1", name="echo", arguments={}, arguments_error="Unexpected token")

    with pytest.raises(ValueError, match='Invalid arguments for tool "echo": Unexpected token'):
        validate_tool_arguments(tool, tool_call)
