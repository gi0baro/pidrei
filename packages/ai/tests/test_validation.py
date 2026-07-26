"""Mirror of pi's validation.test.ts.

pi's first case guards TypeBox's Compile fallback when the Function
constructor is unavailable — a TypeBox-only concern with no pidrei
equivalent; its observable assertion (string "42" coerces to 42 inside an
object schema) is covered here directly.
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
        ({"type": "string"}, None, ""),
        ({"type": "string"}, True, "true"),
        ({"type": "null"}, "", None),
        ({"type": "null"}, 0, None),
        ({"type": "null"}, False, None),
        ({"type": ["number", "string"]}, "1", "1"),
        ({"type": ["boolean", "number"]}, "1", 1),
    ],
)
def test_coerces_serialized_plain_json_schemas_with_ajv_compatible_primitive_rules(schema, input_value, expected):
    tool, tool_call = create_tool_call_with_plain_schema(schema, input_value)
    result = validate_tool_arguments(tool, tool_call)
    assert result == {"value": expected}
    assert type(result["value"]) is type(expected)


@pytest.mark.parametrize(
    ("schema", "input_value"),
    [
        ({"type": "boolean"}, "1"),
        ({"type": "boolean"}, "0"),
        ({"type": "null"}, "null"),
        ({"type": "integer"}, "42.1"),
    ],
)
def test_rejects_invalid_coercions_for_serialized_plain_json_schemas(schema, input_value):
    tool, tool_call = create_tool_call_with_plain_schema(schema, input_value)
    with pytest.raises(ValueError, match="Validation failed"):
        validate_tool_arguments(tool, tool_call)


def test_validate_tool_call_finds_tool_by_name():
    tool, tool_call = create_tool_call_with_plain_schema({"type": "string"}, "hello")
    assert validate_tool_call([tool], tool_call) == {"value": "hello"}

    missing_call = ToolCall(id="tool-2", name="unknown", arguments={})
    with pytest.raises(ValueError, match='Tool "unknown" not found'):
        validate_tool_call([tool], missing_call)


def test_error_message_structure_mirrors_pi():
    tool = Tool(
        name="edit",
        description="Edit tool",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "count": {"type": "number"}},
            "required": ["path", "count"],
        },
    )
    tool_call = ToolCall(id="tool-1", name="edit", arguments={"count": []})

    with pytest.raises(ValueError) as excinfo:
        validate_tool_arguments(tool, tool_call)

    message = str(excinfo.value)
    assert message.startswith('Validation failed for tool "edit":\n')
    assert "  - count: " in message  # localized error path
    assert "  - path: " in message  # missing required property path
    assert "Received arguments:" in message
