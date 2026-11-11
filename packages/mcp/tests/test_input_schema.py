"""Mirror of pi mcp test/input-schema.test.ts."""

from pidrei_mcp import normalize_mcp_input_schema


def test_adds_type_object_and_empty_properties_when_missing():
    assert normalize_mcp_input_schema({}) == {"type": "object", "properties": {}}
    assert normalize_mcp_input_schema({"type": "object"}) == {"type": "object", "properties": {}}


def test_keeps_the_schemas_own_keywords():
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    }
    assert normalize_mcp_input_schema(schema) == schema
    assert normalize_mcp_input_schema({"properties": {"a": {}}, "$defs": {"x": {}}}) == {
        "type": "object",
        "properties": {"a": {}},
        "$defs": {"x": {}},
    }
