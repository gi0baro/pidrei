"""Mirror of pi mcp test/result-schema.test.ts."""

from pidrei_mcp import create_mcp_result_schema


def test_declares_the_call_tool_result_with_any_structured_content_without_an_output_schema():
    schema = create_mcp_result_schema(None)
    assert list(schema["properties"]) == ["content", "structuredContent", "isError"]
    assert schema["properties"]["structuredContent"] == {}
    assert schema["required"] == ["content"]
    assert schema["additionalProperties"] is True


def test_moves_the_output_schemas_local_references_below_structured_content_keeping_data_and_remote_references():
    hit = {"type": "object", "properties": {"id": {"type": "string"}}}
    schema = create_mcp_result_schema(
        {
            "type": "object",
            "properties": {
                "hits": {"type": "array", "items": {"$ref": "#/$defs/Hit"}},
                "children": {"type": "array", "items": {"$ref": "#"}},
                # A property named like a keyword: its value is a schema.
                "default": {"$ref": "#/definitions/Hit"},
                "remote": {"$ref": "https://example.com/schema.json"},
            },
            "$defs": {"Hit": {**hit, "default": {"$ref": "#/stays"}}},
            "definitions": {"Hit": hit},
        }
    )
    moved = schema["properties"]["structuredContent"]
    at = "#/properties/structuredContent"
    assert moved["properties"]["hits"] == {"type": "array", "items": {"$ref": f"{at}/$defs/Hit"}}
    assert moved["properties"]["children"] == {"type": "array", "items": {"$ref": at}}
    assert moved["properties"]["default"] == {"$ref": f"{at}/definitions/Hit"}
    assert moved["properties"]["remote"] == {"$ref": "https://example.com/schema.json"}
    assert moved["$defs"]["Hit"]["default"] == {"$ref": "#/stays"}
