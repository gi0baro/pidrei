"""Mirror of pi mcp src/result-schema.ts: the output schema of an MCP tool
for programs that call it, the whole `CallToolResult`, with the tool's own
output schema as `structuredContent`."""

from typing import Any


# MCP content blocks as one compact type with the fields of text and image blocks, which scripts
# read most. Rendered into every MCP tool's declaration.
_CONTENT_BLOCK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "type": {"enum": ["text", "image", "audio", "resource_link", "resource"]},
        "text": {"type": "string"},
        "data": {"type": "string"},
        "mimeType": {"type": "string"},
    },
    "required": ["type"],
    "additionalProperties": True,
}


def create_mcp_result_schema(structured_content_schema: dict[str, Any] | None) -> dict[str, Any]:
    """Output schema of an MCP tool: the `CallToolResult` programs receive
    (without `_meta`, which callers drop), with the tool's own output schema
    as `structuredContent`. Without one, structured content may be any JSON
    value."""
    return {
        "type": "object",
        "properties": {
            "content": {"type": "array", "items": _CONTENT_BLOCK_SCHEMA},
            "structuredContent": (
                _relocate_schema(structured_content_schema, "#/properties/structuredContent")
                if structured_content_schema
                else {}
            ),
            "isError": {"type": "boolean"},
        },
        "required": ["content"],
        # Fields newer protocol versions add.
        "additionalProperties": True,
    }


# Keywords whose values are data, so a `$ref` key in them is not a reference.
_SCHEMA_DATA_KEYWORDS = frozenset(("const", "enum", "default", "examples"))

# Keywords whose values map names to schemas, so their keys are names, not keywords.
_SCHEMA_MAP_KEYWORDS = frozenset(
    ("properties", "patternProperties", "dependentSchemas", "dependencies", "$defs", "definitions")
)


def _relocate_schema(schema: Any, pointer: str) -> Any:
    """`schema` as a subschema at the JSON pointer `pointer`
    (`#/properties/structuredContent`) of another schema: its local references
    (`#`, `#/$defs/Item`) are rewritten to point below `pointer`, so they
    resolve to the same subschemas as before."""
    # The server's schema is wire input of any JSON shape.
    if isinstance(schema, list):
        return [_relocate_schema(item, pointer) for item in schema]
    if not isinstance(schema, dict):
        return schema
    relocated: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "$ref" and isinstance(value, str) and (value == "#" or value.startswith("#/")):
            relocated[key] = f"{pointer}{value[1:]}"
        elif key in _SCHEMA_DATA_KEYWORDS:
            relocated[key] = value
        elif key in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            relocated[key] = {name: _relocate_schema(subschema, pointer) for name, subschema in value.items()}
        else:
            relocated[key] = _relocate_schema(value, pointer)
    return relocated
