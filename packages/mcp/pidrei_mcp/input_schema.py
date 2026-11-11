"""Mirror of pi mcp src/input-schema.ts: an MCP tool's input schema in the
shape LLM tool APIs accept."""

from typing import Any


def normalize_mcp_input_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The tool's input schema as an object schema. Tool parameters must be
    objects, MCP servers may omit `type`, and some providers reject object
    schemas without `properties`. Other keywords are kept."""
    return {
        **schema,
        "type": schema["type"] if schema.get("type") is not None else "object",
        **({"properties": {}} if "properties" not in schema else {}),
    }
