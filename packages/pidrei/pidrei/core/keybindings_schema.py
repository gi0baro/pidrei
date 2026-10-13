"""Mirror of pi coding-agent src/core/keybindings-schema.ts.

The keybindings.json schema behind the published configuration schema: every
known action with its description, and any other key (extension actions)
taking a keybinding value.
"""

from typing import Any

from .key_id_schema import KEYBINDING_VALUE_SCHEMA, keybinding_value_schema
from .keybindings import KEYBINDINGS


__all__ = ["KEYBINDINGS_SCHEMA", "KEYBINDING_VALUE_SCHEMA"]

_PROPERTIES: dict[str, Any] = {"$schema": {"type": "string"}}
for _id, _definition in KEYBINDINGS.items():
    _PROPERTIES[_id] = keybinding_value_schema(description=_definition["description"])

KEYBINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": _PROPERTIES,
    "additionalProperties": KEYBINDING_VALUE_SCHEMA,
}
