"""Mirror of pi coding-agent src/core/key-id-schema.ts.

The key identifier grammar keybindings.json values must follow: up to four
distinct modifiers, then a letter, a digit, or a named `Key`. The pattern is
also published in the keybindings schema, so it is written in the subset
ECMAScript and Python regexes share (pi's escape set, not `re.escape`, which
escapes characters ECMAScript's unicode mode rejects).
"""

import re
from typing import Any

from pidrei_tui import Key


_MODIFIERS = ("ctrl", "shift", "alt", "super")
_KEY_VALUES = [value for name, value in vars(Key).items() if not name.startswith("_") and isinstance(value, str)]
_BASE_KEY_PATTERN = "(?:[a-z0-9]|{})".format(
    "|".join(re.sub(r"[\\^$.*+?()[\]{}|]", lambda match: f"\\{match.group(0)}", value) for value in _KEY_VALUES)
)
_DUPLICATE_MODIFIER_PATTERN = "|".join(f"{modifier}\\+.*{modifier}\\+" for modifier in _MODIFIERS)

KEY_ID_SCHEMA: dict[str, Any] = {
    "type": "string",
    "pattern": (
        f"^(?!.*(?:{_DUPLICATE_MODIFIER_PATTERN}))(?:(?:{'|'.join(_MODIFIERS)})\\+){{0,4}}{_BASE_KEY_PATTERN}$"
    ),
    "description": "Key identifier with optional ctrl, shift, alt, or super modifiers.",
}


def keybinding_value_schema(**options: Any) -> dict[str, Any]:
    return {"anyOf": [KEY_ID_SCHEMA, {"type": "array", "items": KEY_ID_SCHEMA}], **options}


KEYBINDING_VALUE_SCHEMA = keybinding_value_schema()
