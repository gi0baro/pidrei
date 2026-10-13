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


def _key_id_schema(end_anchor: str) -> dict[str, Any]:
    return {
        "type": "string",
        "pattern": (
            f"^(?!.*(?:{_DUPLICATE_MODIFIER_PATTERN}))(?:(?:{'|'.join(_MODIFIERS)})\\+){{0,4}}"
            f"{_BASE_KEY_PATTERN}{end_anchor}"
        ),
        "description": "Key identifier with optional ctrl, shift, alt, or super modifiers.",
    }


KEY_ID_SCHEMA = _key_id_schema("$")


def keybinding_value_schema(*, key_id_schema: dict[str, Any] = KEY_ID_SCHEMA, **options: Any) -> dict[str, Any]:
    return {"anyOf": [key_id_schema, {"type": "array", "items": key_id_schema}], **options}


KEYBINDING_VALUE_SCHEMA = keybinding_value_schema()

# What keybindings.json values are validated against at runtime. Python's `$` also matches
# before a final newline, where ECMAScript's (read by the editors the published schema serves)
# does not; `\Z` is the whole-string anchor, so "ctrl+x\n" is rejected by both.
RUNTIME_KEYBINDING_VALUE_SCHEMA = keybinding_value_schema(key_id_schema=_key_id_schema("\\Z"))
