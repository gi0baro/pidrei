"""Mirror of pi coding-agent src/modes/interactive/theme/theme-schema.ts.

Theme JSON validation, kept out of `theme.py` on purpose: `theme.py` accepts
documents as-is unless `set_theme_json_validator` installs this validator,
which `main.py` does before anything can load a user-authored theme, as pi
does. Built-in themes do not need validation.

pi validates with TypeBox; here the same schema runs through `jsonschema`, so
the error *text* differs from TypeBox's. The message layout (missing color
tokens, then "  - <path>: <message>" lines) is pi's, and an unexpected key is
reported at its own path.
"""

from typing import Any

import jsonschema

from .theme_tokens import THEME_TOKENS


def _color_value(description: str | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "anyOf": [
            {
                "type": "string",
                "description": (
                    "Hex color (#RGB or #RRGGBB), OKLCH or OKHSL color, variable reference, or empty string for "
                    "terminal default"
                ),
            },
            {"type": "integer", "minimum": 0, "maximum": 255, "description": "256-color palette index (0-255)"},
        ]
    }
    if description:
        schema["description"] = description
    return schema


COLOR_VALUE_SCHEMA = _color_value()

_THEME_COLORS_SCHEMA = {
    "type": "object",
    "required": [name for name, descriptor in THEME_TOKENS.items() if "fallback" not in descriptor],
    "properties": {name: _color_value(descriptor["description"]) for name, descriptor in THEME_TOKENS.items()},
    "description": (
        "Theme color definitions (scrollbar, thinkingMax, and search highlight colors are optional and use "
        "compatible fallbacks)"
    ),
    "additionalProperties": False,
}

THEME_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "colors"],
    "properties": {
        "$schema": {"type": "string", "description": "JSON schema reference"},
        "name": {
            "type": "string",
            "pattern": "^[^/]+$",
            "description": (
                "Theme name. Must not contain '/' because it is reserved for automatic light/dark theme settings."
            ),
        },
        "appearance": {
            "anyOf": [{"type": "string", "const": "dark"}, {"type": "string", "const": "light"}],
            "description": "Background the theme is designed for. Detected from the theme colors when omitted.",
        },
        "vars": {
            "type": "object",
            "additionalProperties": COLOR_VALUE_SCHEMA,
            "description": "Reusable color variables",
        },
        "colors": _THEME_COLORS_SCHEMA,
        "export": {
            "type": "object",
            "properties": {
                "pageBg": _color_value("Page background color"),
                "cardBg": _color_value("Card/container background color"),
                "infoBg": _color_value("Info sections background (system prompt, notices)"),
            },
            "description": "Optional colors for HTML export (defaults derived from userMessageBg if not specified)",
            "additionalProperties": False,
        },
    },
    "title": "PiDrei Coding Agent Theme",
    "description": "Theme schema for PiDrei coding agent",
    "additionalProperties": False,
}

_VALIDATOR = jsonschema.Draft202012Validator(THEME_JSON_SCHEMA)


def _instance_path(error: jsonschema.ValidationError) -> str:
    return "".join(f"/{part}" for part in error.absolute_path)


def validate_theme_json(label: str, json_value: Any) -> dict:
    """Validate one theme document, raising a message that names the offending tokens."""
    if isinstance(json_value, dict) and isinstance(json_value.get("name"), str) and "/" in json_value["name"]:
        raise ValueError(
            f'Invalid theme name "{json_value["name"]}": theme names cannot contain "/" '
            "because it is reserved for automatic light/dark theme settings."
        )

    errors = list(_VALIDATOR.iter_errors(json_value))
    if not errors:
        return json_value

    missing_colors: set[str] = set()
    other_errors: list[str] = []
    for error in errors:
        path = _instance_path(error)
        if error.validator == "required" and path == "/colors":
            missing_colors.update(name for name in error.validator_value if name not in error.instance)
            continue
        if error.validator == "additionalProperties" and error.validator_value is False:
            known = error.schema.get("properties", {})
            for key in error.instance:
                if key not in known:
                    other_errors.append(
                        f"  - {path}/{key}: Additional properties are not allowed ({key!r} was unexpected)"
                    )
            continue
        other_errors.append(f"  - {path or '/'}: {error.message}")

    error_message = f'Invalid theme "{label}":\n'
    if missing_colors:
        error_message += "\nMissing required color tokens:\n"
        error_message += "\n".join(f"  - {color}" for color in sorted(missing_colors))
        error_message += '\n\nPlease add these colors to your theme\'s "colors" object.'
        error_message += "\nSee the built-in themes (dark.json, light.json) for reference values."
    if other_errors:
        error_message += "\n\nOther errors:\n" + "\n".join(other_errors)
    raise ValueError(error_message)
