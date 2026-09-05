"""Mirror of pi coding-agent src/modes/interactive/theme/theme-json.ts.

Theme JSON validation, kept out of `theme.py` on purpose.

pi split it because validating user-authored theme files needs typebox,
which a presentation that only uses built-in themes should never pay for.
The validation here is hand-rolled against pi's typebox schema (same
required color set, same error message layout), so the split is kept for
layout parity: `theme.py` accepts documents as-is unless
`set_theme_json_validator` installs this validator; `main.py` does so before
the first theme loads, as pi does.
"""

# ColorValue: hex "#ff0000", var ref "primary", empty "", or 256-color index.

_REQUIRED_COLORS = [
    # Core UI (10 colors)
    "accent",
    "border",
    "borderAccent",
    "borderMuted",
    "success",
    "error",
    "warning",
    "muted",
    "dim",
    "text",
    "thinkingText",
    # Scrollbar (2 optional: scrollbarTrack, scrollbarThumb)
    # Backgrounds & Content Text (11 required, 2 optional: searchMatchBg,
    # searchMatchText)
    "selectedBg",
    "userMessageBg",
    "userMessageText",
    "customMessageBg",
    "customMessageText",
    "customMessageLabel",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "toolTitle",
    "toolOutput",
    # Markdown (10 colors)
    "mdHeading",
    "mdLink",
    "mdLinkUrl",
    "mdCode",
    "mdCodeBlock",
    "mdCodeBlockBorder",
    "mdQuote",
    "mdQuoteBorder",
    "mdHr",
    "mdListBullet",
    # Tool Diffs (3 colors)
    "toolDiffAdded",
    "toolDiffRemoved",
    "toolDiffContext",
    # Syntax Highlighting (9 colors)
    "syntaxComment",
    "syntaxKeyword",
    "syntaxFunction",
    "syntaxVariable",
    "syntaxString",
    "syntaxNumber",
    "syntaxType",
    "syntaxOperator",
    "syntaxPunctuation",
    # Thinking Level Borders (6 colors; thinkingMax is optional)
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    # Bash Mode (1 color)
    "bashMode",
]

_EXPORT_KEYS = ("pageBg", "cardBg", "infoBg")


def _is_color_value(value) -> bool:
    if isinstance(value, str):
        return True
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255


def _collect_theme_errors(json_value) -> tuple[set, list]:
    """Validate against pi's theme schema shape.

    Returns ``(missing_colors, other_errors)`` where other errors are
    ``"  - <path>: <message>"`` strings like pi's typebox error listing.
    """
    missing_colors: set = set()
    other_errors: list = []

    if not isinstance(json_value, dict):
        other_errors.append("  - /: Expected object")
        return missing_colors, other_errors

    name = json_value.get("name")
    if name is None:
        other_errors.append("  - /: Expected required property")
    elif not isinstance(name, str):
        other_errors.append("  - /name: Expected string")

    schema_ref = json_value.get("$schema")
    if schema_ref is not None and not isinstance(schema_ref, str):
        other_errors.append("  - /$schema: Expected string")

    vars_value = json_value.get("vars")
    if vars_value is not None:
        if not isinstance(vars_value, dict):
            other_errors.append("  - /vars: Expected object")
        else:
            for key, value in vars_value.items():
                if not _is_color_value(value):
                    other_errors.append(f"  - /vars/{key}: Expected color value")

    colors_value = json_value.get("colors")
    if colors_value is None:
        other_errors.append("  - /: Expected required property")
    elif not isinstance(colors_value, dict):
        other_errors.append("  - /colors: Expected object")
    else:
        for key in _REQUIRED_COLORS:
            if key not in colors_value:
                missing_colors.add(key)
        for key, value in colors_value.items():
            if not _is_color_value(value):
                other_errors.append(f"  - /colors/{key}: Expected color value")

    export_value = json_value.get("export")
    if export_value is not None:
        if not isinstance(export_value, dict):
            other_errors.append("  - /export: Expected object")
        else:
            for key in _EXPORT_KEYS:
                value = export_value.get(key)
                if value is not None and not _is_color_value(value):
                    other_errors.append(f"  - /export/{key}: Expected color value")

    return missing_colors, other_errors


def validate_theme_json(label: str, json_value) -> dict:
    """Validate one theme document, raising a message that names the offending tokens."""
    missing_colors, other_errors = _collect_theme_errors(json_value)
    if missing_colors or other_errors:
        error_message = f'Invalid theme "{label}":\n'
        if missing_colors:
            error_message += "\nMissing required color tokens:\n"
            error_message += "\n".join(f"  - {color}" for color in sorted(missing_colors))
            error_message += '\n\nPlease add these colors to your theme\'s "colors" object.'
            error_message += "\nSee the built-in themes (dark.json, light.json) for reference values."
        if other_errors:
            error_message += "\n\nOther errors:\n" + "\n".join(other_errors)
        raise ValueError(error_message)

    if "/" in json_value["name"]:
        raise ValueError(
            f'Invalid theme name "{json_value["name"]}": theme names cannot contain "/" '
            "because it is reserved for automatic light/dark theme settings."
        )
    return json_value
