"""Mirror of pi coding-agent src/core/export-html/ansi-to-html.ts.

ANSI escape code to HTML converter (inline styles). Supports standard and
bright colors, the 256-color palette, RGB true color, bold/dim/italic/
underline, and reset.
"""

import re


# Standard ANSI color palette (0-15)
ANSI_COLORS = [
    "#000000",  # 0: black
    "#800000",  # 1: red
    "#008000",  # 2: green
    "#808000",  # 3: yellow
    "#000080",  # 4: blue
    "#800080",  # 5: magenta
    "#008080",  # 6: cyan
    "#c0c0c0",  # 7: white
    "#808080",  # 8: bright black
    "#ff0000",  # 9: bright red
    "#00ff00",  # 10: bright green
    "#ffff00",  # 11: bright yellow
    "#0000ff",  # 12: bright blue
    "#ff00ff",  # 13: bright magenta
    "#00ffff",  # 14: bright cyan
    "#ffffff",  # 15: bright white
]


def _color_256_to_hex(index: int) -> str:
    # Standard colors (0-15)
    if index < 16:
        return ANSI_COLORS[index]

    # Color cube (16-231): 6x6x6 = 216 colors
    if index < 232:
        cube_index = index - 16
        r = cube_index // 36
        g = (cube_index % 36) // 6
        b = cube_index % 6

        def to_hex(n: int) -> str:
            return format(0 if n == 0 else 55 + n * 40, "02x")

        return f"#{to_hex(r)}{to_hex(g)}{to_hex(b)}"

    # Grayscale (232-255): 24 shades
    gray_hex = format(8 + (index - 232) * 10, "02x")
    return f"#{gray_hex}{gray_hex}{gray_hex}"


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def _create_empty_style() -> dict:
    return {"fg": None, "bg": None, "bold": False, "dim": False, "italic": False, "underline": False}


def _style_to_inline_css(style: dict) -> str:
    parts: list = []
    if style["fg"]:
        parts.append(f"color:{style['fg']}")
    if style["bg"]:
        parts.append(f"background-color:{style['bg']}")
    if style["bold"]:
        parts.append("font-weight:bold")
    if style["dim"]:
        parts.append("opacity:0.6")
    if style["italic"]:
        parts.append("font-style:italic")
    if style["underline"]:
        parts.append("text-decoration:underline")
    return ";".join(parts)


def _has_style(style: dict) -> bool:
    return bool(style["fg"] or style["bg"] or style["bold"] or style["dim"] or style["italic"] or style["underline"])


def _apply_sgr_code(params: list, style: dict) -> None:
    """Parse ANSI SGR (Select Graphic Rendition) codes and update style."""
    i = 0
    while i < len(params):
        code = params[i]

        if code == 0:
            # Reset all
            style.update(_create_empty_style())
        elif code == 1:
            style["bold"] = True
        elif code == 2:
            style["dim"] = True
        elif code == 3:
            style["italic"] = True
        elif code == 4:
            style["underline"] = True
        elif code == 22:
            # Reset bold/dim
            style["bold"] = False
            style["dim"] = False
        elif code == 23:
            style["italic"] = False
        elif code == 24:
            style["underline"] = False
        elif 30 <= code <= 37:
            # Standard foreground colors
            style["fg"] = ANSI_COLORS[code - 30]
        elif code == 38:
            # Extended foreground color
            if i + 1 < len(params) and params[i + 1] == 5 and len(params) > i + 2:
                # 256-color: 38;5;N
                style["fg"] = _color_256_to_hex(params[i + 2])
                i += 2
            elif i + 1 < len(params) and params[i + 1] == 2 and len(params) > i + 4:
                # RGB: 38;2;R;G;B
                style["fg"] = f"rgb({params[i + 2]},{params[i + 3]},{params[i + 4]})"
                i += 4
        elif code == 39:
            # Default foreground
            style["fg"] = None
        elif 40 <= code <= 47:
            # Standard background colors
            style["bg"] = ANSI_COLORS[code - 40]
        elif code == 48:
            # Extended background color
            if i + 1 < len(params) and params[i + 1] == 5 and len(params) > i + 2:
                # 256-color: 48;5;N
                style["bg"] = _color_256_to_hex(params[i + 2])
                i += 2
            elif i + 1 < len(params) and params[i + 1] == 2 and len(params) > i + 4:
                # RGB: 48;2;R;G;B
                style["bg"] = f"rgb({params[i + 2]},{params[i + 3]},{params[i + 4]})"
                i += 4
        elif code == 49:
            # Default background
            style["bg"] = None
        elif 90 <= code <= 97:
            # Bright foreground colors
            style["fg"] = ANSI_COLORS[code - 90 + 8]
        elif 100 <= code <= 107:
            # Bright background colors
            style["bg"] = ANSI_COLORS[code - 100 + 8]
        # Ignore unrecognized codes

        i += 1


# Match ANSI escape sequences: ESC[ followed by params and ending with 'm'
_ANSI_REGEX = re.compile(r"\x1b\[([\d;]*)m")


def _parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def ansi_to_html(text: str) -> str:
    """Convert ANSI-escaped text to HTML with inline styles."""
    style = _create_empty_style()
    result = ""
    last_index = 0
    in_span = False

    for match in _ANSI_REGEX.finditer(text):
        # Add text before this escape sequence
        before_text = text[last_index : match.start()]
        if before_text:
            result += _escape_html(before_text)

        # Parse SGR parameters
        param_str = match.group(1)
        params = [_parse_int(p) for p in param_str.split(";")] if param_str else [0]

        # Close existing span if we have one
        if in_span:
            result += "</span>"
            in_span = False

        # Apply the codes
        _apply_sgr_code(params, style)

        # Open new span if we have any styling
        if _has_style(style):
            result += f'<span style="{_style_to_inline_css(style)}">'
            in_span = True

        last_index = match.end()

    # Add remaining text
    remaining_text = text[last_index:]
    if remaining_text:
        result += _escape_html(remaining_text)

    # Close any open span
    if in_span:
        result += "</span>"

    return result


def ansi_lines_to_html(lines: list) -> str:
    """Convert ANSI-escaped lines to HTML, one div per line."""
    return "".join(f'<div class="ansi-line">{ansi_to_html(line) or "&nbsp;"}</div>' for line in lines)
