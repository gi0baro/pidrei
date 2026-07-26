"""Mirror of pi tui src/terminal-colors.ts.

RgbColor is a camelCase-free record ``{"r": int, "g": int, "b": int}``;
TerminalColorScheme is the literal string "dark" | "light".
"""

import re


_OSC11_BACKGROUND_COLOR_RESPONSE_RE = re.compile(r"^\x1b\]11;([^\x07\x1b]*)(?:\x07|\x1b\\)$", re.IGNORECASE)
_COLOR_SCHEME_REPORT_RE = re.compile(r"^\x1b\[\?997;(1|2)n$")
_HEX_CHANNEL_RE = re.compile(r"^[0-9a-f]+$", re.IGNORECASE)
_HEX6_RE = re.compile(r"^[0-9a-f]{6}$", re.IGNORECASE)
_HEX12_RE = re.compile(r"^[0-9a-f]{12}$", re.IGNORECASE)
_RGB_PREFIX_RE = re.compile(r"^rgba?:", re.IGNORECASE)


def _hex_to_rgb(hex_value: str) -> dict:
    normalized = hex_value.removeprefix("#")
    return {
        "r": int(normalized[0:2], 16),
        "g": int(normalized[2:4], 16),
        "b": int(normalized[4:6], 16),
    }


def _parse_osc_hex_channel(channel: str) -> int | None:
    if not _HEX_CHANNEL_RE.match(channel):
        return None
    maximum = 16 ** len(channel) - 1
    if maximum <= 0:
        return None
    return round((int(channel, 16) / maximum) * 255)


def is_osc11_background_color_response(data: str) -> bool:
    return _OSC11_BACKGROUND_COLOR_RESPONSE_RE.match(data) is not None


def parse_osc11_background_color(data: str) -> dict | None:
    match = _OSC11_BACKGROUND_COLOR_RESPONSE_RE.match(data)
    if not match:
        return None

    value = match.group(1).strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if _HEX6_RE.match(hex_value):
            return _hex_to_rgb(value)
        if _HEX12_RE.match(hex_value):
            r = _parse_osc_hex_channel(hex_value[0:4])
            g = _parse_osc_hex_channel(hex_value[4:8])
            b = _parse_osc_hex_channel(hex_value[8:12])
            return {"r": r, "g": g, "b": b} if r is not None and g is not None and b is not None else None
        return None

    rgb_value = _RGB_PREFIX_RE.sub("", value)
    channels = rgb_value.split("/")
    if len(channels) < 3:
        return None
    r = _parse_osc_hex_channel(channels[0])
    g = _parse_osc_hex_channel(channels[1])
    b = _parse_osc_hex_channel(channels[2])
    return {"r": r, "g": g, "b": b} if r is not None and g is not None and b is not None else None


def parse_terminal_color_scheme_report(data: str) -> str | None:
    match = _COLOR_SCHEME_REPORT_RE.match(data)
    if not match:
        return None
    return "light" if match.group(1) == "2" else "dark"
