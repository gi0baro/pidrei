"""Mirror of pi tui src/colors.ts.

A Color is one of three frozen records, told apart by ``kind`` (pi's tagged
union): `IndexedColor` ("indexed"), `RgbColorValue` ("rgb") and
`OklchColorValue` ("oklch"). RgbColor (plain channels) stays the
``{"r", "g", "b"}`` record of `terminal_colors`; OklchChannels and
OkhslChannels are ``{"l", "c", "h"}`` / ``{"h", "s", "l"}`` records.
TextAttributes / TextStyle are option dicts with pi's keys (``bold``,
``dim``, ``italic``, ``underline``, ``inverse``, ``strikethrough``, plus
``fg`` / ``bg`` colors for TextStyle). TerminalColorMode is "256color" |
"truecolor"; ColorMixSpace is "oklch" | "srgb".
"""

import math
import re
from dataclasses import dataclass
from typing import ClassVar

from .oklab import linear_srgb_to_rgb, okhsl_to_rgb, oklab_to_linear_srgb, rgb_to_okhsl, rgb_to_oklab


@dataclass(frozen=True, slots=True)
class IndexedColor:
    kind: ClassVar[str] = "indexed"
    index: int


@dataclass(frozen=True, slots=True)
class RgbColorValue:
    kind: ClassVar[str] = "rgb"
    r: float
    g: float
    b: float


@dataclass(frozen=True, slots=True)
class OklchColorValue:
    kind: ClassVar[str] = "oklch"
    l: float
    c: float
    h: float


# A concrete color. Every color can be converted to sRGB, so color math never fails.
type Color = IndexedColor | RgbColorValue | OklchColorValue


def _is_finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_finite(value, name: str) -> None:
    if not _is_finite(value):
        raise ValueError(f"{name} must be finite")


def _js_number(value) -> str:
    # JS template interpolation of a number: integral floats print bare.
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def indexed_color(index: int) -> IndexedColor:
    if not isinstance(index, int) or isinstance(index, bool) or index < 0 or index > 255:
        raise ValueError(f"ANSI color index must be an integer from 0 to 255: {_js_number(index)}")
    return IndexedColor(index)


def rgb_color(r: float, g: float, b: float) -> RgbColorValue:
    for name, value in (("r", r), ("g", g), ("b", b)):
        _require_finite(value, name)
        if value < 0 or value > 255:
            raise ValueError(f"{name} must be between 0 and 255: {_js_number(value)}")
    return RgbColorValue(r, g, b)


def oklch_color(l: float, c: float, h: float) -> OklchColorValue:
    _require_finite(l, "l")
    _require_finite(c, "c")
    _require_finite(h, "h")
    if l < 0 or l > 1:
        raise ValueError(f"l must be between 0 and 1: {_js_number(l)}")
    if c < 0:
        raise ValueError(f"c must not be negative: {_js_number(c)}")
    return OklchColorValue(l, c, math.fmod(math.fmod(h, 360) + 360, 360))


_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?"
_OKLCH_RE = re.compile(
    rf"^oklch\(\s*({_NUMBER_PATTERN})(%)?\s+({_NUMBER_PATTERN})\s+({_NUMBER_PATTERN})(?:deg)?\s*\)$",
    re.IGNORECASE,
)
_OKHSL_RE = re.compile(
    rf"^okhsl\(\s*({_NUMBER_PATTERN})(?:deg)?\s+({_NUMBER_PATTERN})(%)?\s+({_NUMBER_PATTERN})(%)?\s*\)$",
    re.IGNORECASE,
)
_HEX_RE = re.compile(r"^#([\da-f]{3}|[\da-f]{6})$", re.IGNORECASE)


def okhsl_color(h: float, s: float, l: float) -> RgbColorValue:
    """An OKHSL color, converted to sRGB. Saturation is relative to the sRGB
    gamut at the hue and lightness, so equal saturation looks equally colorful
    across hues and lightness.

    ``h`` in degrees; ``s`` and ``l`` 0-1.
    """
    _require_finite(h, "h")
    _require_finite(s, "s")
    _require_finite(l, "l")
    if s < 0 or s > 1:
        raise ValueError(f"s must be between 0 and 1: {_js_number(s)}")
    if l < 0 or l > 1:
        raise ValueError(f"l must be between 0 and 1: {_js_number(l)}")
    rgb = okhsl_to_rgb(h, s, l)
    return rgb_color(rgb["r"], rgb["g"], rgb["b"])


def color_to_okhsl(color: Color) -> dict:
    return rgb_to_okhsl(color_to_rgb(color))


def parse_color(value: str | int) -> Color:
    if isinstance(value, int) and not isinstance(value, bool):
        return indexed_color(value)

    hex_match = _HEX_RE.match(value)
    if hex_match:
        digits = hex_match.group(1)
        if len(digits) == 3:
            digits = "".join(digit + digit for digit in digits)
        return rgb_color(int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))

    oklch = _OKLCH_RE.match(value)
    if oklch:
        lightness = float(oklch.group(1)) / (100 if oklch.group(2) else 1)
        return oklch_color(lightness, float(oklch.group(3)), float(oklch.group(4)))

    okhsl = _OKHSL_RE.match(value)
    if okhsl:
        saturation = float(okhsl.group(2)) / (100 if okhsl.group(3) else 1)
        lightness = float(okhsl.group(4)) / (100 if okhsl.group(5) else 1)
        return okhsl_color(float(okhsl.group(1)), saturation, lightness)

    raise ValueError(f"Invalid color value: {value}")


_BASIC_COLORS = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (192, 192, 192),
    (128, 128, 128),
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
)
_CUBE_VALUES = (0, 95, 135, 175, 215, 255)
_GRAY_VALUES = tuple(8 + index * 10 for index in range(24))


def _indexed_to_rgb(index: int) -> dict:
    if index < 16:
        r, g, b = _BASIC_COLORS[index]
        return {"r": r, "g": g, "b": b}
    if index < 232:
        cube_index = index - 16
        return {
            "r": _CUBE_VALUES[cube_index // 36],
            "g": _CUBE_VALUES[(cube_index % 36) // 6],
            "b": _CUBE_VALUES[cube_index % 6],
        }
    gray = 8 + (index - 232) * 10
    return {"r": gray, "g": gray, "b": gray}


def _is_in_srgb_gamut(linear) -> bool:
    epsilon = 1e-7
    return all(-epsilon <= channel <= 1 + epsilon for channel in linear)


def _oklch_to_rgb(color: OklchColorValue) -> dict:
    # Gamut mapping keeps the hue fixed, so its direction is computed once and scaled by chroma.
    radians = (color.h * math.pi) / 180
    cos = math.cos(radians)
    sin = math.sin(radians)

    def at_chroma(chroma: float):
        return oklab_to_linear_srgb((color.l, chroma * cos, chroma * sin))

    direct = at_chroma(color.c)
    if _is_in_srgb_gamut(direct):
        return linear_srgb_to_rgb(direct)

    # Reduce chroma until the color fits. The achromatic color is always in
    # gamut, so it is the fallback when no bisection step fits, e.g.
    # `oklch(100% 0.3 150)` must map to white.
    linear = at_chroma(0)
    low = 0.0
    high = color.c
    for _ in range(20):
        chroma = (low + high) / 2
        candidate = at_chroma(chroma)
        if _is_in_srgb_gamut(candidate):
            low = chroma
            linear = candidate
        else:
            high = chroma
    return linear_srgb_to_rgb(linear)


def color_to_rgb(color: Color) -> dict:
    if isinstance(color, IndexedColor):
        return _indexed_to_rgb(color.index)
    if isinstance(color, RgbColorValue):
        return {"r": color.r, "g": color.g, "b": color.b}
    return _oklch_to_rgb(color)


def color_to_oklch(color: Color) -> dict:
    if isinstance(color, OklchColorValue):
        return {"l": color.l, "c": color.c, "h": color.h}
    lightness, a, b = rgb_to_oklab(color_to_rgb(color))
    return {"l": lightness, "c": math.hypot(a, b), "h": ((math.atan2(b, a) * 180) / math.pi + 360) % 360}


def _js_round(value: float) -> int:
    # JS Math.round: halves round toward +infinity.
    return math.floor(value + 0.5)


def color_to_hex(color: Color) -> str:
    rgb = color_to_rgb(color)
    return f"#{_js_round(rgb['r']):02x}{_js_round(rgb['g']):02x}{_js_round(rgb['b']):02x}"


def mix_colors(first: Color, second: Color, amount: float, space: str = "oklch") -> Color:
    _require_finite(amount, "amount")
    if amount < 0 or amount > 1:
        raise ValueError(f"amount must be between 0 and 1: {_js_number(amount)}")

    if space == "srgb":
        a = color_to_rgb(first)
        b = color_to_rgb(second)
        return rgb_color(
            a["r"] + (b["r"] - a["r"]) * amount,
            a["g"] + (b["g"] - a["g"]) * amount,
            a["b"] + (b["b"] - a["b"]) * amount,
        )

    a = color_to_oklch(first)
    b = color_to_oklch(second)
    first_hue = b["h"] if a["c"] < 1e-7 else a["h"]
    second_hue = first_hue if b["c"] < 1e-7 else b["h"]
    hue_delta = ((second_hue - first_hue + 540) % 360) - 180
    return oklch_color(
        a["l"] + (b["l"] - a["l"]) * amount,
        a["c"] + (b["c"] - a["c"]) * amount,
        first_hue + hue_delta * amount,
    )


def _find_closest(values, target: float) -> int:
    closest_index = 0
    closest_distance = math.inf
    for index, value in enumerate(values):
        distance = abs(target - value)
        if distance < closest_distance:
            closest_index = index
            closest_distance = distance
    return closest_index


def _color_distance(first: dict, second: dict) -> float:
    dr = first["r"] - second["r"]
    dg = first["g"] - second["g"]
    db = first["b"] - second["b"]
    return dr * dr * 0.299 + dg * dg * 0.587 + db * db * 0.114


def _rgb_to_ansi256(color: dict) -> int:
    r_index = _find_closest(_CUBE_VALUES, color["r"])
    g_index = _find_closest(_CUBE_VALUES, color["g"])
    b_index = _find_closest(_CUBE_VALUES, color["b"])
    cube_color = {"r": _CUBE_VALUES[r_index], "g": _CUBE_VALUES[g_index], "b": _CUBE_VALUES[b_index]}
    cube_index = 16 + 36 * r_index + 6 * g_index + b_index

    gray = _js_round(0.299 * color["r"] + 0.587 * color["g"] + 0.114 * color["b"])
    gray_offset = _find_closest(_GRAY_VALUES, gray)
    gray_value = _GRAY_VALUES[gray_offset]
    spread = max(color["r"], color["g"], color["b"]) - min(color["r"], color["g"], color["b"])
    if spread < 10 and _color_distance(color, {"r": gray_value, "g": gray_value, "b": gray_value}) < _color_distance(
        color, cube_color
    ):
        return 232 + gray_offset
    return cube_index


def _color_ansi(color: Color, mode: str, background: bool) -> str:
    code = 48 if background else 38
    if isinstance(color, IndexedColor):
        return f"\x1b[{code};5;{color.index}m"

    rgb = color_to_rgb(color)
    if mode == "truecolor":
        return f"\x1b[{code};2;{_js_round(rgb['r'])};{_js_round(rgb['g'])};{_js_round(rgb['b'])}m"
    return f"\x1b[{code};5;{_rgb_to_ansi256(rgb)}m"


def foreground_ansi(color: Color, mode: str) -> str:
    return _color_ansi(color, mode, False)


def background_ansi(color: Color, mode: str) -> str:
    return _color_ansi(color, mode, True)


def style_text(text: str, options: dict, mode: str) -> str:
    fg = options.get("fg")
    bg = options.get("bg")
    return style_text_with_ansi(
        text,
        foreground_ansi(fg, mode) if fg is not None else None,
        background_ansi(bg, mode) if bg is not None else None,
        options,
    )


def style_text_with_ansi(text: str, fg_ansi: str | None, bg_ansi: str | None, options: dict) -> str:
    """Like `style_text()`, but with precomputed color escape sequences, e.g.
    cached theme colors. Colors in ``options`` are ignored."""
    # Resets are prepended so they close in reverse order of the opening sequences.
    prefix = ""
    suffix = ""
    if fg_ansi:
        prefix += fg_ansi
        suffix = "\x1b[39m"
    if bg_ansi:
        prefix += bg_ansi
        suffix = f"\x1b[49m{suffix}"
    if options.get("bold"):
        prefix += "\x1b[1m"
    if options.get("dim"):
        prefix += "\x1b[2m"
    if options.get("bold") or options.get("dim"):
        suffix = f"\x1b[22m{suffix}"
    if options.get("italic"):
        prefix += "\x1b[3m"
        suffix = f"\x1b[23m{suffix}"
    if options.get("underline"):
        prefix += "\x1b[4m"
        suffix = f"\x1b[24m{suffix}"
    if options.get("inverse"):
        prefix += "\x1b[7m"
        suffix = f"\x1b[27m{suffix}"
    if options.get("strikethrough"):
        prefix += "\x1b[9m"
        suffix = f"\x1b[29m{suffix}"
    return f"{prefix}{text}{suffix}"
