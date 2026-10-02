"""Mirror of pi coding-agent src/modes/interactive/theme/system-theme.ts.

The `system` theme: pi's colors derived from the terminal's own theme.

Every token belongs to a color family (its hue) and has contrast rules: it
must reach a contrast level on the background and on the panels it is drawn
on. Hue and saturation come from the terminal's palette color for the
family's ANSI slot, or from the family's own hue when the terminal reports no
palette. Lightness comes from the rules alone. Colors are built in OKHSL,
whose saturation is relative to the sRGB gamut, and fade toward gray near
black and white. A palette color never gains OKLCH chroma when it moves to
another lightness, so pastel palettes stay pastel.

A contrast level is a target-lightness curve: the OKLab lightness a token
needs, given the lightness of the surface below it. The curves were fitted to
the reference theme design from the "Pi themes: system and light/dark"
review. On dark backgrounds they aim for nearly fixed lightness; on light
backgrounds the required difference grows as the background darkens.

Depending on what the terminal reports, the theme is generated in one of
three tiers:
- background and palette: hues from the palette, lightness from the background;
- background only: the families' own hues, lightness from the background;
- nothing: ANSI palette indices and the default colors, which the terminal
  renders itself.

RgbColor is the ``{"r", "g", "b"}`` record of `pidrei_tui.terminal_colors`.
SystemThemeInput is a record with the optional keys ``foreground``,
``background``, ``palette`` (ANSI colors 0-15), ``saturation`` (0, grayscale,
to 1) and ``appearanceHint`` (the appearance when the terminal did not report
its background); SystemThemeColors is ``{"colors", "dim", "appearance"}``.
"""

import math

from pidrei_tui import (
    color_to_okhsl,
    color_to_oklch,
    color_to_rgb,
    okhsl_color,
    oklab_to_okhsl_lightness,
    oklch_color,
    rgb_color,
)


SYSTEM_THEME_NAME = "system"

# ============================================================================
# Recipe: color families and their tokens
# ============================================================================

# A family's OKHSL hue and saturation: `max` at mid lightness, falling toward
# `min` at black and white; `slot` is the ANSI palette slot the family takes
# its hue and saturation from.
_FAMILIES: dict[str, dict] = {
    "neutral": {"hue": 231.49, "saturation": {"min": 0.02, "max": 0.08}, "slot": 8},
    "blue": {"hue": 231.49, "saturation": {"min": 0.1, "max": 0.68}, "slot": 4},
    "green": {"hue": 158.68, "saturation": {"min": 0.1, "max": 0.76}, "slot": 2},
    "red": {"hue": 20, "saturation": {"min": 0.1, "max": 0.92}, "slot": 1},
    "yellow": {"hue": 82.36, "saturation": {"min": 0.5, "max": 1}, "slot": 3},
    "orange": {"hue": 52, "saturation": {"min": 0.12, "max": 0.85}, "slot": 3},
    "violet": {"hue": 295, "saturation": {"min": 0.2, "max": 0.6}, "slot": 5},
    "calamine": {"hue": 202.43, "saturation": {"min": 0.1, "max": 0.74}, "slot": 6},
    "thinkingSlate": {"hue": 231.49, "saturation": {"min": 0.08, "max": 0.2}, "slot": 4},
    "thinkingBlue": {"hue": 231.49, "saturation": {"min": 0.2, "max": 0.45}, "slot": 4},
    "thinkingPeriwinkle": {"hue": 263.25, "saturation": {"min": 0.3, "max": 0.6}, "slot": 6},
    "thinkingViolet": {"hue": 295, "saturation": {"min": 0.4, "max": 0.75}, "slot": 5},
    "thinkingMagenta": {"hue": 337.5, "saturation": {"min": 0.5, "max": 0.85}, "slot": 13},
    "thinkingRed": {"hue": 20, "saturation": {"min": 0.95, "max": 1}, "slot": 1},
}

_TOKEN_FAMILIES: dict[str, str] = {
    "selectedBg": "blue",
    "searchMatchBg": "orange",
    "userMessageBg": "blue",
    "customMessageBg": "violet",
    "toolPendingBg": "neutral",
    "toolSuccessBg": "green",
    "toolErrorBg": "red",
    "text": "neutral",
    "userMessageText": "neutral",
    "customMessageText": "neutral",
    "toolTitle": "neutral",
    "syntaxOperator": "neutral",
    "syntaxPunctuation": "neutral",
    "muted": "neutral",
    "dim": "neutral",
    "thinkingText": "neutral",
    "toolOutput": "neutral",
    "mdLinkUrl": "neutral",
    "mdQuote": "neutral",
    "mdQuoteBorder": "neutral",
    "mdHr": "neutral",
    "mdCodeBlockBorder": "neutral",
    "toolDiffContext": "neutral",
    "syntaxComment": "neutral",
    "scrollbarTrack": "neutral",
    "scrollbarThumb": "neutral",
    "searchMatchText": "neutral",
    "borderMuted": "neutral",
    "accent": "violet",
    "borderAccent": "violet",
    "customMessageLabel": "violet",
    "mdCode": "violet",
    "mdListBullet": "violet",
    "syntaxType": "violet",
    "border": "blue",
    "mdLink": "blue",
    "syntaxKeyword": "blue",
    "syntaxVariable": "calamine",
    "success": "green",
    "mdCodeBlock": "green",
    "toolDiffAdded": "green",
    "bashMode": "green",
    "syntaxNumber": "green",
    "error": "red",
    "toolDiffRemoved": "red",
    "warning": "yellow",
    "mdHeading": "yellow",
    "syntaxFunction": "yellow",
    "syntaxString": "orange",
    "thinkingOff": "neutral",
    "thinkingMinimal": "thinkingSlate",
    "thinkingLow": "thinkingBlue",
    "thinkingMedium": "thinkingPeriwinkle",
    "thinkingHigh": "thinkingViolet",
    "thinkingXhigh": "thinkingMagenta",
    "thinkingMax": "thinkingRed",
}

# Palette slots for tokens that would otherwise share a hue with a similar token.
_TOKEN_SLOTS: dict[str, int] = {"syntaxString": 2, "syntaxNumber": 5, "searchMatchBg": 3}

# ============================================================================
# Contrast levels and rules
# ============================================================================

# Target-lightness curves: a polynomial in the surface's OKLab lightness giving
# the OKLab lightness a token needs on it. `reachable` is the range of surface
# lightness where the level can be reached; beyond it the level is relaxed.
_LEVELS: dict[str, dict] = {
    "panel": {
        "dark": {"coefficients": (0.29131, -0.39746, 2.33185, -0.85524, -1.2076, 0.86276), "reachable": (0, 0.979)},
        "light": {
            "coefficients": (-3.74073, 27.94549, -78.44258, 112.6798, -79.60015, 22.11277),
            "reachable": (0.348, 1),
        },
    },
    "track": {
        "dark": {"coefficients": (0.39028, -0.23015, 0.83573, 2.43829, -4.38292, 2.01582), "reachable": (0, 0.946)},
        "light": {
            "coefficients": (-5.24921, 38.37322, -107.28833, 152.10005, -106.17127, 29.18061),
            "reachable": (0.368, 1),
        },
    },
    "thinking0": {
        "dark": {"coefficients": (0.52988, -0.05809, -0.30924, 4.63567, -6.52933, 2.89108), "reachable": (0, 0.873)},
        "light": {
            "coefficients": (-28.27749, 182.85284, -469.62416, 603.15916, -384.59976, 97.35147),
            "reachable": (0.51, 1),
        },
    },
    "thinking1": {
        "dark": {"coefficients": (0.55278, -0.03667, -0.45659, 4.95347, -6.90265, 3.0706), "reachable": (0, 0.858)},
        "light": {
            "coefficients": (-37.10484, 235.86282, -596.62344, 754.3633, -474.00763, 118.3551),
            "reachable": (0.535, 1),
        },
    },
    "thinking2": {
        "dark": {"coefficients": (0.57486, -0.01765, -0.58987, 5.25227, -7.27175, 3.25532), "reachable": (0, 0.842)},
        "light": {
            "coefficients": (-59.89653, 377.05024, -945.07843, 1182.03145, -734.96375, 181.68658),
            "reachable": (0.556, 1),
        },
    },
    "thinking3": {
        "dark": {"coefficients": (0.59621, -0.00062, -0.71148, 5.53588, -7.6392, 3.44606), "reachable": (0, 0.827)},
        "light": {
            "coefficients": (-72.07122, 445.84082, -1099.57352, 1353.88793, -829.53392, 202.26164),
            "reachable": (0.58, 1),
        },
    },
    "thinking4": {
        "dark": {"coefficients": (0.61691, 0.01462, -0.82288, 5.80651, -8.00641, 3.64333), "reachable": (0, 0.811)},
        "light": {
            "coefficients": (-110.14338, 674.21488, -1645.75941, 2004.32367, -1215.15899, 293.3183),
            "reachable": (0.6, 1),
        },
    },
    "thinking5": {
        "dark": {"coefficients": (0.63702, 0.02826, -0.92498, 6.06465, -8.37246, 3.84651), "reachable": (0, 0.795)},
        "light": {
            "coefficients": (-175.47701, 1063.54495, -2570.70594, 3098.80776, -1860.15527, 444.76392),
            "reachable": (0.62, 1),
        },
    },
    "thinking6": {
        "dark": {"coefficients": (0.65658, 0.04044, -1.01835, 6.30989, -8.73529, 4.05439), "reachable": (0, 0.779)},
        "light": {
            "coefficients": (-183.81712, 1094.70055, -2602.68539, 3088.71276, -1826.91131, 430.75931),
            "reachable": (0.643, 1),
        },
    },
    "subtle": {
        "dark": {"coefficients": (0.56762, -0.02475, -0.5383, 5.12628, -7.10931, 3.17324), "reachable": (0, 0.848)},
        "light": {
            "coefficients": (-232.85459, 1376.54473, -3249.11801, 3827.91186, -2248.29472, 526.55751),
            "reachable": (0.657, 1),
        },
    },
    "thumb": {
        "dark": {"coefficients": (0.60323, 0.00278, -0.73328, 5.57157, -7.68067, 3.46933), "reachable": (0, 0.823)},
        "light": {
            "coefficients": (-82.89897, 511.01355, -1255.98095, 1540.76821, -940.68087, 228.58523),
            "reachable": (0.586, 1),
        },
    },
    "readable": {
        "dark": {"coefficients": (0.66937, 0.04704, -1.06871, 6.43941, -8.9332, 4.17229), "reachable": (0, 0.77)},
        "light": {
            "coefficients": (-1554.52576, 8733.56817, -19604.93507, 21977.72696, -12300.99599, 2749.81288),
            "reachable": (0.751, 1),
        },
    },
    "emphasis": {
        "dark": {"coefficients": (0.7303, 0.07695, -1.31626, 7.1681, -10.14436, 4.92846), "reachable": (0, 0.712)},
        "light": {
            "coefficients": (-4948.31942, 26870.91986, -58334.48399, 63280.17197, -34298.01053, 7430.30146),
            "reachable": (0.811, 1),
        },
    },
    "textOnPanel": {
        "dark": {"coefficients": (0.86713, 0.05232, -0.89428, 4.79014, -5.5432, 1.75023), "reachable": (0, 0.542)},
        "light": {
            "coefficients": (-8570.89457, 43954.60805, -90084.00702, 92220.6791, -47152.15802, 9632.27113),
            "reachable": (0.867, 1),
        },
    },
    "text": {
        "dark": {"coefficients": (0.89242, 0.02311, -0.44862, 2.34417, -0.06084, -2.63844), "reachable": (0, 0.5)},
        "light": {
            "coefficients": (-2004.67048, 6664.47299, -6060.70202, -1792.61209, 5133.82359, -1939.85583),
            "reachable": (0.894, 1),
        },
    },
}

# A surface is a background token, "background" or "scrollbarTrack". A rule is
# ``(token, on, level)``: the token must reach the level on every surface in
# `on`.
_TOOL_PANELS = ["toolPendingBg", "toolSuccessBg", "toolErrorBg"]
_MESSAGE_PANELS = ["userMessageBg", "customMessageBg"]
_PANELS = [
    "userMessageBg",
    "toolPendingBg",
    "toolSuccessBg",
    "toolErrorBg",
    "selectedBg",
    "searchMatchBg",
    "customMessageBg",
]
_THINKING = [
    "thinkingOff",
    "thinkingMinimal",
    "thinkingLow",
    "thinkingMedium",
    "thinkingHigh",
    "thinkingXhigh",
    "thinkingMax",
]
_THINKING_LEVELS = ["thinking0", "thinking1", "thinking2", "thinking3", "thinking4", "thinking5", "thinking6"]


def _each(tokens: list, on: list, level: str) -> list:
    return [(token, on, level) for token in tokens]


_RULES: list[tuple[str, list, str]] = [
    *_each(_PANELS, ["background"], "panel"),
    ("text", ["background"], "text"),
    ("text", ["selectedBg"], "textOnPanel"),
    ("userMessageText", ["userMessageBg"], "textOnPanel"),
    ("toolTitle", _TOOL_PANELS, "textOnPanel"),
    *_each(["accent", "success", "error", "warning"], ["background", "selectedBg", *_TOOL_PANELS], "readable"),
    ("muted", ["background", "selectedBg", "customMessageBg", *_TOOL_PANELS], "readable"),
    ("dim", ["background", "selectedBg", "customMessageBg", *_TOOL_PANELS], "subtle"),
    ("thinkingText", ["background"], "readable"),
    ("customMessageText", ["customMessageBg", *_TOOL_PANELS], "readable"),
    ("customMessageLabel", ["background", "customMessageBg", "selectedBg", *_TOOL_PANELS], "readable"),
    ("toolOutput", ["background", *_TOOL_PANELS], "readable"),
    *_each(
        ["mdHeading", "mdLink", "mdLinkUrl", "mdCode", "mdQuote", "mdCodeBlockBorder", "mdListBullet"],
        ["background", *_MESSAGE_PANELS],
        "readable",
    ),
    ("mdCodeBlock", ["background", *_MESSAGE_PANELS, *_TOOL_PANELS], "readable"),
    *_each(["toolDiffAdded", "toolDiffRemoved", "toolDiffContext"], ["background", *_TOOL_PANELS], "readable"),
    *_each(
        [
            "syntaxComment",
            "syntaxKeyword",
            "syntaxFunction",
            "syntaxVariable",
            "syntaxString",
            "syntaxNumber",
            "syntaxType",
            "syntaxOperator",
            "syntaxPunctuation",
        ],
        ["background", *_MESSAGE_PANELS, *_TOOL_PANELS],
        "readable",
    ),
    ("searchMatchText", ["searchMatchBg"], "readable"),
    *_each(["bashMode", "border", "borderAccent"], ["background"], "readable"),
    ("borderMuted", ["background"], "subtle"),
    *_each(["mdQuoteBorder", "mdHr"], ["background", *_MESSAGE_PANELS, *_TOOL_PANELS], "readable"),
    ("scrollbarTrack", ["background"], "track"),
    ("scrollbarThumb", ["scrollbarTrack"], "thumb"),
    *((token, ["background"], _THINKING_LEVELS[index]) for index, token in enumerate(_THINKING)),
]

# Relaxation compresses levels stronger than this one toward it before weakening all levels.
_READABLE_FLOOR = {"dark": "readable", "light": "subtle"}

# Body text uses the terminal's foreground when it reaches this level, which is clearly stronger than muted.
_FOREGROUND_LEVEL = "emphasis"

# Text-level tokens that take the terminal's foreground.
_FOREGROUND_TOKENS = ["text", "userMessageText", "toolTitle"]

# WCAG 2 contrast ratio that body text must reach on the surfaces it is drawn on.
_TEXT_MINIMUM_WCAG_CONTRAST = 4.5


def _solve_order() -> list[str]:
    """Tokens in dependency order: every surface before the tokens drawn on it."""
    order: list[str] = []

    def visit(token: str) -> None:
        if token in order:
            return
        for rule_token, on, _level in _RULES:
            if rule_token != token:
                continue
            for surface in on:
                if surface != "background":
                    visit(surface)
        order.append(token)

    for rule_token, _on, _level in _RULES:
        visit(rule_token)
    return order


_SOLVE_ORDER = _solve_order()

# ============================================================================
# Public API
# ============================================================================


def _oklab_lightness(color: dict) -> float:
    """OKLab lightness of an sRGB color, 0-1."""
    return color_to_oklch(rgb_color(color["r"], color["g"], color["b"]))["l"]


def relative_luminance(color: dict) -> float:
    """WCAG 2 relative luminance."""

    def linear(channel: float) -> float:
        value = channel / 255
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return 0.2126 * linear(color["r"]) + 0.7152 * linear(color["g"]) + 0.0722 * linear(color["b"])


def wcag_contrast(first: dict, second: dict) -> float:
    """WCAG 2 contrast ratio, 1-21."""
    a = relative_luminance(first)
    b = relative_luminance(second)
    return (max(a, b) + 0.05) / (min(a, b) + 0.05)


def terminal_appearance(background: dict, foreground: dict | None = None) -> str:
    """Whether a terminal is dark or light, from its reported colors: the
    direction of its own foreground when text can be readable that way,
    otherwise dark when white text has more contrast on the background than
    black text."""
    white = {"r": 255, "g": 255, "b": 255}
    black = {"r": 0, "g": 0, "b": 0}
    white_contrast = wcag_contrast(white, background)
    black_contrast = wcag_contrast(black, background)
    if foreground:
        foreground_l = _oklab_lightness(foreground)
        background_l = _oklab_lightness(background)
        if abs(foreground_l - background_l) > 0.05:
            appearance = "dark" if foreground_l > background_l else "light"
            best = white_contrast if appearance == "dark" else black_contrast
            if best >= _TEXT_MINIMUM_WCAG_CONTRAST:
                return appearance
    return "dark" if white_contrast >= black_contrast else "light"


# ============================================================================
# Generation
# ============================================================================


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(maximum, max(minimum, value))


def _js_round(value: float) -> int:
    # JS Math.round: halves round toward +infinity.
    return math.floor(value + 0.5)


def _hex_of(color: dict) -> str:
    return "#" + "".join(f"{_js_round(color[channel]):02x}" for channel in ("r", "g", "b"))


def _gaussian(x: float) -> float:
    return math.exp(-((x - 0.5) ** 2) / (2 * 0.25**2))


def _bell_weight(lightness: float) -> float:
    """Saturation weight at a lightness: a Gaussian (center 0.5, sigma 0.25),
    0 at black and white, 1 in the middle."""
    return (_gaussian(lightness) - _gaussian(0)) / (1 - _gaussian(0))


def _saturation_curve(family: dict, lightness: float) -> float:
    """A family's saturation curve relative to its maximum: 1 at mid
    lightness, `min / max` at black and white."""
    minimum = family["saturation"]["min"]
    maximum = family["saturation"]["max"]
    floor = minimum / maximum if maximum > 0 else 1
    return floor + (1 - floor) * _bell_weight(lightness)


def _level_target(level: str, appearance: str, surface_l: float) -> float | None:
    """The target lightness for a level on a surface, or None where the level cannot be reached."""
    curve = _LEVELS[level][appearance]
    low, high = curve["reachable"]
    if surface_l < low or surface_l > high:
        return None
    return sum(coefficient * surface_l**power for power, coefficient in enumerate(curve["coefficients"]))


def _extreme_of(values: list, lighter: bool) -> float:
    # JS Math.max()/Math.min() of nothing: -Infinity / Infinity.
    if not values:
        return -math.inf if lighter else math.inf
    return max(values) if lighter else min(values)


def generate_system_theme_colors(input_: dict) -> dict:
    """Generate the system theme's colors from the terminal's reported colors."""
    requested_saturation = input_.get("saturation")
    saturation = _clamp(1 if requested_saturation is None else requested_saturation, 0, 1)
    background = input_.get("background")
    foreground = input_.get("foreground")
    if not background:
        return _indexed_colors(saturation, input_.get("appearanceHint"))
    input_palette = input_.get("palette")
    palette = [_source_of(color) for color in input_palette] if input_palette and len(input_palette) == 16 else None

    appearance = terminal_appearance(background, foreground)
    lighter = appearance == "dark"
    extreme = 1 if lighter else 0
    background_l = _oklab_lightness(background)

    def paint(token: str, oklab_l: float) -> dict:
        """A token's color at an OKLab lightness. With a palette, the palette
        color's saturation applies at its own lightness and falls off toward
        black and white along the family's curve, never rising above it."""
        lightness = oklab_to_okhsl_lightness(oklab_l)
        family = _FAMILIES[_TOKEN_FAMILIES[token]]
        if palette is None:
            minimum = family["saturation"]["min"]
            maximum = family["saturation"]["max"]
            return _rgb_of(
                okhsl_color(
                    family["hue"], (minimum + (maximum - minimum) * _bell_weight(lightness)) * saturation, lightness
                )
            )
        return _anchored(palette[_TOKEN_SLOTS.get(token, family["slot"])], family, lightness, saturation)

    def target(level: str, surface_l: float, t: float) -> float | None:
        """The lightness a rule needs on a surface, relaxed by `t`: from 0 to
        1, levels stronger than the readable floor move toward it; from 1 to
        2, all levels move toward the surface itself."""
        reached = _level_target(level, appearance, surface_l)
        if reached is None and t == 0:
            return None
        distance = (reached if reached is not None else extreme) - surface_l
        floor_target = _level_target(_READABLE_FLOOR[appearance], appearance, surface_l)
        floor = (floor_target if floor_target is not None else extreme) - surface_l
        compressed = distance - (distance - floor) * min(t, 1) if abs(distance) > abs(floor) else distance
        return surface_l + compressed * (1 - max(0, t - 1))

    # Keep a panel light enough (or dark enough) that white (or black) text
    # still reaches the body text minimum on it. This only matters for
    # backgrounds near mid-gray, where it barely does on the background.
    extreme_text = {"r": 255, "g": 255, "b": 255} if lighter else {"r": 0, "g": 0, "b": 0}

    def readable(color: dict) -> bool:
        return wcag_contrast(extreme_text, color) >= _TEXT_MINIMUM_WCAG_CONTRAST

    def limit_panel(token: str, lightness: float) -> dict:
        color = paint(token, lightness)
        if readable(color):
            return color
        low, high = background_l, lightness
        for _ in range(20):
            middle = (low + high) / 2
            if readable(paint(token, middle)):
                low = middle
            else:
                high = middle
        return paint(token, low)

    def solve(t: float) -> dict | None:
        colors = {"background": background}
        for token in _SOLVE_ORDER:
            targets: list[float] = []
            for rule_token, on, level in _RULES:
                if rule_token != token:
                    continue
                for surface in on:
                    surface_color = colors.get(surface, background)
                    value = target(level, _oklab_lightness(surface_color), t)
                    if value is None or value < 0 or value > 1:
                        return None
                    targets.append(value)
            lightness = _extreme_of(targets, lighter)
            colors[token] = limit_panel(token, lightness) if token in _PANELS else paint(token, lightness)
        return colors

    relaxation = 0.0
    colors = solve(0)
    if colors is None:
        # Mid-gray backgrounds cannot fit every level: relax as little as
        # possible. Full relaxation always fits.
        low, high = 0.0, 2.0
        colors = solve(high)
        for _ in range(20):
            middle = (low + high) / 2
            attempt = solve(middle)
            if attempt is not None:
                high, colors = middle, attempt
            else:
                low = middle
        relaxation = high
    solved = colors if colors is not None else {}

    def surfaces_of(token: str) -> list[dict]:
        return [
            solved.get(surface, background)
            for rule_token, on, _level in _RULES
            if rule_token == token
            for surface in on
        ]

    result: dict = {}
    for token in _TOKEN_FAMILIES:
        color = solved.get(token)
        result[token] = _hex_of(color) if color else ""

    for token in _FOREGROUND_TOKENS:
        surfaces = surfaces_of(token)
        # Body text uses the terminal's own foreground where it is clearly
        # stronger than muted text; otherwise the foreground's hue at just
        # enough lightness.
        text = solved.get(token)
        if foreground:
            targets = [target(_FOREGROUND_LEVEL, _oklab_lightness(surface), relaxation) for surface in surfaces]
            if all(value is not None and 0 <= value <= 1 for value in targets):
                needed = _extreme_of(targets, lighter)
                foreground_l = _oklab_lightness(foreground)
                if (foreground_l >= needed) if lighter else (foreground_l <= needed):
                    result[token] = ""
                    continue
                text = _anchored(
                    _source_of(foreground), _FAMILIES["neutral"], oklab_to_okhsl_lightness(needed), saturation
                )
        # Body text keeps at least 4.5:1 on the surfaces it is drawn on, even
        # on relaxed mid-gray backgrounds.
        if text:
            result[token] = _hex_of(_with_text_contrast(text, surfaces, lighter))
    return {"colors": result, "dim": [], "appearance": appearance}


def _rgb_of(color) -> dict:
    return {"r": color.r, "g": color.g, "b": color.b}


def _okhsl_of(color: dict) -> dict:
    return color_to_okhsl(rgb_color(color["r"], color["g"], color["b"]))


def _source_of(color: dict) -> dict:
    """A terminal color's OKHSL channels and its OKLCH chroma (``{"h", "s", "l", "chroma"}``)."""
    return {**_okhsl_of(color), "chroma": color_to_oklch(rgb_color(color["r"], color["g"], color["b"]))["c"]}


def _anchored(source: dict, family: dict, lightness: float, saturation: float) -> dict:
    """A source color's hue at another OKHSL lightness. Its saturation applies
    at its own lightness and falls off toward black and white along the
    family's saturation curve, never rising above it.

    OKHSL saturation is relative to the most chroma sRGB allows at a
    lightness, so the same saturation can mean more chroma elsewhere:
    Catppuccin Frappe's pink #f4b8e4 (chroma 0.089) would become #eb76d1
    (0.180) at the lightness the accent needs. Chroma is therefore also capped
    at the source's, with the same falloff."""
    anchor = _saturation_curve(family, source["l"])
    falloff = min(1, _saturation_curve(family, lightness) / anchor) if anchor > 0 else 1
    color = okhsl_color(source["h"], source["s"] * falloff * saturation, lightness)
    cap = source["chroma"] * falloff * saturation
    oklch = color_to_oklch(color)
    return _rgb_of(color) if oklch["c"] <= cap else color_to_rgb(oklch_color(oklch["l"], cap, source["h"]))


def _with_text_contrast(color: dict, surfaces: list[dict], lighter: bool) -> dict:
    """Move a text color toward white or black until it reaches the WCAG
    minimum on every surface."""

    def meets(candidate: dict) -> bool:
        return all(wcag_contrast(candidate, surface) >= _TEXT_MINIMUM_WCAG_CONTRAST for surface in surfaces)

    if meets(color):
        return color
    okhsl = _okhsl_of(color)

    def at(lightness: float) -> dict:
        return _rgb_of(okhsl_color(okhsl["h"], okhsl["s"], lightness))

    extreme = 1 if lighter else 0
    if not meets(at(extreme)):
        return at(extreme)
    low, high = okhsl["l"], float(extreme)
    for _ in range(20):
        middle = (low + high) / 2
        if meets(at(middle)):
            high = middle
        else:
            low = middle
    return at(high)


def _indexed_colors(saturation: float, appearance: str | None) -> dict:
    """Colors for terminals that reported nothing: the terminal renders ANSI
    indices 0-15 and the default colors with its own theme, so they fit any
    background. Neutral tokens below body text are faint (SGR 2) instead of
    bright black, which some themes make nearly invisible. Panels have no
    background."""
    colors: dict = {}
    dim: list[str] = []
    for token, family_name in _TOKEN_FAMILIES.items():
        if token in _PANELS:
            colors[token] = ""
            continue
        neutral = family_name == "neutral"
        colors[token] = (
            _TOKEN_SLOTS.get(token, _FAMILIES[family_name]["slot"]) if not neutral and saturation > 0 else ""
        )
        if neutral and token not in _FOREGROUND_TOKENS:
            dim.append(token)
    return {"colors": colors, "dim": dim, "appearance": appearance}
