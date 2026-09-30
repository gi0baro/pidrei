"""Mirror of pi tui src/oklab.ts.

Oklab and OKHSL <-> sRGB conversion. `colors.py` builds its OKLCH, OKHSL, and
color mixing on it.

Oklab and OKHSL are Björn Ottosson's color spaces; OKHSL's saturation is
relative to the sRGB gamut at each hue and lightness. This is a port of his
reference implementation (https://bottosson.github.io/posts/colorpicker/),
Copyright (c) 2021 Björn Ottosson, used under the MIT license:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions: The above copyright
notice and this permission notice shall be included in all copies or
substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS",
WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

Vectors are 3-tuples; RgbColor is the ``{"r", "g", "b"}`` record of
`terminal_colors`.
"""

import math
import sys


type Vector = tuple[float, float, float]
type Matrix = tuple[Vector, Vector, Vector]


def _multiply(m: Matrix, v: Vector) -> Vector:
    x, y, z = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z,
        m[1][0] * x + m[1][1] * y + m[1][2] * z,
        m[2][0] * x + m[2][1] * y + m[2][2] * z,
    )


# ============================================================================
# OKHSL <-> sRGB
# ============================================================================

_LINEAR_SRGB_TO_LMS: Matrix = (
    (0.4122214694707629, 0.5363325372617349, 0.0514459932675022),
    (0.2119034958178251, 0.6806995506452344, 0.1073969535369405),
    (0.0883024591900564, 0.2817188391361215, 0.6299787016738222),
)
_LMS_TO_LAB: Matrix = (
    (0.210454268309314, 0.793617774702305, -0.0040720430116193),
    (1.9779985324311684, -2.42859224204858, 0.450593709617411),
    (0.0259040424655478, 0.7827717124575296, -0.8086757549230774),
)
_LAB_TO_LMS: Matrix = (
    (1, 0.3963377773761749, 0.2158037573099136),
    (1, -0.1055613458156586, -0.0638541728258133),
    (1, -0.0894841775298119, -1.2914855480194092),
)
_LMS_TO_LINEAR_SRGB: Matrix = (
    (4.0767416360759583, -3.3077115392580629, 0.2309699031821043),
    (-1.2684379732850315, 2.6097573492876882, -0.341319376002657),
    (-0.0041960761386756, -0.7034186179359362, 1.7076146940746117),
)
# Per sRGB channel (red, green, blue): the (a, b) half-plane where that
# channel clips first, and the polynomial approximating the maximum
# saturation there.
_SATURATION_FIT: tuple = (
    ((-1.8817031, -0.80936501), (1.19086277, 1.76576728, 0.59662641, 0.75515197, 0.56771245)),
    ((1.8144408, -1.19445267), (0.73956515, -0.45954404, 0.08285427, 0.12541073, -0.14503204)),
    ((0.13110758, 1.81333971), (1.35733652, -0.00915799, -1.1513021, -0.50559606, 0.00692167)),
)
_K1 = 0.206
_K2 = 0.03
_K3 = (1 + _K1) / (1 + _K2)


def oklab_to_okhsl_lightness(x: float) -> float:
    """Oklab lightness to OKHSL lightness."""
    return 0.5 * (_K3 * x - _K1 + math.sqrt((_K3 * x - _K1) ** 2 + 4 * _K2 * _K3 * x))


def _okhsl_to_oklab_lightness(x: float) -> float:
    """OKHSL lightness to Oklab lightness."""
    return (x * x + _K1 * x) / (_K3 * (x + _K2))


def _linear_to_srgb(value: float) -> float:
    """sRGB transfer function: linear to encoded channel, both 0-1."""
    return 1.055 * value ** (1 / 2.4) - 0.055 if value > 0.0031308 else 12.92 * value


def _srgb_to_linear(value: float) -> float:
    """Inverse sRGB transfer function: encoded to linear channel, both 0-1."""
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _cube(value: float) -> float:
    return value**3


def oklab_to_linear_srgb(lab: Vector) -> Vector:
    """Oklab [L, a, b] to linear sRGB [r, g, b] (0-1, may leave the gamut)."""
    lms = _multiply(_LAB_TO_LMS, lab)
    return _multiply(_LMS_TO_LINEAR_SRGB, (_cube(lms[0]), _cube(lms[1]), _cube(lms[2])))


def _linear_srgb_to_oklab(rgb: Vector) -> Vector:
    """Linear sRGB [r, g, b] (0-1) to Oklab [L, a, b]."""
    lms = _multiply(_LINEAR_SRGB_TO_LMS, rgb)
    return _multiply(_LMS_TO_LAB, (_cbrt(lms[0]), _cbrt(lms[1]), _cbrt(lms[2])))


def _cbrt(value: float) -> float:
    # JS Math.cbrt: real cube root, negative for negative input.
    return math.copysign(abs(value) ** (1 / 3), value)


def rgb_to_oklab(rgb: dict) -> Vector:
    """sRGB channels (0-255) to Oklab [L, a, b]."""
    return _linear_srgb_to_oklab(
        (
            _srgb_to_linear(rgb["r"] / 255),
            _srgb_to_linear(rgb["g"] / 255),
            _srgb_to_linear(rgb["b"] / 255),
        )
    )


def _js_round(value: float) -> int:
    # JS Math.round: halves round toward +infinity.
    return math.floor(value + 0.5)


def linear_srgb_to_rgb(linear: Vector) -> dict:
    """Linear sRGB [r, g, b] to sRGB channels (0-255, rounded), clipping
    out-of-gamut channels."""
    r, g, b = (_js_round(min(1, max(0, _linear_to_srgb(value))) * 255) for value in linear)
    return {"r": r, "g": g, "b": b}


def _lms_slopes(a: float, b: float) -> Vector:
    """Rate of change of each cube-root LMS component along a chroma direction (a, b)."""
    return (
        _LAB_TO_LMS[0][1] * a + _LAB_TO_LMS[0][2] * b,
        _LAB_TO_LMS[1][1] * a + _LAB_TO_LMS[1][2] * b,
        _LAB_TO_LMS[2][1] * a + _LAB_TO_LMS[2][2] * b,
    )


def _max_saturation(a: float, b: float) -> float:
    """Largest saturation (C/L) inside sRGB for hue (a, b): polynomial fit plus one Halley step."""
    channel = next(index for index, ((x, y), _) in enumerate(_SATURATION_FIT) if index == 2 or x * a + y * b > 1)
    k0, k1, k2, k3, k4 = _SATURATION_FIT[channel][1]
    weights = _LMS_TO_LINEAR_SRGB[channel]
    saturation = k0 + k1 * a + k2 * b + k3 * a * a + k4 * a * b

    slopes = _lms_slopes(a, b)
    base = [1 + saturation * k for k in slopes]

    def dot(values: list) -> float:
        return sum(weights[index] * value for index, value in enumerate(values))

    f = dot([value**3 for value in base])
    f1 = dot([3 * slopes[index] * value**2 for index, value in enumerate(base)])
    f2 = dot([6 * slopes[index] ** 2 * value for index, value in enumerate(base)])
    return saturation - (f * f1) / (f1 * f1 - 0.5 * f * f2)


def _cusp(a: float, b: float) -> tuple[float, float]:
    """Oklab lightness and chroma of the most saturated sRGB color of hue (a, b)."""
    saturation = _max_saturation(a, b)
    lightness = _cbrt(1 / max(oklab_to_linear_srgb((1, saturation * a, saturation * b))))
    return lightness, lightness * saturation


def _max_chroma(a: float, b: float, lightness: float, cusp: tuple[float, float]) -> float:
    """Chroma where the constant-lightness line at `lightness` leaves the sRGB gamut."""
    cusp_l, cusp_c = cusp
    if lightness <= cusp_l:
        return (cusp_c * lightness) / cusp_l
    # Upper half: triangle edge, then one Halley step against each channel reaching 1.
    t = (cusp_c * (lightness - 1)) / (cusp_l - 1)
    slopes = _lms_slopes(a, b)
    lms = [lightness + t * k for k in slopes]
    cubes = [value**3 for value in lms]
    first = [3 * slopes[index] * value**2 for index, value in enumerate(lms)]
    second = [6 * slopes[index] ** 2 * value for index, value in enumerate(lms)]

    def dot(row: Vector, values: list) -> float:
        return row[0] * values[0] + row[1] * values[1] + row[2] * values[2]

    steps = []
    for row in _LMS_TO_LINEAR_SRGB:
        f = dot(row, cubes) - 1
        f1 = dot(row, first)
        f2 = dot(row, second)
        u = f1 / (f1 * f1 - 0.5 * f * f2)
        steps.append(-f * u if u >= 0 else sys.float_info.max)
    return t + min(steps)


def _chroma_stops(big_l: float, a: float, b: float) -> tuple[float, float, float]:
    """OKHSL's chroma reference points at lightness L and hue (a, b): [c0, cMid, cMax]."""
    peak = _cusp(a, b)
    c_max = _max_chroma(a, b, big_l, peak)
    k = c_max / min(big_l * (peak[1] / peak[0]), (1 - big_l) * (peak[1] / (1 - peak[0])))
    mid_s = 0.11516993 + 1 / (
        7.4477897
        + 4.1590124 * b
        + a
        * (
            -2.19557347
            + 1.75198401 * b
            + a * (-2.13704948 - 10.02301043 * b + a * (-4.24894561 + 5.38770819 * b + 4.69891013 * a))
        )
    )
    mid_t = 0.11239642 + 1 / (
        1.6132032
        - 0.68124379 * b
        + a
        * (
            0.40370612
            + 0.90148123 * b
            + a * (-0.27087943 + 0.6122399 * b + a * (0.00299215 - 0.45399568 * b - 0.14661872 * a))
        )
    )
    c_mid = 0.9 * k * math.sqrt(math.sqrt(1 / (1 / (big_l * mid_s) ** 4 + 1 / ((1 - big_l) * mid_t) ** 4)))
    c0 = math.sqrt(1 / (1 / (big_l * 0.4) ** 2 + 1 / ((1 - big_l) * 0.8) ** 2))
    return c0, c_mid, c_max


def okhsl_to_rgb(hue: float, saturation: float, lightness: float) -> dict:
    """Convert OKHSL to sRGB channels (0-255, rounded), clipping out-of-gamut channels.

    ``hue`` in degrees; ``saturation`` and ``lightness`` 0-1.
    """
    big_l = _okhsl_to_oklab_lightness(lightness)
    lab: Vector = (big_l, 0, 0)
    if 0 < big_l < 1 and saturation > 0:
        angle = (2 * math.pi * (((hue % 360) + 360) % 360)) / 360
        a = math.cos(angle)
        b = math.sin(angle)
        c0, c_mid, c_max = _chroma_stops(big_l, a, b)
        # Chroma rises from 0 through cMid at s = 0.8 to cMax at s = 1.
        if saturation < 0.8:
            t = 1.25 * saturation
            k1 = 0.8 * c0
            chroma = (t * k1) / (1 - (1 - k1 / c_mid) * t)
        else:
            t = 5 * (saturation - 0.8)
            k1 = (0.2 * c_mid**2 * 1.25**2) / c0
            chroma = c_mid + (t * k1) / (1 - (1 - k1 / (c_max - c_mid)) * t)
        lab = (big_l, chroma * a, chroma * b)
    return linear_srgb_to_rgb(oklab_to_linear_srgb(lab))


def rgb_to_okhsl(rgb: dict) -> dict:
    """Convert sRGB channels (0-255) to OKHSL.

    Returns ``{"h", "s", "l"}``: hue in degrees (0 for grays), saturation and
    lightness 0-1.
    """
    big_l, lab_a, lab_b = rgb_to_oklab(rgb)
    chroma = math.hypot(lab_a, lab_b)
    lightness = oklab_to_okhsl_lightness(big_l)
    if chroma < 1e-9 or lightness <= 0 or lightness >= 1:
        return {"h": 0, "s": 0, "l": lightness}

    hue = ((math.atan2(lab_b, lab_a) * 180) / math.pi + 360) % 360
    c0, c_mid, c_max = _chroma_stops(big_l, lab_a / chroma, lab_b / chroma)
    if chroma < c_mid:
        k1 = 0.8 * c0
        saturation = 0.8 * (chroma / (k1 + (1 - k1 / c_mid) * chroma))
    else:
        k1 = (0.2 * c_mid**2 * 1.25**2) / c0
        offset = chroma - c_mid
        saturation = 0.8 + 0.2 * (offset / (k1 + (1 - k1 / (c_max - c_mid)) * offset))
    return {"h": hue, "s": min(1, max(0, saturation)), "l": lightness}
