"""Mirror of pi tui test/colors.test.ts."""

import re

import pytest

from pidrei_tui import (
    OklchColorValue,
    RgbColorValue,
    color_to_hex,
    color_to_okhsl,
    color_to_rgb,
    indexed_color,
    okhsl_color,
    oklch_color,
    parse_color,
    rgb_color,
    style_text,
)


def test_parses_hex_and_oklch_colors_and_rejects_everything_else():
    assert parse_color("#abc") == RgbColorValue(170, 187, 204)
    assert parse_color("oklch(62% 0.1 200)") == OklchColorValue(0.62, 0.1, 200)
    with pytest.raises(ValueError, match="Invalid color value"):
        parse_color("")
    with pytest.raises(ValueError, match="Invalid color value"):
        parse_color("red")


def test_gamut_maps_oklch_to_srgb_including_the_lightness_limits():
    assert color_to_rgb(oklch_color(0.627955, 0.257683, 29.2339)) == {"r": 255, "g": 0, "b": 0}
    assert color_to_rgb(oklch_color(1, 0.3, 150)) == {"r": 255, "g": 255, "b": 255}
    assert color_to_rgb(oklch_color(0, 0.3, 150)) == {"r": 0, "g": 0, "b": 0}


def test_parses_okhsl_colors_and_round_trips_them():
    # Full saturation at the red cusp is pure sRGB red.
    assert parse_color("okhsl(29.23 100% 56.8%)") == rgb_color(255, 0, 0)
    assert parse_color("OKHSL(250deg 60% 55%)") == okhsl_color(250, 0.6, 0.55)
    with pytest.raises(ValueError, match="s must be between 0 and 1"):
        parse_color("okhsl(250 160% 55%)")
    for hex_value in ["#4f8eb3", "#20242a", "#f8f9fa"]:
        okhsl = color_to_okhsl(parse_color(hex_value))
        assert color_to_hex(okhsl_color(okhsl["h"], okhsl["s"], okhsl["l"])) == hex_value


def test_styles_text_and_closes_sequences_in_reverse_order():
    assert (
        style_text(
            "Ready", {"fg": rgb_color(18, 52, 86), "bg": indexed_color(9), "bold": True, "italic": True}, "truecolor"
        )
        == "\x1b[38;2;18;52;86m\x1b[48;5;9m\x1b[1m\x1b[3mReady\x1b[23m\x1b[22m\x1b[49m\x1b[39m"
    )
    assert re.match(r"^\x1b\[38;5;\d+mReady\x1b\[39m$", style_text("Ready", {"fg": rgb_color(18, 52, 86)}, "256color"))
