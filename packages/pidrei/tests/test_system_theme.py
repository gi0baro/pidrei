"""Mirror of pi coding-agent test/system-theme.test.ts."""

import pytest

from pidrei.modes.interactive.theme import (
    get_available_themes,
    get_theme_by_name,
    get_theme_export_colors,
    set_terminal_colors,
)
from pidrei.modes.interactive.theme.system_theme import generate_system_theme_colors, wcag_contrast
from pidrei_tui import color_to_oklch, color_to_rgb, parse_color, rgb_color


def rgb(hex_value: str) -> dict:
    return color_to_rgb(parse_color(hex_value))


def lightness(color: dict) -> float:
    return color_to_oklch(rgb_color(color["r"], color["g"], color["b"]))["l"]


DRACULA = {
    "background": rgb("#282a36"),
    "foreground": rgb("#f8f8f2"),
    "palette": [
        rgb(value)
        for value in [
            *["#21222c", "#ff5555", "#50fa7b", "#f1fa8c", "#bd93f9", "#ff79c6", "#8be9fd", "#f8f8f2"],
            *["#6272a4", "#ff6e6e", "#69ff94", "#ffffa5", "#d6acff", "#ff92df", "#a4ffff", "#ffffff"],
        ]
    ],
}

# Dark with a palette, light with an unreadable foreground, background only, and mid-gray.
TERMINALS = {
    "dracula": DRACULA,
    "solarizedLight": {"background": rgb("#fdf6e3"), "foreground": rgb("#657b83")},
    "backgroundOnly": {"background": rgb("#1e1e1e")},
    "midGray": {"background": rgb("#808080"), "foreground": rgb("#ffffff")},
}

PANELS = ["userMessageBg", "toolPendingBg", "toolSuccessBg", "toolErrorBg", "selectedBg"]


def resolved(input_: dict, token: str) -> dict:
    value = generate_system_theme_colors(input_)["colors"][token]
    if value == "":
        return input_["background"] if token in PANELS else input_["foreground"]
    return rgb(value)


@pytest.fixture(autouse=True)
def _reset_terminal_colors():
    yield
    set_terminal_colors({})


class TestGenerateSystemThemeColors:
    def test_keeps_body_text_readable_wcag_4_5_on_the_background_and_its_panels(self):
        for name, input_ in TERMINALS.items():
            text = resolved(input_, "text")
            for surface in [input_["background"], resolved(input_, "selectedBg")]:
                assert wcag_contrast(text, surface) >= 4.5, name
            assert wcag_contrast(resolved(input_, "toolTitle"), resolved(input_, "toolErrorBg")) >= 4.5, name

    def test_orders_foreground_roles_by_contrast_and_keeps_panels_close_to_the_background(self):
        for name, input_ in TERMINALS.items():
            background = lightness(input_["background"])

            def offset(token: str, input_=input_, background=background) -> float:
                return lightness(resolved(input_, token)) - background

            # On mid-gray the levels collapse to the strongest reachable color.
            if name != "midGray":
                assert abs(offset("text")) > abs(offset("muted")), name
                assert abs(offset("muted")) > abs(offset("dim")), name
            lighter = generate_system_theme_colors(input_)["appearance"] == "dark"
            for panel in PANELS:
                assert wcag_contrast(resolved(input_, panel), input_["background"]) < 2, f"{name} {panel}"
                assert (offset(panel) > 0) is lighter, f"{name} {panel}"

    def test_uses_the_terminal_foreground_and_palette_hues(self):
        assert generate_system_theme_colors(DRACULA)["colors"]["text"] == ""
        # Solarized's foreground is below 4.5:1 on its own background, so text is darkened.
        assert generate_system_theme_colors(TERMINALS["solarizedLight"])["colors"]["text"] != ""

        def hue(color: dict) -> float:
            return color_to_oklch(rgb_color(color["r"], color["g"], color["b"]))["h"]

        assert abs(hue(resolved(DRACULA, "error")) - hue(DRACULA["palette"][1])) < 8

    # https://github.com/earendil-works/pi/issues/10255
    def test_keeps_pastel_palette_colors_pastel_at_other_lightnesses(self):
        frappe = {
            "background": rgb("#303446"),
            "foreground": rgb("#c6d0f5"),
            "palette": [
                rgb(value)
                for value in [
                    *["#51576d", "#e78284", "#a6d189", "#e5c890", "#8caaee", "#f4b8e4", "#81c8be", "#b5bfe2"],
                    *["#626880", "#e67172", "#8ec772", "#d9ba73", "#7b9ef0", "#f2a4db", "#5abfb5", "#a5adce"],
                ]
            ],
        }

        def chroma(color: dict) -> float:
            return color_to_oklch(rgb_color(color["r"], color["g"], color["b"]))["c"]

        pink = frappe["palette"][5]
        accent = resolved(frappe, "accent")
        # The accent is darker than the pink, but must not gain chroma (it was 2x before the cap).
        assert lightness(accent) < lightness(pink) - 0.05
        assert chroma(accent) <= chroma(pink) * 1.03
        for panel in ["userMessageBg", "customMessageBg"]:
            assert chroma(resolved(frappe, panel)) <= 0.1, panel

    def test_renders_grayscale_at_zero_saturation(self):
        colors = generate_system_theme_colors({**DRACULA, "saturation": 0})["colors"]
        assert color_to_oklch(parse_color(colors["error"]))["c"] < 0.005

    def test_falls_back_to_palette_indices_and_faint_text_without_a_background(self):
        generated = generate_system_theme_colors({"appearanceHint": "light"})
        colors = generated["colors"]
        assert generated["appearance"] == "light"
        assert [colors["error"], colors["text"], colors["userMessageBg"]] == [1, "", ""]
        assert "muted" in generated["dim"]
        assert generate_system_theme_colors({"saturation": 0})["colors"]["error"] == ""


class TestSystemTheme:
    @pytest.mark.tonio
    async def test_is_listed_first_has_no_export_colors_and_is_generated_from_the_terminal_colors(self):
        assert (await get_available_themes())[0] == "system"
        assert await get_theme_export_colors("system") == {}

        set_terminal_colors(DRACULA)
        system = await get_theme_by_name("system")
        assert system.appearance == "dark"
        assert system.get_fg_ansi("text") == "\x1b[39m"
        assert system.get_fg_ansi("error").startswith("\x1b[38;")

    @pytest.mark.tonio
    async def test_renders_faint_tokens_with_sgr_2_and_closes_it(self):
        system = await get_theme_by_name("system")
        assert system.fg("muted", "x") == "\x1b[39m\x1b[2mx\x1b[22;39m"
        assert system.style("x", {"fg": "muted"}) == "\x1b[39m\x1b[2mx\x1b[22m\x1b[39m"
