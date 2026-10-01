"""Mirror of pi coding-agent test/theme-style.test.ts."""

import json
import os

import pytest

from pidrei.config import get_themes_dir
from pidrei.modes.interactive.theme import load_theme_from_path, set_terminal_colors
from pidrei_tui import OklchColorValue, color_to_hex, okhsl_color, style_text


@pytest.fixture(autouse=True)
def _reset_terminal_colors():
    yield
    set_terminal_colors({})


@pytest.fixture
def load_theme(tmp_path):
    """Load a copy of a built-in theme, modified by `edit`."""

    async def load(base: str, edit=None):
        with open(os.path.join(get_themes_dir(), f"{base}.json"), encoding="utf-8") as f:
            theme_json = json.load(f)
        if edit is not None:
            edit(theme_json)
        path = tmp_path / f"{theme_json['name']}-{len(list(tmp_path.iterdir()))}.json"
        path.write_text(json.dumps(theme_json), encoding="utf-8")
        return await load_theme_from_path(str(path), "truecolor")

    return load


@pytest.mark.tonio
async def test_renders_theme_tokens_the_same_as_the_generic_text_styler(load_theme):
    theme = await load_theme("dark")
    assert theme.style("Ready", {"fg": "success", "bg": "toolSuccessBg", "bold": True}) == style_text(
        "Ready", {"fg": theme.colors["success"], "bg": theme.colors["toolSuccessBg"], "bold": True}, "truecolor"
    )


@pytest.mark.tonio
async def test_rejects_unknown_tokens_and_tokens_in_the_wrong_slot(load_theme):
    theme = await load_theme("dark")
    with pytest.raises(ValueError, match="Unknown theme color: notAToken"):
        theme.style("x", {"fg": "notAToken"})
    # Background tokens are not foreground colors; use theme.colors["userMessageBg"].
    with pytest.raises(ValueError, match="Unknown theme color: userMessageBg"):
        theme.style("x", {"fg": "userMessageBg"})


@pytest.mark.tonio
async def test_loads_oklch_theme_values(load_theme):
    def edit(theme_json):
        theme_json["colors"]["accent"] = "oklch(62% 0.1 200)"

    theme = await load_theme("dark", edit)
    assert theme.colors["accent"] == OklchColorValue(0.62, 0.1, 200)


@pytest.mark.tonio
async def test_loads_okhsl_theme_values_including_through_variables(load_theme):
    def edit(theme_json):
        theme_json["vars"] = {**theme_json.get("vars", {}), "brand": "okhsl(250 60% 55%)"}
        theme_json["colors"]["accent"] = "brand"
        theme_json["colors"]["error"] = "okhsl(20 90% 60%)"

    theme = await load_theme("dark", edit)
    assert color_to_hex(theme.colors["accent"]) == color_to_hex(okhsl_color(250, 0.6, 0.55))
    assert color_to_hex(theme.colors["error"]) == color_to_hex(okhsl_color(20, 0.9, 0.6))


@pytest.mark.tonio
async def test_detects_the_appearance_unless_it_is_declared(load_theme):
    assert (await load_theme("dark")).appearance == "dark"
    assert (await load_theme("light")).appearance == "light"
    # Without a declaration, the appearance is detected from the theme's own colors.
    for base in ["dark", "light"]:
        assert (await load_theme(base, lambda theme_json: theme_json.pop("appearance"))).appearance == base

    def declare_light(theme_json):
        theme_json["appearance"] = "light"

    assert (await load_theme("dark", declare_light)).appearance == "light"

    # Palette colors 0-15 follow the terminal palette, so such themes follow the terminal background.
    def palette_only(theme_json):
        theme_json.pop("appearance")
        for key in theme_json["colors"]:
            theme_json["colors"][key] = 0 if key.endswith("Bg") else 7

    palette_theme = await load_theme("dark", palette_only)
    assert palette_theme.appearance == "dark"
    set_terminal_colors({"background": {"r": 250, "g": 250, "b": 250}})
    assert palette_theme.appearance == "light"


@pytest.mark.tonio
async def test_renders_empty_tokens_as_terminal_defaults_and_reports_concrete_colors_for_them(load_theme):
    def edit(theme_json):
        theme_json["colors"]["text"] = ""
        theme_json["colors"]["userMessageBg"] = ""

    theme = await load_theme("dark", edit)
    assert theme.fg("text", "x") == "\x1b[39mx\x1b[39m"
    assert theme.bg("userMessageBg", "x") == "\x1b[49mx\x1b[49m"
    assert color_to_hex(theme.colors["text"]) == "#e5e5e7"
    assert color_to_hex(theme.colors["userMessageBg"]) == "#000000"

    set_terminal_colors({"foreground": {"r": 200, "g": 210, "b": 220}, "background": {"r": 10, "g": 20, "b": 30}})
    assert color_to_hex(theme.colors["text"]) == "#c8d2dc"
    assert color_to_hex(theme.colors["userMessageBg"]) == "#0a141e"
