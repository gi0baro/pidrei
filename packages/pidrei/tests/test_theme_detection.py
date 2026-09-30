"""Mirror of pi coding-agent test/theme-detection.test.ts."""

import re

import pytest

from pidrei.modes.interactive.theme import (
    detect_color_fg_bg_theme,
    detect_terminal_theme,
    get_theme_by_name,
    parse_auto_theme_setting,
    resolve_theme_setting,
)
from pidrei_tui import reset_capabilities_cache, set_capabilities


@pytest.fixture(autouse=True)
def _reset_capabilities(request):
    request.addfinalizer(reset_capabilities_cache)


class TestDetectColorFgBgTheme:
    def test_classifies_the_last_field_by_palette_index_like_vim(self):
        assert detect_color_fg_bg_theme({"COLORFGBG": "15;0"}) == "dark"
        assert detect_color_fg_bg_theme({"COLORFGBG": "0;7;15"}) == "light"
        # Solarized Dark's background is bright black.
        assert detect_color_fg_bg_theme({"COLORFGBG": "12;8"}) == "dark"
        # rxvt writes "default" when the background is not a palette color.
        assert detect_color_fg_bg_theme({"COLORFGBG": "15;default"}) is None
        assert detect_color_fg_bg_theme({}) is None


class TestDetectTerminalTheme:
    def test_prefers_the_background_then_the_reported_scheme_then_colorfgbg_then_dark(self):
        env = {"COLORFGBG": "0;15"}
        assert detect_terminal_theme({"background": {"r": 8, "g": 8, "b": 8}}, "light", env) == "dark"
        assert detect_terminal_theme({}, "dark", env) == "dark"
        assert detect_terminal_theme({}, None, env) == "light"
        assert detect_terminal_theme({}, None, {}) == "dark"

    def test_follows_the_foreground_when_text_is_readable_that_way(self):
        background = {"r": 118, "g": 118, "b": 118}
        assert detect_terminal_theme({"background": background}) == "light"
        assert detect_terminal_theme({"background": background, "foreground": {"r": 255, "g": 255, "b": 255}}) == "dark"
        # White text cannot reach 4.5:1 on mid-gray.
        mid_gray = {"r": 128, "g": 128, "b": 128}
        assert detect_terminal_theme({"background": mid_gray, "foreground": {"r": 255, "g": 255, "b": 255}}) == "light"


class TestThemeColorMode:
    @pytest.mark.tonio
    async def test_uses_terminal_capabilities(self):
        set_capabilities({"images": None, "trueColor": False, "hyperlinks": False})
        ansi256_theme = await get_theme_by_name("dark")
        assert ansi256_theme is not None
        assert ansi256_theme.get_color_mode() == "256color"
        assert re.fullmatch(r"\x1b\[38;5;\d+m", ansi256_theme.get_fg_ansi("accent"))

        set_capabilities({"images": None, "trueColor": True, "hyperlinks": False})
        truecolor_theme = await get_theme_by_name("dark")
        assert truecolor_theme is not None
        assert truecolor_theme.get_color_mode() == "truecolor"
        assert re.fullmatch(r"\x1b\[38;2;\d+;\d+;\d+m", truecolor_theme.get_fg_ansi("accent"))


class TestThemeSettingHelpers:
    def test_parses_and_resolves_automatic_theme_settings(self):
        assert parse_auto_theme_setting("light/dark") == {"lightTheme": "light", "darkTheme": "dark"}
        assert resolve_theme_setting("dark", "light") == "dark"
        assert resolve_theme_setting("light/dark", "light") == "light"
        assert resolve_theme_setting("light/dark", "dark") == "dark"
        assert resolve_theme_setting("light/dark/extra", "dark") is None
