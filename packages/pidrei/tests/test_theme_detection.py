"""Mirror of pi coding-agent test/theme-detection.test.ts."""

import re

import pytest

from pidrei.modes.interactive.theme import (
    detect_terminal_background_from_env,
    detect_terminal_background_theme,
    get_theme_by_name,
    get_theme_for_rgb_color,
    parse_auto_theme_setting,
    resolve_theme_setting,
)
from pidrei_tui import reset_capabilities_cache, set_capabilities


@pytest.fixture(autouse=True)
def _reset_capabilities(request):
    request.addfinalizer(reset_capabilities_cache)


class TestDetectTerminalBackgroundFromEnv:
    def test_uses_the_colorfgbg_background_color_index(self):
        detection = detect_terminal_background_from_env({"env": {"COLORFGBG": "0;15"}})
        assert detection["theme"] == "light"
        assert detection["source"] == "COLORFGBG"
        assert detection["confidence"] == "high"

        detection = detect_terminal_background_from_env({"env": {"COLORFGBG": "15;0"}})
        assert detection["theme"] == "dark"
        assert detection["source"] == "COLORFGBG"
        assert detection["confidence"] == "high"

    def test_uses_the_last_colorfgbg_field_as_the_background(self):
        assert detect_terminal_background_from_env({"env": {"COLORFGBG": "0;7;15"}})["theme"] == "light"

    def test_defaults_to_dark_without_terminal_background_hints(self):
        detection = detect_terminal_background_from_env({"env": {}})
        assert detection["theme"] == "dark"
        assert detection["source"] == "fallback"
        assert detection["confidence"] == "low"


class TestDetectTerminalBackgroundTheme:
    @pytest.mark.tonio
    async def test_uses_the_queried_terminal_background_before_environment_hints(self):
        queried_timeout_ms = None

        class Ui:
            async def query_terminal_background_color(self, *, timeout_ms):
                nonlocal queried_timeout_ms
                queried_timeout_ms = timeout_ms
                return {"r": 250, "g": 250, "b": 250}

        detection = await detect_terminal_background_theme({"env": {"COLORFGBG": "15;0"}, "timeoutMs": 250, "ui": Ui()})

        assert queried_timeout_ms == 250
        assert detection["theme"] == "light"
        assert detection["source"] == "terminal background"
        assert detection["confidence"] == "high"

    @pytest.mark.tonio
    async def test_falls_back_to_environment_hints_when_the_terminal_query_returns_no_color(self):
        class Ui:
            async def query_terminal_background_color(self, *, timeout_ms):
                return None

        detection = await detect_terminal_background_theme({"env": {"COLORFGBG": "15;0"}, "timeoutMs": 250, "ui": Ui()})

        assert detection["theme"] == "dark"
        assert detection["source"] == "COLORFGBG"
        assert detection["confidence"] == "high"

    @pytest.mark.tonio
    async def test_falls_back_to_environment_hints_when_the_terminal_query_fails(self):
        class Ui:
            async def query_terminal_background_color(self, *, timeout_ms):
                raise RuntimeError("terminal write failed")

        detection = await detect_terminal_background_theme({"env": {"COLORFGBG": "0;15"}, "timeoutMs": 250, "ui": Ui()})

        assert detection["theme"] == "light"
        assert detection["source"] == "COLORFGBG"
        assert detection["confidence"] == "high"


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


class TestThemeDetectionFromRgb:
    def test_classifies_rgb_colors_by_luminance(self):
        assert get_theme_for_rgb_color({"r": 8, "g": 8, "b": 8}) == "dark"
        assert get_theme_for_rgb_color({"r": 250, "g": 250, "b": 250}) == "light"


class TestThemeSettingHelpers:
    def test_parses_and_resolves_automatic_theme_settings(self):
        assert parse_auto_theme_setting("light/dark") == {"lightTheme": "light", "darkTheme": "dark"}
        assert resolve_theme_setting("dark", "light") == "dark"
        assert resolve_theme_setting("light/dark", "light") == "light"
        assert resolve_theme_setting("light/dark", "dark") == "dark"
        assert resolve_theme_setting("light/dark/extra", "dark") is None
