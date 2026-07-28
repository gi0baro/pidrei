"""Regression tests for the theme submenu's coroutine-returning callbacks.

Not a pi mirror: pi types these component callbacks sync (`=> void`) and
calls them bare, so nothing there can be dropped. The port made previews and
persistence async (never-block rule), and the invocation sites in
`SelectList`/`SettingsList` used to discard the returned coroutines — theme
preview on navigation, Esc, and Apply silently did nothing. These tests pin
the awaited flow end to end.
"""

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.settings_selector import ThemeSubmenu
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei_tui import set_keybindings


DOWN = "\x1b[B"
ESC = "\x1b"
ENTER = "\n"

THEMES = ["dark", "light"]


@pytest.fixture(autouse=True)
def _setup():
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager())


def _submenu(current_setting: str, previews: list, done_calls: list) -> ThemeSubmenu:
    async def on_theme_preview(theme_name: str) -> None:
        previews.append(theme_name)

    async def done(value: str | None = None) -> None:
        done_calls.append(value)

    return ThemeSubmenu(current_setting, "dark", THEMES, {"onThemePreview": on_theme_preview}, done)


class TestThemeSubmenu:
    @pytest.mark.tonio
    async def test_previews_theme_while_navigating_single_menu(self):
        previews: list = []
        submenu = _submenu("dark", previews, [])

        await submenu.handle_input(DOWN)

        assert len(previews) == 1

    @pytest.mark.tonio
    async def test_escape_restores_preview_and_reports_done(self):
        previews: list = []
        done_calls: list = []
        submenu = _submenu("dark", previews, done_calls)

        await submenu.handle_input(ESC)

        assert previews == ["dark"]  # original setting restored
        assert done_calls == [None]

    @pytest.mark.tonio
    async def test_selecting_a_theme_applies_it(self):
        done_calls: list = []
        submenu = _submenu("dark", [], done_calls)

        await submenu.handle_input(ENTER)

        assert done_calls == ["dark"]

    @pytest.mark.tonio
    async def test_automatic_menu_apply_reports_the_combined_setting(self):
        done_calls: list = []
        submenu = _submenu("light/dark", [], done_calls)

        # Items: light-theme, dark-theme, apply, single-mode.
        await submenu.handle_input(DOWN)
        await submenu.handle_input(DOWN)
        await submenu.handle_input(ENTER)

        assert done_calls == ["light/dark"]

    @pytest.mark.tonio
    async def test_automatic_menu_escape_restores_preview_and_reports_done(self):
        previews: list = []
        done_calls: list = []
        submenu = _submenu("light/dark", previews, done_calls)

        await submenu.handle_input(ESC)

        assert previews == ["light/dark"]
        assert done_calls == [None]
