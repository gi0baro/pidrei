"""Mirror of pi coding-agent test/settings-selector.test.ts.

pi casts a four-key object to `SettingsConfig`; JS reads the missing keys as
undefined, so here the config is spelled out with neutral values (a Python
dict would raise KeyError instead).
"""

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.settings_selector import SettingsSelectorComponent
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei_tui import set_keybindings


BASE_CONFIG = {
    "autoCompact": True,
    "autocompleteMaxVisible": 5,
    "autoResizeImages": True,
    "availableThemes": [],
    "availableThinkingLevels": [],
    "blockImages": False,
    "clearOnShrink": False,
    "collapseChangelog": False,
    "currentTheme": "dark",
    "defaultProjectTrust": "ask",
    "doubleEscapeAction": "none",
    "editorPaddingX": 1,
    "enableProviderAttribution": True,
    "enableSkillCommands": False,
    "followUpMode": "queue",
    "fullscreenScrollbar": "auto",
    "hideThinkingBlock": False,
    "httpIdleTimeoutMs": 0,
    "imageWidthCells": 40,
    "outputPad": 1,
    "quietStartup": False,
    "showCacheMissNotices": True,
    "showHardwareCursor": False,
    "showImages": True,
    "showTerminalProgress": False,
    "steeringMode": "interrupt",
    "terminalTheme": "dark",
    "thinkingLevel": "off",
    "transport": "auto",
    "treeFilterMode": "default",
    "tuiMode": "regular",
    "warnings": {},
}


@pytest.fixture(autouse=True)
def _theme():
    init_theme_sync("dark")
    set_keybindings(KeybindingsManager(None, None))


@pytest.mark.tonio
async def test_cycles_through_fullscreen_scrollbar_modes():
    changes: list[str] = []

    def on_change(mode: str) -> None:
        changes.append(mode)

    async def on_cancel() -> None:
        pass

    selector = SettingsSelectorComponent(
        dict(BASE_CONFIG),
        {
            "onFullscreenScrollbarChange": on_change,
            "onWarningsChange": lambda warnings: None,
            "onCancel": on_cancel,
        },
    )
    settings_list = selector.get_settings_list()

    for character in "Fullscreen scrollbar":
        await settings_list.handle_input(character)
    await settings_list.handle_input("\r")
    await settings_list.handle_input("\r")
    await settings_list.handle_input("\r")

    assert changes == ["always", "hidden", "auto"]
