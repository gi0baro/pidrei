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
    "availableDefaultModels": [],
    "currentTheme": "dark",
    "defaultModel": "not set",
    "defaultProjectTrust": "ask",
    "doubleEscapeAction": "none",
    "editorPaddingX": 1,
    "enableProviderAttribution": True,
    "enableSkillCommands": False,
    "followUpMode": "queue",
    "fullscreenExitOutput": "transcript",
    "fullscreenScrollbar": "auto",
    "fullscreenCopyOnSelect": True,
    "hideThinkingBlock": False,
    "httpIdleTimeoutMs": 0,
    "imageWidthCells": 40,
    "modelThinkingLevels": {},
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
async def test_cycles_through_fullscreen_settings():
    exit_output_changes: list[str] = []
    scrollbar_changes: list[str] = []
    copy_on_select_changes: list[bool] = []

    async def on_cancel() -> None:
        pass

    callbacks = {
        "onFullscreenExitOutputChange": exit_output_changes.append,
        "onFullscreenScrollbarChange": scrollbar_changes.append,
        "onFullscreenCopyOnSelectChange": copy_on_select_changes.append,
        "onWarningsChange": lambda warnings: None,
        "onCancel": on_cancel,
    }

    async def cycle(label: str, count: int) -> None:
        settings_list = SettingsSelectorComponent(dict(BASE_CONFIG), callbacks).get_settings_list()
        for character in label:
            await settings_list.handle_input(character)
        for _ in range(count):
            await settings_list.handle_input("\r")

    await cycle("Fullscreen exit output", 2)
    assert exit_output_changes == ["resume-hint", "transcript"]
    await cycle("Fullscreen scrollbar", 3)
    assert scrollbar_changes == ["always", "hidden", "auto"]
    await cycle("Fullscreen copy on select", 2)
    assert copy_on_select_changes == [False, True]
