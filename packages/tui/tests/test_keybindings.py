"""Mirror of pi tui test/keybindings.test.ts."""

from pidrei_tui.keybindings import TUI_KEYBINDINGS, KeybindingsManager


def test_binds_ctrl_j_as_default_newline_alias():
    keybindings = KeybindingsManager(TUI_KEYBINDINGS)

    assert keybindings.get_keys("tui.input.newLine") == ["shift+enter", "ctrl+j"]
    assert keybindings.matches("\n", "tui.input.newLine") is True
    assert keybindings.matches("\x1b[106;5u", "tui.input.newLine") is True


def test_binds_modified_and_unmodified_editor_viewport_navigation():
    keybindings = KeybindingsManager(TUI_KEYBINDINGS)

    assert keybindings.get_keys("tui.editor.cursorLineStart") == ["home", "ctrl+home", "ctrl+a"]
    assert keybindings.get_keys("tui.editor.cursorLineEnd") == ["end", "ctrl+end", "ctrl+e"]
    assert keybindings.get_keys("tui.editor.pageUp") == ["pageUp", "ctrl+pageUp"]
    assert keybindings.get_keys("tui.editor.pageDown") == ["pageDown", "ctrl+pageDown"]


def test_leaves_dedicated_prompt_history_navigation_unbound_by_default():
    keybindings = KeybindingsManager(TUI_KEYBINDINGS)

    assert keybindings.get_keys("tui.editor.historyPrevious") == []
    assert keybindings.get_keys("tui.editor.historyNext") == []


def test_binds_unmodified_terminal_viewport_shortcuts_to_alternate_screen_navigation():
    keybindings = KeybindingsManager(TUI_KEYBINDINGS)

    assert keybindings.get_keys("tui.altScreen.pageUp") == ["pageUp"]
    assert keybindings.get_keys("tui.altScreen.pageDown") == ["pageDown"]
    assert keybindings.get_keys("tui.altScreen.halfPageUp") == []
    assert keybindings.get_keys("tui.altScreen.halfPageDown") == []
    assert keybindings.get_keys("tui.altScreen.previousPrompt") == ["ctrl+shift+up"]
    assert keybindings.get_keys("tui.altScreen.nextPrompt") == ["ctrl+shift+down"]
    assert keybindings.get_keys("tui.altScreen.top") == ["home"]
    assert keybindings.get_keys("tui.altScreen.bottom") == ["end"]


def test_does_not_evict_selector_confirm_when_input_submit_is_rebound():
    keybindings = KeybindingsManager(
        TUI_KEYBINDINGS,
        {"tui.input.submit": ["enter", "ctrl+enter"]},
    )

    assert keybindings.get_keys("tui.input.submit") == ["enter", "ctrl+enter"]
    assert keybindings.get_keys("tui.select.confirm") == ["enter"]


def test_does_not_evict_cursor_bindings_when_another_action_reuses_the_same_key():
    keybindings = KeybindingsManager(
        TUI_KEYBINDINGS,
        {"tui.select.up": ["up", "ctrl+p"]},
    )

    assert keybindings.get_keys("tui.select.up") == ["up", "ctrl+p"]
    assert keybindings.get_keys("tui.editor.cursorUp") == ["up"]


def test_still_reports_direct_user_binding_conflicts_without_evicting_defaults():
    keybindings = KeybindingsManager(
        TUI_KEYBINDINGS,
        {
            "tui.input.submit": "ctrl+x",
            "tui.select.confirm": "ctrl+x",
        },
    )

    assert keybindings.get_conflicts() == [
        {
            "key": "ctrl+x",
            "keybindings": ["tui.input.submit", "tui.select.confirm"],
        }
    ]
    assert keybindings.get_keys("tui.editor.cursorLeft") == ["left", "ctrl+b"]
