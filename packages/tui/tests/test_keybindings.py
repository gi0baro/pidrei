"""Mirror of pi tui test/keybindings.test.ts."""

from pidrei_tui.keybindings import TUI_KEYBINDINGS, KeybindingsManager


def test_binds_ctrl_j_as_default_newline_alias():
    keybindings = KeybindingsManager(TUI_KEYBINDINGS)

    assert keybindings.get_keys("tui.input.newLine") == ["shift+enter", "ctrl+j"]
    assert keybindings.matches("\n", "tui.input.newLine") is True
    assert keybindings.matches("\x1b[106;5u", "tui.input.newLine") is True


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
