"""Mirror of pi tui test/editor-history-keybindings.test.ts."""

import pytest

from pidrei_tui.components.editor import Editor
from pidrei_tui.keybindings import TUI_KEYBINDINGS, KeybindingsManager, set_keybindings
from pidrei_tui.tui_main_screen import TuiMainScreen

from .themes import default_editor_theme
from .virtual_terminal import VirtualTerminal


@pytest.fixture(autouse=True)
def _restore_keybindings(request):
    request.addfinalizer(lambda: set_keybindings(KeybindingsManager(TUI_KEYBINDINGS)))


@pytest.mark.tonio
async def test_browses_history_directly_without_first_moving_the_cursor():
    set_keybindings(
        KeybindingsManager(
            TUI_KEYBINDINGS,
            {"tui.editor.historyPrevious": "ctrl+p", "tui.editor.historyNext": "ctrl+n"},
        )
    )
    editor = Editor(TuiMainScreen(VirtualTerminal()), default_editor_theme)
    editor.add_to_history("older prompt")
    editor.add_to_history("newer\nmultiline prompt")
    editor.set_text("draft")
    await editor.handle_input("\x1b[D")
    await editor.handle_input("\x1b[D")

    await editor.handle_input("\x10")  # Ctrl+P
    assert editor.get_text() == "newer\nmultiline prompt"
    assert editor.get_cursor() == {"line": 0, "col": 0}

    await editor.handle_input("\x10")  # Ctrl+P
    assert editor.get_text() == "older prompt"

    await editor.handle_input("\x0e")  # Ctrl+N
    assert editor.get_text() == "newer\nmultiline prompt"
    assert editor.get_cursor() == {"line": 1, "col": 16}

    await editor.handle_input("\x0e")  # Ctrl+N
    assert editor.get_text() == "draft"
    assert editor.get_cursor() == {"line": 0, "col": 3}
