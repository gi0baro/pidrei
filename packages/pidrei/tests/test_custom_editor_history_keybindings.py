"""Mirror of pi coding-agent test/custom-editor-history-keybindings.test.ts."""

import sys
from pathlib import Path

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.custom_editor import CustomEditor
from pidrei.modes.interactive.theme import get_editor_theme, init_theme_sync
from pidrei_tui import set_keybindings
from pidrei_tui.tui_main_screen import TuiMainScreen


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


@pytest.fixture(autouse=True)
def _restore_keybindings(request):
    init_theme_sync("dark")
    request.addfinalizer(lambda: set_keybindings(KeybindingsManager()))


@pytest.mark.tonio
async def test_gives_an_explicit_history_binding_precedence_over_model_cycling():
    keybindings = KeybindingsManager({"tui.editor.historyPrevious": "ctrl+p", "tui.editor.historyNext": "ctrl+n"})
    set_keybindings(keybindings)
    editor = CustomEditor(TuiMainScreen(VirtualTerminal()), get_editor_theme(), keybindings)
    model_cycles = 0

    async def on_cycle() -> None:
        nonlocal model_cycles
        model_cycles += 1

    editor.on_action("app.model.cycleForward", on_cycle)
    editor.add_to_history("previous prompt")
    editor.set_text("draft")

    await editor.handle_input("\x10")  # Ctrl+P
    assert editor.get_text() == "previous prompt"
    assert model_cycles == 0

    await editor.handle_input("\x0e")  # Ctrl+N
    assert editor.get_text() == "draft"
