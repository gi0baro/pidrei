"""Mirror of pi coding-agent test/custom-editor-keybindings.test.ts."""

import sys
from pathlib import Path

import pytest
import tonio.colored as tonio

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.custom_editor import CustomEditor
from pidrei.modes.interactive.theme import get_editor_theme, init_theme
from pidrei_tui import Editor, set_keybindings
from pidrei_tui.tui_main_screen import TuiMainScreen


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


@pytest.fixture(autouse=True)
async def _restore_keybindings(request):
    await init_theme("dark")
    request.addfinalizer(lambda: set_keybindings(KeybindingsManager.in_memory()))


@pytest.mark.tonio
async def test_gives_an_explicit_history_binding_precedence_over_model_cycling():
    keybindings = KeybindingsManager.in_memory(
        {"tui.editor.historyPrevious": "ctrl+p", "tui.editor.historyNext": "ctrl+n"}
    )
    set_keybindings(keybindings)
    editor = CustomEditor(TuiMainScreen(VirtualTerminal()), get_editor_theme(), keybindings)
    model_cycles = 0

    async def on_cycle() -> None:
        nonlocal model_cycles
        model_cycles += 1

    editor.on_action("app.model.cycleForward", on_cycle)
    editor.add_to_history("previous prompt")
    editor.set_text("draft")

    editor.handle_input("\x10")  # Ctrl+P
    assert editor.get_text() == "previous prompt"
    assert model_cycles == 0

    editor.handle_input("\x0e")  # Ctrl+N
    assert editor.get_text() == "draft"


# CustomEditor selection page keybindings


@pytest.mark.tonio
async def test_pages_an_open_autocomplete_menu_before_pasting_an_image_with_the_same_key(monkeypatch):
    # #7629
    # Bind both explicitly: the paste default is Alt+V on Windows and WSL.
    keybindings = KeybindingsManager.in_memory({"tui.select.pageDown": "ctrl+v", "app.clipboard.pasteImage": "ctrl+v"})
    set_keybindings(keybindings)
    # pi waits for the menu with `vi.waitFor`; here the autocomplete request the editor spawns sets an
    # Event when it has finished, its apply included.
    request_done = tonio.Event()
    run_request = Editor._run_autocomplete_request

    def tracked_request(self, *args, **kwargs):
        request = run_request(self, *args, **kwargs)

        async def run() -> None:
            try:
                await request
            finally:
                request_done.set()

        return run()

    monkeypatch.setattr(Editor, "_run_autocomplete_request", tracked_request)
    editor = CustomEditor(TuiMainScreen(VirtualTerminal()), get_editor_theme(), keybindings)
    pastes = 0

    def on_paste_image() -> None:
        nonlocal pastes
        pastes += 1

    editor.on_paste_image = on_paste_image
    submitted: list[str] = []
    editor.on_submit = submitted.append

    class Provider:
        async def get_suggestions(self, *_args, **_kwargs):
            items = [{"value": f"/command-{index}", "label": f"command-{index}"} for index in range(12)]
            return {"items": items, "prefix": "/"}

        def apply_completion(self, lines, cursor_line, _cursor_col, item, _prefix):
            return {
                "lines": [item["value"] if index == cursor_line else line for index, line in enumerate(lines)],
                "cursorLine": cursor_line,
                "cursorCol": len(item["value"]),
            }

    editor.set_autocomplete_provider(Provider())

    editor.handle_input("/")
    await request_done.wait(5)
    assert editor.is_showing_autocomplete() is True
    editor.handle_input("\x16")  # Ctrl+V
    editor.handle_input("\r")

    assert pastes == 0
    assert submitted == ["/command-5"]

    editor.handle_input("\x16")  # Ctrl+V with autocomplete closed pastes again
    assert pastes == 1
