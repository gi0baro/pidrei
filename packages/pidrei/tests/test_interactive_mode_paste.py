"""Mirror of pi coding-agent test/interactive-mode-paste.test.ts.

pi drives InteractiveMode.prototype methods on plain host objects; here
``_Host`` does the same with the unbound methods. Factories are synchronous
here, so "after it fails to open" uses a raising factory and the error comes
out of the `_show_extension_custom()` call itself (pi: the returned promise
rejects).
"""

import sys
import threading
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from pidrei.core.keybindings import KeybindingsManager
from pidrei.modes.interactive.components.custom_editor import CustomEditor
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import get_editor_theme, init_theme
from pidrei_tui import Container, Editor, set_keybindings
from pidrei_tui.tui_main_screen import TuiMainScreen


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tui" / "tests"))
from virtual_terminal import VirtualTerminal


# Pastes over 10 lines collapse into a marker whose content only the receiving editor can expand.
COLLAPSED_PASTE_LINE_COUNT = 12


def _pasted_text(label: str) -> str:
    return "\n".join(f"{label} {i + 1}" for i in range(COLLAPSED_PASTE_LINE_COUNT))


def _paste(editor, text: str) -> None:
    editor.handle_input(f"\x1b[200~{text}\x1b[201~")
    # Without a collapsed marker, the transition under test would copy plain text and prove nothing.
    assert editor.get_text() != text


def _submit(editor) -> str | None:
    submitted = None

    def on_submit(text: str) -> None:
        nonlocal submitted
        submitted = text

    editor.on_submit = on_submit
    editor.handle_input("\r")
    return submitted


class _Host(SimpleNamespace):
    """Attributes set here shadow InteractiveMode; every other name is its method, bound to this host."""

    def __getattr__(self, name):
        return partial(getattr(InteractiveMode, name), self)


@pytest.fixture
async def paste_setup():
    await init_theme("dark")
    keybindings = KeybindingsManager.in_memory()
    set_keybindings(keybindings)
    yield SimpleNamespace(keybindings=keybindings, ui=TuiMainScreen(VirtualTerminal()))
    set_keybindings(KeybindingsManager.in_memory())


# Regression tests for #9809.


@pytest.mark.tonio
async def test_keeps_collapsed_paste_content_between_the_default_and_a_replacement_editor(paste_setup):
    ui = paste_setup.ui
    default_editor = CustomEditor(ui, get_editor_theme(), paste_setup.keybindings)
    host = _Host(
        ui=ui,
        _keybindings=paste_setup.keybindings,
        _extension_tui=ui,
        _default_editor=default_editor,
        editor=default_editor,
        _editor_container=Container(),
        _extension_dialogs=[],
        _editor_component_factory=None,
        _autocomplete_provider=None,
        _active_status_indicator=None,
        _dispose_active_selector=lambda: None,
    )

    into_replacement = _pasted_text("into replacement")
    _paste(host.editor, into_replacement)
    host._set_custom_editor_component(lambda tui, theme, _keybindings: Editor(tui, theme))
    assert _submit(host.editor) == into_replacement

    back_to_default = _pasted_text("back to default")
    _paste(host.editor, back_to_default)
    host._set_custom_editor_component(None)
    assert _submit(host.editor) == back_to_default


def _create_custom_ui_host(paste_setup) -> _Host:
    ui = paste_setup.ui
    return _Host(
        ui=ui,
        _keybindings=paste_setup.keybindings,
        _extension_tui=ui,
        editor=Editor(ui, get_editor_theme()),
        _editor_container=Container(),
        _extension_dialogs=[],
        _program_status=SimpleNamespace(set_blocked=lambda _source, _status: None),
        _dispose_active_selector=lambda: None,
    )


@pytest.mark.tonio
async def test_keeps_collapsed_paste_content_around_non_overlay_custom_ui_after_it_closes(paste_setup):
    host = _create_custom_ui_host(paste_setup)
    draft = _pasted_text("draft")
    _paste(host.editor, draft)
    component = Container()
    closers = []

    def factory(_tui, _theme, _keybindings, done):
        closers.append(done)
        return component

    shown = host._show_extension_custom(factory)
    assert host._editor_container.children[0] is component
    closers[0]("closed")
    assert await shown == "closed"

    assert _submit(host.editor) == draft


@pytest.mark.tonio
async def test_keeps_collapsed_paste_content_around_non_overlay_custom_ui_after_it_fails_to_open(paste_setup):
    host = _create_custom_ui_host(paste_setup)
    draft = _pasted_text("draft")
    _paste(host.editor, draft)

    def factory(_tui, _theme, _keybindings, _done):
        raise RuntimeError("factory failed")

    with pytest.raises(RuntimeError, match="factory failed"):
        host._show_extension_custom(factory)

    assert _submit(host.editor) == draft


@pytest.mark.tonio
async def test_keeps_collapsed_paste_content_when_queued_messages_are_restored_above_it(paste_setup):
    editor = Editor(paste_setup.ui, get_editor_theme())
    host = _Host(
        editor=editor,
        ui=SimpleNamespace(state_lock=threading.RLock(), request_render=lambda force=False: None),
        _clear_all_queues=lambda: {"steering": [], "followUp": ["queued follow-up"]},
        _update_pending_messages_display=lambda: None,
    )

    draft = _pasted_text("draft")
    _paste(editor, draft)
    host._restore_queued_messages_to_editor()

    assert _submit(editor) == f"queued follow-up\n\n{draft}"
