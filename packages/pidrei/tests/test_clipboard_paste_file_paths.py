"""Mirror of pi coding-agent test/clipboard-paste-file-paths.test.ts.

pi grabs `handleClipboardPaste` off InteractiveMode.prototype and calls it on
a fake `this` with the clipboard modules mocked; here the method is called on
a stub object and the three readers are swapped on the interactive_mode
module. pidrei's own `read_clipboard_file_paths` never returns paths (no
native helper), so these cases pin the handler's contract for when one does.
"""

import contextlib
import threading
from types import SimpleNamespace

import pytest

from pidrei.modes.interactive import interactive_mode
from pidrei.modes.interactive.interactive_mode import InteractiveMode


@contextlib.contextmanager
def _clipboard(file_paths=None, *, file_paths_error: Exception | None = None):
    calls = {"image": 0, "text": 0}

    async def read_clipboard_file_paths():
        if file_paths_error is not None:
            raise file_paths_error
        return file_paths

    async def read_clipboard_image():
        calls["image"] += 1
        return {"bytes": bytes([0x89, 0x50, 0x4E, 0x47]), "mimeType": "image/png"}

    async def read_clipboard_text():
        calls["text"] += 1

    originals = (
        interactive_mode.read_clipboard_file_paths,
        interactive_mode.read_clipboard_image,
        interactive_mode.read_clipboard_text,
    )
    interactive_mode.read_clipboard_file_paths = read_clipboard_file_paths
    interactive_mode.read_clipboard_image = read_clipboard_image
    interactive_mode.read_clipboard_text = read_clipboard_text
    try:
        yield calls
    finally:
        (
            interactive_mode.read_clipboard_file_paths,
            interactive_mode.read_clipboard_image,
            interactive_mode.read_clipboard_text,
        ) = originals


def _context(editor, *, is_bash_mode: bool = False) -> SimpleNamespace:
    errors: list[str] = []
    return SimpleNamespace(
        editor=editor,
        _is_bash_mode=is_bash_mode,
        show_error=errors.append,
        errors=errors,
        ui=SimpleNamespace(state_lock=threading.Lock(), request_render=lambda: None),
    )


def _editor(text: str | None = None, cursor_col: int | None = None) -> SimpleNamespace:
    inserted: list[str] = []
    editor = SimpleNamespace(insert_text_at_cursor=inserted.append, inserted=inserted)
    if text is not None:
        editor.get_text = lambda: text
        editor.get_cursor = lambda: {"line": 0, "col": cursor_col}
    return editor


@pytest.mark.tonio
async def test_finder_file_paths_take_precedence_over_their_icon_image():
    # Regression test for #9999.
    file_paths = ["/tmp/screenshot.png", "/tmp/My Photos/photo.png"]
    context = _context(_editor())

    with _clipboard(file_paths) as calls:
        await InteractiveMode._handle_clipboard_paste(context)

    assert context.editor.inserted == ["\n".join(file_paths)]
    assert calls["image"] == 0


@pytest.mark.tonio
async def test_clipboard_file_paths_containing_terminal_control_characters_are_rejected():
    context = _context(_editor())

    with _clipboard(["/tmp/photo\x1b]0;unsafe\x07.png"]) as calls:
        await InteractiveMode._handle_clipboard_paste(context)

    assert context.editor.inserted == []
    assert calls["image"] == 0
    assert context.errors == ["Failed to paste from clipboard: Clipboard file path contains control characters"]


@pytest.mark.tonio
async def test_native_file_path_errors_are_shown_without_falling_through_to_the_icon_image():
    context = _context(_editor())

    with _clipboard(file_paths_error=Exception("Native clipboard file read failed")) as calls:
        await InteractiveMode._handle_clipboard_paste(context)

    assert context.editor.inserted == []
    assert calls["image"] == 0
    assert context.errors == ["Failed to paste from clipboard: Native clipboard file read failed"]


@pytest.mark.tonio
@pytest.mark.parametrize(("editor_text", "cursor_col"), [("Review:", 7), ("確認", 2)], ids=["punctuation", "unicode"])
async def test_clipboard_file_paths_are_separated_from_preceding_text(editor_text, cursor_col):
    context = _context(_editor(editor_text, cursor_col))

    with _clipboard(["/tmp/photo.png"]) as calls:
        await InteractiveMode._handle_clipboard_paste(context)

    assert context.editor.inserted == [" /tmp/photo.png"]
    assert calls["image"] == 0


@pytest.mark.tonio
async def test_bash_mode_shell_quotes_file_paths_and_inserts_them_as_arguments():
    context = _context(_editor("catDEST", 3), is_bash_mode=True)

    with _clipboard(["/tmp/My Photos/photo.png", "/tmp/$(touch hacked).png", "/tmp/plain.png"]) as calls:
        await InteractiveMode._handle_clipboard_paste(context)

    assert context.editor.inserted == [" '/tmp/My Photos/photo.png' '/tmp/$(touch hacked).png' /tmp/plain.png "]
    assert calls["image"] == 0
