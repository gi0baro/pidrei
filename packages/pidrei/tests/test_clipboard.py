"""Mirror of pi coding-agent test/clipboard.test.ts (Wayland read cases).

pi's file mocks `child_process` and its native clipboard addon; pidrei has no
native addon (see `clipboard.py`), so what is mirrored is the part that still
exists: the Wayland reader is preferred, is invoked with the exact `wl-paste`
argv, and an empty clipboard reads as None rather than falling through.
"""

import contextlib
import os
import sys

import pytest

from pidrei.utils import clipboard


@contextlib.contextmanager
def _wayland_session(output: str | None):
    """Force the Wayland branch and record the argv it runs."""
    calls: list[list[str]] = []

    async def read_output(command: list) -> str | None:
        calls.append(command)
        return output

    previous_env = {name: os.environ.get(name) for name in ("WAYLAND_DISPLAY", "TERMUX_VERSION", "DISPLAY")}
    os.environ["WAYLAND_DISPLAY"] = "wayland-0"
    os.environ.pop("TERMUX_VERSION", None)
    os.environ.pop("DISPLAY", None)
    original_is_wayland = clipboard.is_wayland_session
    original_read_output = clipboard._read_output
    clipboard.is_wayland_session = lambda: True
    clipboard._read_output = read_output
    try:
        yield calls
    finally:
        clipboard.is_wayland_session = original_is_wayland
        clipboard._read_output = original_read_output
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# pi fakes `process.platform`; `sys.platform` is process-global and these tests
# run on a threaded runtime, so faking it would leak across tasks. `read_clipboard_text`
# checks darwin before Wayland, so on macOS the branch under test is unreachable
# and the reader really does run `pbpaste` — nothing here is worth asserting there.
NON_DARWIN_ONLY = pytest.mark.skipif(sys.platform == "darwin", reason="the Wayland reader is unreachable on macOS")


@NON_DARWIN_ONLY
@pytest.mark.tonio
async def test_reads_the_wayland_clipboard_as_plain_text():
    with _wayland_session("Wayland text") as calls:
        assert await clipboard.read_clipboard_text() == "Wayland text"

    assert calls == [["wl-paste", "--no-newline", "--type", "text"]]


@NON_DARWIN_ONLY
@pytest.mark.tonio
async def test_reads_an_empty_wayland_clipboard_as_none():
    with _wayland_session(""):
        assert await clipboard.read_clipboard_text() is None
