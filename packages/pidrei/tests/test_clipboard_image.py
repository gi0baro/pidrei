"""Mirror of pi coding-agent test/clipboard-image.test.ts.

pi mocks `runClipboardCommand` and its native clipboard helper; here
`run_clipboard_command` is swapped on the module. pidrei has no native helper
(see `clipboard_image.py`), so the darwin/win32 native-read cases, "returns null
without a native helper" and the native-error propagation cases have nothing to
mirror, and the Linux cases where pi's last resort is the native X11 reader
assert pidrei's end of the chain instead: no image.
"""

import contextlib

import pytest
from tonio.colored import fs

from pidrei.utils import clipboard_image
from pidrei.utils.clipboard_image import read_clipboard_image


PNG = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 13, 0x49, 0x48, 0x44, 0x52])


@contextlib.contextmanager
def _commands(result):
    """Swap the command runner; `result(command, args)` is awaited per call."""
    calls: list[tuple[str, list[str]]] = []

    async def run(command, args, **_options):
        calls.append((command, list(args)))
        return await result(command, args)

    original = clipboard_image.run_clipboard_command
    clipboard_image.run_clipboard_command = run
    try:
        yield calls
    finally:
        clipboard_image.run_clipboard_command = original


@pytest.mark.tonio
@pytest.mark.parametrize("present", [True, False])
@pytest.mark.parametrize(
    ("backend", "command", "env"),
    [
        ("wayland", "wl-paste", {"WAYLAND_DISPLAY": "1", "DISPLAY": ":0"}),
        ("x11", "xclip", {"DISPLAY": ":0"}),
    ],
)
async def test_command_image_presence_stops_fallback(backend, command, env, present):
    async def result(name, args):
        assert name == command
        listing = "--list-types" in args or "TARGETS" in args
        if listing:
            return b"text/plain\nimage/png\n" if present else b"text/plain\n"
        return PNG

    with _commands(result) as calls:
        image = await read_clipboard_image({"platform": "linux", "env": env})

    assert image == ({"bytes": PNG, "mimeType": "image/png"} if present else None)
    assert len(calls) == (2 if present else 1)


@pytest.mark.tonio
async def test_x11_command_failures_read_as_no_image():
    # pi's native X11 reader answers after these failures; pidrei has none.
    async def result(_name, _args):
        return None

    with _commands(result) as calls:
        assert await read_clipboard_image({"platform": "linux", "env": {"DISPLAY": ":0"}}) is None

    assert [name for name, _ in calls] == ["xclip"] * 5


@pytest.mark.tonio
async def test_wayland_falls_back_to_x11_after_wl_paste_fails():
    async def result(name, args):
        if name == "wl-paste":
            return None
        return b"image/png\n" if "TARGETS" in args else PNG

    with _commands(result):
        image = await read_clipboard_image({"platform": "linux", "env": {"WAYLAND_DISPLAY": "1"}})

    assert image == {"bytes": PNG, "mimeType": "image/png"}


@pytest.mark.tonio
async def test_wsl_tries_powershell_after_the_linux_tools_fail():
    tmp_file: str | None = None

    async def result(name, args):
        nonlocal tmp_file
        if name in ("wl-paste", "xclip"):
            return None
        if name == "wslpath":
            tmp_file = args[1]
            return b"C:\\Users\\O'Hare\\clip.png\n"
        if name == "powershell.exe":
            assert "$path = 'C:\\Users\\O''Hare\\clip.png'" in args[2]
            assert tmp_file is not None, "wslpath should be called before powershell.exe"
            await fs.Path(tmp_file).write_bytes(PNG)
            return b"ok\n"
        raise AssertionError(f"Unexpected command: {name}")

    with _commands(result):
        image = await read_clipboard_image({"platform": "linux", "env": {"WSL_DISTRO_NAME": "Ubuntu"}})

    assert image == {"bytes": PNG, "mimeType": "image/png"}


@pytest.mark.tonio
async def test_termux_does_not_read_image_clipboards():
    async def result(_name, _args):
        raise AssertionError("no command expected")

    with _commands(result) as calls:
        assert await read_clipboard_image({"platform": "linux", "env": {"TERMUX_VERSION": "0.119"}}) is None

    assert calls == []
