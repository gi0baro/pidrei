"""Mirror of pi coding-agent test/clipboard.test.ts.

pi mocks its native clipboard helper, `runClipboardCommand` and
`os.platform()`. pidrei has no native helper (see `clipboard.py`), so the cases
that only exercise it (native reads/writes, rejected or read-only native
writers) have nothing to mirror; `run_clipboard_command` is swapped on the
module instead. `sys.platform` is process-global and these tests run on a
threaded runtime, so it is not faked: the Linux candidate chains are asserted
off macOS only (on macOS pbpaste/pbcopy stand in for the native helper).
"""

import contextlib
import io
import os
import sys

import pytest
import tonio.colored as tonio

from pidrei.utils import clipboard


_ENV_NAMES = ("SSH_CONNECTION", "SSH_CLIENT", "MOSH_CONNECTION", "WAYLAND_DISPLAY", "DISPLAY", "TERMUX_VERSION")

LINUX_ONLY = pytest.mark.skipif(sys.platform != "linux", reason="asserts the Linux clipboard candidate chain")


@contextlib.contextmanager
def _clipboard(result=b"", **env: str):
    """Swap the command runner, stub the env and capture stdout; yields (calls, stdout).

    `result` is the runner's answer: bytes/None for every command, or a callable
    `(command, args) -> awaitable of bytes | None`.
    """
    calls: list[tuple[str, list[str], dict]] = []

    async def run(command, args, **options):
        calls.append((command, list(args), options))
        if callable(result):
            return await result(command, args)
        return result

    previous_env = {name: os.environ.get(name) for name in _ENV_NAMES}
    for name in _ENV_NAMES:
        os.environ.pop(name, None)
    os.environ.update(env)
    original_run = clipboard.run_clipboard_command
    clipboard.run_clipboard_command = run
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            yield calls, stdout
    finally:
        clipboard.run_clipboard_command = original_run
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _names(calls) -> list[str]:
    return [command for command, _, _ in calls]


def _osc52_writes(stdout: io.StringIO) -> int:
    return stdout.getvalue().count("\x1b]52;c;")


# -- readClipboardText --------------------------------------------------------


@LINUX_ONLY
@pytest.mark.tonio
@pytest.mark.parametrize("text", ["clipboard text", ""])
@pytest.mark.parametrize(
    ("env", "command", "args", "expected_calls"),
    [
        ("WAYLAND_DISPLAY", "wl-paste", ["--no-newline", "--type", "text"], ["wl-paste"]),
        ("DISPLAY", "xclip", ["-selection", "clipboard", "-out"], ["xclip"]),
        ("DISPLAY", "xsel", ["--clipboard", "--output"], ["xclip", "xsel"]),
        ("TERMUX_VERSION", "termux-clipboard-get", [], ["termux-clipboard-get"]),
    ],
)
async def test_command_result_stops_fallback(env, command, args, expected_calls, text):
    # Regression test for #7248: empty Wayland content must not fall through to stale X11.
    async def result(name, _args):
        return text.encode() if name == command else None

    with _clipboard(result, **{"DISPLAY": ":0", env: "1"}) as (calls, _):
        assert await clipboard.read_clipboard_text() == (text or None)

    assert _names(calls) == expected_calls
    assert calls[-1] == (command, args, {"timeout_ms": 5000})


@LINUX_ONLY
@pytest.mark.tonio
async def test_reads_none_after_command_failures():
    # pi falls back to its native X11 reader here; pidrei has none.
    with _clipboard(None, DISPLAY=":0", WAYLAND_DISPLAY="wayland-0") as (calls, _):
        assert await clipboard.read_clipboard_text() is None

    assert _names(calls) == ["wl-paste", "xclip", "xsel"]


@LINUX_ONLY
@pytest.mark.tonio
async def test_falls_back_to_x11_tools_when_wl_paste_is_unavailable():
    async def result(name, _args):
        return None if name == "wl-paste" else b"X11 text"

    with _clipboard(result, WAYLAND_DISPLAY="wayland-0", DISPLAY=":0"):
        assert await clipboard.read_clipboard_text() == "X11 text"


# -- copyToClipboard ----------------------------------------------------------


@LINUX_ONLY
@pytest.mark.tonio
async def test_linux_writes_through_the_platform_tools():
    with _clipboard(DISPLAY=":0") as (calls, stdout):
        await clipboard.copy_to_clipboard("hello")

    assert calls == [("xclip", ["-selection", "clipboard"], {"input": "hello", "timeout_ms": 5000})]
    assert _osc52_writes(stdout) == 0


@LINUX_ONLY
@pytest.mark.tonio
async def test_waits_for_the_command_write_before_emitting_remote_osc52():
    started = tonio.Event()
    complete = tonio.Event()

    async def result(_name, _args):
        started.set()
        await complete.wait()
        return b""

    with _clipboard(result, DISPLAY=":0", SSH_CONNECTION="client server") as (calls, stdout):
        copy = tonio.spawn(clipboard.copy_to_clipboard("hello"))
        await started.wait(5)
        assert started.is_set()
        assert _osc52_writes(stdout) == 0
        complete.set()
        await copy

    assert _osc52_writes(stdout) == 1
    assert _names(calls) == ["xclip"]


@LINUX_ONLY
@pytest.mark.tonio
async def test_tries_xclip_and_xsel_after_wl_copy_fails():
    async def result(name, _args):
        return b"" if name == "xsel" else None

    with _clipboard(result, WAYLAND_DISPLAY="wayland-0", DISPLAY=":0") as (calls, stdout):
        await clipboard.copy_to_clipboard("hello")

    assert _names(calls) == ["wl-copy", "xclip", "xsel"]
    assert _osc52_writes(stdout) == 0


@pytest.mark.tonio
async def test_uses_osc52_when_command_writes_fail():
    with _clipboard(None) as (_, stdout):
        await clipboard.copy_to_clipboard("hello")

    assert _osc52_writes(stdout) == 1


@pytest.mark.tonio
async def test_does_not_emit_oversized_osc52_payloads():
    with _clipboard(None) as (_, stdout), pytest.raises(Exception, match="Failed to copy to clipboard"):
        await clipboard.copy_to_clipboard("x" * 80_000)

    assert _osc52_writes(stdout) == 0
