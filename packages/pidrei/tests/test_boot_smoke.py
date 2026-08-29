"""pidrei-specific: the real CLI boots interactive mode and completes a turn.

No pi counterpart — this exists because 2,199 green mirrored tests coexisted
with an interactive mode that crashed in `init()` (PLAN.md Phase 4.5). Every
other test drives methods against hand-built fakes; nothing constructed
InteractiveMode against a real AgentSession, so four defects (an action-dict
key mismatch, an `await` on a sync method, two un-awaited coroutines and a
method referenced without calling it) were invisible.

The whole stack runs for real: `python -m pidrei` in a pty, a models.json
custom provider pointing at a local fake openai-completions endpoint served
here, one prompt typed on the pty, the reply asserted on an emulated screen.
Hermetic: no network beyond loopback, agent dir and HOME redirected to a
temp dir.
"""

import fcntl
import json
import os
import pty
import re
import signal
import struct
import subprocess
import sys
import termios

import pyte
import pytest
import tonio.colored as tonio
from tonio.colored import net

from pidrei.config import ENV_AGENT_DIR


COLS, ROWS = 100, 30
BOOT_TIMEOUT = 40.0
REPLY_TIMEOUT = 40.0
EXIT_TIMEOUT = 15.0

# APC (the renderer's cursor marker) is stripped before feeding pyte, which
# does not implement it — same deviation as the tui package's harness.
_APC_RE = re.compile(r"\x1b_.*?\x07", re.DOTALL)

_CHUNK = {
    "id": "chatcmpl-smoke",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "demo-model",
}


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


_RESPONSE_BODY = b"".join(
    [
        _sse({**_CHUNK, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "pong"}}]}),
        _sse({**_CHUNK, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        _sse(
            {
                **_CHUNK,
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13},
            }
        ),
        b"data: [DONE]\n\n",
    ]
)


async def _handle_request(stream) -> None:
    """Answer one chat-completions request with a canned SSE stream."""
    request = b""
    while b"\r\n\r\n" not in request:
        chunk = await stream.receive_some()
        if not chunk:
            stream.close()
            return
        request += chunk

    await stream.send_all(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Cache-Control: no-cache\r\n"
        b"Connection: close\r\n"
        b"\r\n" + _RESPONSE_BODY
    )
    stream.close()


async def _accept_loop(listener) -> None:
    while True:
        try:
            stream = await listener.accept()
        except Exception:
            return  # listener closed by the test
        tonio.spawn.without_tracking(_handle_request(stream))


def _write_agent_dir(agent_dir, port: int) -> None:
    agent_dir.mkdir(parents=True, exist_ok=True)
    models = {
        "providers": {
            "smoke": {
                "name": "Smoke",
                "baseUrl": f"http://127.0.0.1:{port}/v1",
                "apiKey": "smoke-key",
                "api": "openai-completions",
                "models": [
                    {
                        "id": "demo-model",
                        "name": "Demo Model",
                        "reasoning": False,
                        "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": 128000,
                        "maxTokens": 4096,
                    }
                ],
            }
        }
    }
    (agent_dir / "models.json").write_text(json.dumps(models))
    (agent_dir / "auth.json").write_text("{}")


class _Screen:
    """Minimal terminal emulator over the pty master."""

    def __init__(self, fd: int):
        self._fd = fd
        self._screen = pyte.Screen(COLS, ROWS)
        self._stream = pyte.Stream(self._screen)
        self.raw = ""

    def pump(self) -> None:
        while True:
            try:
                data = os.read(self._fd, 65536)
            except BlockingIOError, OSError:
                return
            if not data:
                return
            text = data.decode("utf-8", "replace")
            self.raw += text
            self._stream.feed(_APC_RE.sub("", text))

    @property
    def text(self) -> str:
        return "\n".join(line.rstrip() for line in self._screen.display)


def _assert_no_child_error(raw: str, marker: str) -> None:
    """Fail with the offending output, not a truncated repr of the whole pty."""
    if marker not in raw:
        return
    excerpt = _APC_RE.sub("", raw[raw.index(marker) : raw.index(marker) + 3000])
    raise AssertionError(f"child process printed {marker}:\n{excerpt}")


async def _wait_for(screen: _Screen, needle: str, timeout: float) -> None:
    waited = 0.0
    while waited < timeout:
        screen.pump()
        if needle in screen.text:
            return
        await tonio.sleep(0.1)
        waited += 0.1
    raise AssertionError(f"timed out waiting for {needle!r}; screen was:\n{screen.text}")


@pytest.mark.tonio
async def test_interactive_mode_boots_and_completes_a_turn(tmp_path):
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)

    listener = (await net.open_tcp_listeners(0, host="127.0.0.1"))[0]
    port = listener.socket.getsockname()[1]
    tonio.spawn.without_tracking(_accept_loop(listener))
    _write_agent_dir(agent_dir, port)

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    os.set_blocking(master, False)

    # ASYNC220: the child is a long-lived pty session driven below, not an
    # awaited subprocess call; tonio.open_process cannot hand it a pty.
    process = subprocess.Popen(  # noqa: ASYNC220
        [sys.executable, "-m", "pidrei", "--model", "smoke/demo-model"],
        cwd=str(project_dir),
        env={
            **os.environ,
            ENV_AGENT_DIR: str(agent_dir),
            "HOME": str(tmp_path),
            "PIDREI_OFFLINE": "1",
            "TERM": "xterm-256color",
            "COLUMNS": str(COLS),
            "LINES": str(ROWS),
        },
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
    )
    os.close(slave)
    screen = _Screen(master)

    try:
        # Boots: banner rendered, footer bound to the configured model.
        await _wait_for(screen, "pidrei v", BOOT_TIMEOUT)
        await _wait_for(screen, "demo-model", BOOT_TIMEOUT)

        # Completes a turn against the local provider.
        os.write(master, b"say pong\r")
        await _wait_for(screen, "pong", REPLY_TIMEOUT)

        # Two Ctrl+C exit cleanly.
        os.write(master, b"\x03")
        await tonio.sleep(0.3)
        os.write(master, b"\x03")

        waited = 0.0
        while process.poll() is None and waited < EXIT_TIMEOUT:
            screen.pump()
            await tonio.sleep(0.1)
            waited += 0.1
        screen.pump()

        assert process.poll() == 0, f"exit code {process.poll()}; screen was:\n{screen.text}"
        _assert_no_child_error(screen.raw, "Traceback")
        # Interactive mode filters never-awaited coroutine warnings by design
        # (tonio abandonment noise; see main.py), so this marker guards every
        # *other* RuntimeWarning class. A keep-the-warnings escape hatch was
        # tried and reverted: the benign noise fired during Ctrl-C teardown on
        # a slow macOS runner and flaked this gate.
        _assert_no_child_error(screen.raw, "RuntimeWarning")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        os.close(master)
        listener.close()


_UI_PROBE_EXTENSION = """
def extension(pi):
    async def on_session_start(_event, ctx):
        # Exercise the documented `ctx.ui` surface against the real TUI:
        # awaitable theme accessors, the theme object, and the sync setters.
        themes = await ctx.ui.get_all_themes()
        ctx.ui.get_editor_text()
        marker = "EXT-STATUS-OK" if themes else "EXT-STATUS-NO-THEMES"
        ctx.ui.set_status("ui-probe", ctx.ui.theme.fg("accent", marker))
        ctx.ui.set_widget("ui-probe", ["EXT-WIDGET-OK"])
        ctx.ui.notify("EXT-NOTIFY-OK", "info")

    pi.on("session_start", on_session_start)
"""


@pytest.mark.tonio
async def test_extension_drives_ctx_ui_against_the_real_tui(tmp_path):
    """A real extension's `ctx.ui` calls against a real InteractiveMode.

    Guards the ctx.ui contract (task #86): production used to hand extensions
    a camelCase dict while docs/examples/no-op used snake_case attributes, and
    every unit test faked `ctx` with a SimpleNamespace — so the mismatch was
    invisible until an extension ran against the real TUI.
    """
    agent_dir = tmp_path / "agent"
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)

    listener = (await net.open_tcp_listeners(0, host="127.0.0.1"))[0]
    port = listener.socket.getsockname()[1]
    tonio.spawn.without_tracking(_accept_loop(listener))
    _write_agent_dir(agent_dir, port)
    extensions_dir = agent_dir / "extensions"
    extensions_dir.mkdir(parents=True, exist_ok=True)
    (extensions_dir / "ui_probe.py").write_text(_UI_PROBE_EXTENSION)

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    os.set_blocking(master, False)

    process = subprocess.Popen(  # noqa: ASYNC220 (same pty note as above)
        [sys.executable, "-m", "pidrei", "--model", "smoke/demo-model"],
        cwd=str(project_dir),
        env={
            **os.environ,
            ENV_AGENT_DIR: str(agent_dir),
            "HOME": str(tmp_path),
            "PIDREI_OFFLINE": "1",
            "TERM": "xterm-256color",
            "COLUMNS": str(COLS),
            "LINES": str(ROWS),
        },
        stdin=slave,
        stdout=slave,
        stderr=slave,
        start_new_session=True,
    )
    os.close(slave)
    screen = _Screen(master)

    try:
        await _wait_for(screen, "pidrei v", BOOT_TIMEOUT)
        # The probe ran: awaited theme accessors returned themes, the sync
        # setters landed in the footer status and the widget area.
        await _wait_for(screen, "EXT-STATUS-OK", BOOT_TIMEOUT)
        await _wait_for(screen, "EXT-WIDGET-OK", BOOT_TIMEOUT)

        os.write(master, b"\x03")
        await tonio.sleep(0.3)
        os.write(master, b"\x03")

        waited = 0.0
        while process.poll() is None and waited < EXIT_TIMEOUT:
            screen.pump()
            await tonio.sleep(0.1)
            waited += 0.1
        screen.pump()

        assert process.poll() == 0, f"exit code {process.poll()}; screen was:\n{screen.text}"
        _assert_no_child_error(screen.raw, "Traceback")
        # Interactive mode filters never-awaited coroutine warnings by design
        # (tonio abandonment noise; see main.py), so this marker guards every
        # *other* RuntimeWarning class. A keep-the-warnings escape hatch was
        # tried and reverted: the benign noise fired during Ctrl-C teardown on
        # a slow macOS runner and flaked this gate.
        _assert_no_child_error(screen.raw, "RuntimeWarning")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        os.close(master)
        listener.close()
