"""pidrei-specific: ProcessTerminal end-to-end over a real pty.

No pi counterpart (node tests cannot re-point process.stdin at a pty); this
covers the port's tonio input pump, raw-mode handling, and live Kitty
negotiation, with the test playing the terminal-emulator side on the pty
master.
"""

import os
import pty
import termios

import pytest
import tonio.colored as tonio
from tonio._colored._fd import FdStream

from pidrei_tui.components import Text
from pidrei_tui.keys import set_kitty_protocol_active
from pidrei_tui.terminal import ProcessTerminal
from pidrei_tui.tui_main_screen import TuiMainScreen


async def _read_available(fd: int, wait: float = 0.05) -> bytes:
    await tonio.sleep(wait)
    chunks = []
    while True:
        try:
            chunk = os.read(fd, 65536)
        except BlockingIOError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.mark.tonio
async def test_pty_pump_negotiation_and_input_end_to_end():
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    inputs = []
    pastes_included = []

    async def record_input(data: str) -> None:
        # Terminal awaits its input handler now.
        inputs.append(data)

    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    try:
        await terminal.start(record_input, lambda: None)

        # Raw mode entered on the tty: echo off, canonical mode off.
        attrs = termios.tcgetattr(slave)
        assert attrs[3] & termios.ECHO == 0
        assert attrs[3] & termios.ICANON == 0
        # Output processing stays on (node's raw mode keeps it too).
        assert attrs[1] & termios.OPOST != 0

        # Startup wrote bracketed-paste enable and the Kitty query.
        startup = await _read_available(master)
        assert b"\x1b[?2004h" in startup
        assert b"\x1b[>7u\x1b[?u\x1b[c" in startup

        # Reply as a Kitty-capable terminal: protocol activates, nothing is
        # forwarded to the input handler.
        os.write(master, b"\x1b[?7u")
        await tonio.sleep(0.05)
        assert terminal.kitty_protocol_active is True
        assert inputs == []

        # Type some keys, including a multi-byte codepoint split across
        # chunk boundaries mid-UTF-8.
        os.write(master, b"hi \xf0\x9f")
        os.write(master, b"\x8e\x89")
        await tonio.sleep(0.1)
        assert inputs == ["h", "i", " ", "🎉"]

        # Bracketed paste is re-wrapped for the editor.
        inputs.clear()
        os.write(master, b"\x1b[200~pasted\x1b[201~")
        await tonio.sleep(0.1)
        pastes_included = list(inputs)
        assert pastes_included == ["\x1b[200~pasted\x1b[201~"]

        # Writes reach the pty unbuffered.
        await terminal.write("out")
        assert await _read_available(master) == b"out"
    finally:
        await terminal.stop()
        set_kitty_protocol_active(False)

    # stop() restored the tty state and disabled what it enabled.
    attrs = termios.tcgetattr(slave)
    assert attrs[3] & termios.ECHO != 0
    assert attrs[3] & termios.ICANON != 0
    teardown = await _read_available(master)
    assert b"\x1b[?2004l" in teardown
    assert b"\x1b[<u" in teardown

    os.close(master)
    os.close(slave)


@pytest.mark.tonio
async def test_pty_drain_input_returns_after_idle():
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    inputs = []

    async def record_input(data: str) -> None:
        inputs.append(data)

    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    try:
        await terminal.start(record_input, lambda: None)
        await _read_available(master)

        # Late input during drain is swallowed, and drain returns on idle
        # well before max_ms.
        os.write(master, b"\x1b[97;1:3u")
        await terminal.drain_input(2000, 50)
        assert inputs == []

        # The input handler is restored afterwards.
        os.write(master, b"x")
        await tonio.sleep(0.1)
        assert inputs == ["x"]
    finally:
        await terminal.stop()
        set_kitty_protocol_active(False)

    os.close(master)
    os.close(slave)


# output pump


@pytest.mark.tonio
async def test_output_pump_keeps_fifo_order_and_write_waits_for_a_slow_reader():
    """pi's synchronous `stdout.write` gives it ordering and completion for
    free; here both come from the single output pump. A sync writer
    (`set_title`) queued before a frame must reach the fd before it, and
    `write()` must not return until its bytes are out — on a pipe nobody is
    reading, it has to park until the reader drains."""
    in_r, in_w = os.pipe()
    out_r, out_w = os.pipe()
    os.set_blocking(out_r, False)
    terminal = ProcessTerminal(input_fd=in_r, output_fd=out_w)

    async def no_input(_data: str) -> None:
        pass

    try:
        await terminal.start(no_input, lambda: None)
        startup = await _read_available(out_r)
        assert startup.endswith(b"\x1b[>7u\x1b[?u\x1b[c")

        frame = "x" * (1 << 20)  # far beyond the pipe's capacity
        done = tonio.Event()

        async def write_frame() -> None:
            await terminal.write(frame)
            done.set()

        async with tonio.scope() as scope:
            terminal.set_title("before the frame")
            scope.spawn(write_frame())
            await tonio.sleep(0.05)
            assert not done.is_set(), "write() returned while the pipe was still full"

            received = b""
            while len(received) < len(b"\x1b]0;before the frame\x07") + len(frame):
                received += await _read_available(out_r, wait=0.001)
            assert received.startswith(b"\x1b]0;before the frame\x07")
            assert received[len(b"\x1b]0;before the frame\x07") :] == frame.encode()
            await done.wait(1.0)
            assert done.is_set(), "write() must complete once its bytes are on the wire"
    finally:
        await terminal.stop()
        for fd in (in_r, in_w, out_r, out_w):
            os.close(fd)
    set_kitty_protocol_active(False)


@pytest.mark.tonio
async def test_pty_input_survives_a_raising_input_handler():
    """A handler exception is routed to the owner's on_error and the pump
    keeps reading — input must not die for good (the 0.84.2.5 freeze)."""
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    inputs = []
    errors = []
    got_error = tonio.Event()
    got_y = tonio.Event()

    async def record_input(data: str) -> None:
        if data == "x":
            raise RuntimeError("handler blew up")
        inputs.append(data)
        if data == "y":
            got_y.set()

    def on_error(error: BaseException) -> None:
        errors.append(error)
        got_error.set()

    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    terminal.input_owner.on_error = on_error
    try:
        await terminal.start(record_input, lambda: None)

        os.write(master, b"x")
        await got_error.wait(2.0)
        assert got_error.is_set(), "handler exception must reach input_owner.on_error"
        assert [type(error).__name__ for error in errors] == ["RuntimeError"]

        os.write(master, b"y")
        await got_y.wait(2.0)
        assert got_y.is_set(), "input after a handler exception must still be delivered"
        assert inputs == ["y"]
    finally:
        await terminal.stop()
        set_kitty_protocol_active(False)
    os.close(master)
    os.close(slave)


# TUI island stress: input storms while mutations stream (PROPER_MT_DESIGN
# step 1). "Frozen" would show as the sentinel key never arriving or frames
# going silent — the 0.84.2.5 failure mode the island makes structural.


@pytest.mark.tonio
async def test_pty_tui_survives_an_input_storm_with_concurrent_mutations():
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    # The test plays the terminal-emulator side of the pty through tonio's
    # own fd stream (async send/receive on the runtime); the stream owns
    # `master` and closes it when dropped.
    emulator = FdStream(master)
    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    tui = TuiMainScreen(terminal)
    streamed = Text("start", 1, 0)

    received: list[str] = []
    sentinel_seen = tonio.Event()

    class Sink:
        def render(self, width):
            return ["sink"]

        def invalidate(self):
            pass

        async def handle_input(self, data):
            received.append(data)
            if data == "z":
                sentinel_seen.set()

    sink = Sink()
    tui.add_child(streamed)
    tui.add_child(sink)
    tui.set_focus(sink)

    render_errors: list[BaseException] = []

    async def on_render_error(error: BaseException) -> None:
        render_errors.append(error)

    tui.set_render_error_handler(on_render_error)

    stop_mutating = tonio.Event()
    frames_flowing = tonio.Event()
    frame_markers = [0]

    async def drain_output() -> None:
        # Keep the pty drained so the output pump can never wedge on a
        # full buffer, and count frames (synchronized-output opens) as
        # they arrive. Parked here at the end; the scope cancel unwinds
        # it (§4.5 contract).
        while True:
            data = await emulator.receive_some()
            if not data:
                return
            frame_markers[0] += data.count(b"\x1b[?2026h")
            if frame_markers[0] > 1:
                frames_flowing.set()

    async def mutate_loop() -> None:
        # The agent-listener shape: mutations posted to the owner from
        # another task, each followed by a render request.
        i = 0
        while not stop_mutating.is_set():
            i += 1

            async def apply(i=i) -> None:
                streamed.set_text(f"streamed content {i}")

            tui.input_owner.post(apply)
            tui.request_render()
            await stop_mutating.wait(0.005)

    try:
        await tui.start()
        async with tonio.scope() as scope:
            scope.spawn(drain_output())
            scope.spawn(mutate_loop())
            # Reply as a Kitty-capable terminal to settle negotiation.
            await emulator.send_all(b"\x1b[?7u")
            # Key bursts with escape sequences split across writes,
            # while mutations stream and frames go out.
            for _ in range(25):
                await emulator.send_all(b"abcd")
                await emulator.send_all(b"\x1b[1;5")  # split escape...
                await emulator.send_all(b"C")  # ...completed: ctrl+right
            await emulator.send_all(b"z")  # sentinel: input still alive?
            await sentinel_seen.wait(5.0)
            await frames_flowing.wait(5.0)
            stop_mutating.set()
            scope.cancel()

        assert sentinel_seen.is_set(), "input went silent under the storm"
        assert received.count("a") == 25, "keys were lost under the storm"
        assert render_errors == [], "rendering raised under the storm"
        assert frames_flowing.is_set(), "frames stopped flowing"
    finally:
        await tui.stop()
        set_kitty_protocol_active(False)
    emulator._fd.close()  # closes master
    os.close(slave)
