"""pidrei-specific: ProcessTerminal end-to-end over a real pty.

No pi counterpart (node tests cannot re-point process.stdin at a pty); this
covers the port's tonio input pump, raw-mode handling, and live Kitty
negotiation, with the test playing the terminal-emulator side on the pty
master.
"""

import contextlib
import os
import pty
import termios
import threading
import time

import pytest
import tonio.colored as tonio
from tonio.colored.io import FdStream

from pidrei_tui import terminal as terminal_module
from pidrei_tui.components import Text
from pidrei_tui.keys import set_kitty_protocol_active
from pidrei_tui.terminal import ProcessTerminal
from pidrei_tui.tui_main_screen import TuiMainScreen


KITTY_QUERY = b"\x1b[>7u\x1b[?u\x1b[c"


async def _receive_until(stream: FdStream, *needles: bytes, timeout: float = 5.0) -> bytes:
    """Read the emulator side until every needle has arrived; return all of it.

    Replaces "sleep, then read whatever is there": pty and pipe reads come
    in whatever chunks the kernel hands over (line-sized on macOS), so the
    wait is for the bytes themselves, bounded so a miss fails the caller's
    assertion instead of hanging.
    """
    received = b""

    async def read() -> None:
        nonlocal received
        while not all(needle in received for needle in needles):
            chunk = await stream.receive_some()
            if not chunk:
                return
            received += chunk

    await tonio.time.timeout(read(), timeout)
    return received


class _InputLog:
    """The terminal's input handler for these tests: records each input and
    wakes `until` (the pump delivers on the terminal's input owner, another
    task) — the wait is for the input itself, not a sleep."""

    def __init__(self) -> None:
        self.items: list[str] = []
        self._lock = threading.Lock()
        self._waiters: list[tuple[int, tonio.Event]] = []

    async def record(self, data: str) -> None:
        with self._lock:
            self.items.append(data)
            count = len(self.items)
            waiters = list(self._waiters)
        for wanted, reached in waiters:
            if count >= wanted:
                reached.set()

    async def until(self, count: int, timeout: float = 5.0) -> list[str]:
        """Wait until `count` inputs have arrived; return the log."""
        reached = tonio.Event()
        with self._lock:
            if len(self.items) >= count:
                return list(self.items)
            self._waiters.append((count, reached))
        await reached.wait(timeout)
        with self._lock:
            self._waiters.remove((count, reached))
            return list(self.items)


class _ManualClock:
    """Stands in for terminal.py's `_time`: `monotonic()` holds still until
    `advance()`, and `read` is set on its first call."""

    def __init__(self) -> None:
        # Starts at the real reading, so stamps taken before the swap
        # (`_last_read_time`) stay comparable.
        self.now = time.monotonic()
        self.read = tonio.Event()

    def monotonic(self) -> float:
        self.read.set()
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@contextlib.contextmanager
def _manual_terminal_clock():
    clock = _ManualClock()
    original = terminal_module._time
    terminal_module._time = clock
    try:
        yield clock
    finally:
        terminal_module._time = original


@pytest.mark.tonio
async def test_pty_pump_negotiation_and_input_end_to_end():
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    # Reads go through tonio's fd stream (it owns `master` from here and
    # closes it at the end); inputs are plain `os.write`s.
    emulator = FdStream(master)
    inputs = _InputLog()

    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    try:
        await terminal.start(inputs.record, None)

        # Raw mode entered on the tty: echo off, canonical mode off.
        attrs = termios.tcgetattr(slave)
        assert attrs[3] & termios.ECHO == 0
        assert attrs[3] & termios.ICANON == 0
        # Output processing stays on (node's raw mode keeps it too).
        assert attrs[1] & termios.OPOST != 0

        # Startup wrote bracketed-paste enable and the Kitty query.
        startup = await _receive_until(emulator, b"\x1b[?2004h", KITTY_QUERY)
        assert b"\x1b[?2004h" in startup
        assert KITTY_QUERY in startup

        # Reply as a Kitty-capable terminal: protocol activates, nothing is
        # forwarded to the input handler. The pump handles input in order,
        # so once the keys typed after it have arrived the reply has been
        # handled: the checks follow the keys instead of a sleep.
        os.write(master, b"\x1b[?7u")

        # Type some keys, including a multi-byte codepoint split across
        # chunk boundaries mid-UTF-8.
        os.write(master, b"hi \xf0\x9f")
        os.write(master, b"\x8e\x89")
        assert await inputs.until(4) == ["h", "i", " ", "🎉"]
        assert terminal.kitty_protocol_active is True

        # Bracketed paste is re-wrapped for the editor. The trailing key
        # marks the end: nothing else arrived from the paste before it.
        os.write(master, b"\x1b[200~pasted\x1b[201~")
        os.write(master, b"z")
        assert (await inputs.until(6))[4:] == ["\x1b[200~pasted\x1b[201~", "z"]

        # Writes reach the pty unbuffered.
        await terminal.write("out")
        assert await _receive_until(emulator, b"out") == b"out"
    finally:
        await terminal.stop()
        set_kitty_protocol_active(False)

    # stop() restored the tty state and disabled what it enabled.
    attrs = termios.tcgetattr(slave)
    assert attrs[3] & termios.ECHO != 0
    assert attrs[3] & termios.ICANON != 0
    teardown = await _receive_until(emulator, b"\x1b[?2004l", b"\x1b[<u")
    assert b"\x1b[?2004l" in teardown
    assert b"\x1b[<u" in teardown

    emulator._fd.close()  # closes master
    os.close(slave)


@pytest.mark.tonio
async def test_pty_drain_input_returns_after_idle():
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    emulator = FdStream(master)  # owns `master` from here
    inputs = _InputLog()

    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    try:
        await terminal.start(inputs.record, None)
        await _receive_until(emulator, KITTY_QUERY)

        # Late input during drain is swallowed, and drain returns on idle
        # well before max_ms. The input used to be written *before* calling
        # drain_input, so the pump could forward it before drain detached
        # the handler; and "idle" was 50ms of wall clock. Here drain's clock
        # is manual: its first read (stamped right after it detaches the
        # handler) says the window is open, and nothing idles it out until
        # the test advances the clock.
        handle = terminal._stdin_data_handler
        delivered = tonio.Event()

        async def handle_then_signal(data: str) -> None:
            await handle(data)
            delivered.set()

        terminal._stdin_data_handler = handle_then_signal
        drained = tonio.Event()

        async def drain() -> None:
            await terminal.drain_input(2000, 50)
            drained.set()

        with _manual_terminal_clock() as clock:
            async with tonio.scope() as scope:
                scope.spawn(drain())
                await clock.read.wait(5)
                assert clock.read.is_set(), "drain_input never started"
                os.write(master, b"\x1b[97;1:3u")
                await delivered.wait(5)
                assert delivered.is_set(), "the late input never reached the terminal"
                assert inputs.items == []
                clock.advance(0.05)
                await drained.wait(5)
                assert drained.is_set(), "drain_input did not return on idle"
        assert inputs.items == []

        # The input handler is restored afterwards.
        os.write(master, b"x")
        assert await inputs.until(1) == ["x"]
    finally:
        await terminal.stop()
        set_kitty_protocol_active(False)

    emulator._fd.close()  # closes master
    os.close(slave)


@pytest.mark.tonio
async def test_pty_drain_input_cuts_between_chunks():
    # pidrei-only: in pi the handler change and input events share one thread, so
    # drain's cut falls between events. Here input is handled on the input owner;
    # the cut is an owner job, so a chunk being handled when drain is called
    # completes — "b", in the same chunk as the parked "a", is still delivered.
    master, slave = pty.openpty()
    os.set_blocking(master, False)
    emulator = FdStream(master)  # owns `master` from here
    items: list[str] = []
    parked = tonio.Event()
    release = tonio.Event()

    async def record(data: str) -> None:
        items.append(data)
        if data == "a":
            parked.set()
            await release.wait(5)

    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    try:
        await terminal.start(record, None)
        await _receive_until(emulator, KITTY_QUERY)

        # One chunk; its handling parks on "a".
        os.write(master, b"ab")
        await parked.wait(5)
        assert parked.is_set()

        # The pump is inside `run` for that chunk, so the next `run` is drain
        # asking the owner for its cut.
        owner_run = terminal.input_owner.run
        cut_requested = tonio.Event()

        async def observed_run(fn):
            cut_requested.set()
            await owner_run(fn)

        terminal.input_owner.run = observed_run
        drained = tonio.Event()

        async def drain() -> None:
            await terminal.drain_input(2000, 50)
            drained.set()

        with _manual_terminal_clock() as clock:
            async with tonio.scope() as scope:
                scope.spawn(drain())
                try:
                    await cut_requested.wait(5)
                    assert cut_requested.is_set(), "drain_input did not cut on the input owner"
                    # Queued behind the chunk being handled: nothing cut yet.
                    assert terminal._input_handler is record
                    release.set()
                    # Drain's first clock read follows its cut.
                    await clock.read.wait(5)
                    assert clock.read.is_set()
                    assert items == ["a", "b"]
                    assert terminal._input_handler is None
                    clock.advance(0.05)
                    await drained.wait(5)
                    assert drained.is_set()
                finally:
                    # On a failed check the scope still joins drain: let it end.
                    release.set()
                    clock.advance(10)
    finally:
        release.set()
        await terminal.stop()
        set_kitty_protocol_active(False)

    emulator._fd.close()  # closes master
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
    reader = FdStream(out_r)  # owns `out_r` from here
    terminal = ProcessTerminal(input_fd=in_r, output_fd=out_w)

    async def no_input(_data: str) -> None:
        pass

    try:
        await terminal.start(no_input, None)
        startup = await _receive_until(reader, KITTY_QUERY)
        assert startup.endswith(KITTY_QUERY)

        title = b"\x1b]0;before the frame\x07"
        frame = "x" * (1 << 20)  # far beyond the pipe's capacity
        total = len(title) + len(frame)
        # A pipe holds at most this much (64KiB on Linux, less on macOS):
        # while more than it is still unread, the frame cannot all be on
        # the wire, so write() must still be parked.
        pipe_capacity_bound = 1 << 18
        done = tonio.Event()

        async def write_frame() -> None:
            await terminal.write(frame)
            done.set()

        async with tonio.scope() as scope:
            terminal.set_title("before the frame")
            scope.spawn(write_frame())

            # The "still parked" check runs on every chunk while the backlog
            # exceeds the pipe, instead of once after a 50ms sleep that only
            # hoped an early return would have shown by then.
            received = b""
            while len(received) < total:
                if total - len(received) > pipe_capacity_bound:
                    assert not done.is_set(), "write() returned while the pipe was still full"
                chunk, completed = await tonio.time.timeout(reader.receive_some(), 5)
                assert completed and chunk, "the output pump stopped writing"
                received += chunk
            assert received.startswith(title)
            assert received[len(title) :] == frame.encode()
            await done.wait(5)
            assert done.is_set(), "write() must complete once its bytes are on the wire"
    finally:
        await terminal.stop()
        reader._fd.close()  # closes out_r
        for fd in (in_r, in_w, out_w):
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
        await terminal.start(record_input, None)

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
