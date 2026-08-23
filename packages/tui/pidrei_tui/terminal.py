"""Mirror of pi tui src/terminal.ts.

Port deviations (documented once here):

- pi drives input through node's `process.stdin` data events; here `start()`
  spawns a tonio input pump (`tonio.io.register` + `arm_r` over a
  non-blocking stdin fd, incremental UTF-8 decode) and a SIGWINCH resize
  watcher inside a tonio scope, so `start`/`stop` are async. The resize
  watcher needs the runtime created with `tonio.run(..., signals=
  [signal.SIGWINCH])`; without it resize events never fire (SIGWINCH's
  default disposition is to be ignored, so nothing breaks).
- Raw mode is entered with termios mirroring libuv's TTY_MODE_RAW (the mode
  behind node's `setRawMode(true)`): input flags BRKINT/ICRNL/INPCK/ISTRIP/
  IXON off, CS8 on, ECHO/ICANON/IEXTEN/ISIG off, VMIN=1/VTIME=0 — output
  processing (OPOST/ONLCR) stays on. `stop()` restores the saved attributes.
- pi re-raises SIGWINCH in `start()` because node caches terminal dimensions
  and misses resizes while stopped; `os.get_terminal_size()` queries the tty
  fresh on every read, so nothing to refresh.
- `columns`/`rows` read the output fd (test seam: `input_fd`/`output_fd`
  constructor arguments replace pi's tests patching the `process.stdin`/
  `process.stdout` globals).
- The win32 VT-input helper and the darwin native-modifiers addon are not
  ported (POSIX-only port; native modifier addons omitted per plan) — the
  Apple Terminal Shift+Enter rewrite therefore never sees a pressed Shift.
- Timer callbacks may fire on a different tonio worker thread than the input
  pump, so negotiation-buffer state transitions hold a re-entrant sync lock
  (never held across an await) and timer callbacks re-check identity under
  it (clearTimeout determinism).
- Output is a single pump task (`_output_pump`): every writer enqueues a
  complete sequence, the pump emits them in FIFO order with `arm_w`
  readiness. `write()` waits for its bytes to go out (backpressure for the
  render loop); the sync methods (`set_title`, `set_progress`, cursor and
  clear helpers, negotiation) only enqueue. pi writes `stdout` synchronously
  from its one thread, which gives it ordering and atomicity for free.
- Env rename: PI_TUI_WRITE_LOG → PIDREI_TUI_WRITE_LOG.
"""

import codecs
import math
import os
import re
import select
import signal as signal_module
import sys
import termios
import threading
import time as _time
from typing import Any, Protocol

import tonio.colored as tonio
from tonio.colored import io as tonio_io, signals as tonio_signals, sync as tonio_sync
from tonio.colored.sync import channel as tonio_channel

from ._timers import Interval, Timeout
from .keys import set_kitty_protocol_active
from .stdin_buffer import StdinBuffer


TERMINAL_PROGRESS_KEEPALIVE_MS = 1000
TERMINAL_PROGRESS_ACTIVE_SEQUENCE = "\x1b]9;4;3\x07"
TERMINAL_PROGRESS_CLEAR_SEQUENCE = "\x1b]9;4;0\x07"
APPLE_TERMINAL_SHIFT_ENTER_SEQUENCE = "\x1b[13;2u"
DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS = 7
KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_MS = 150
KITTY_KEYBOARD_PROTOCOL_QUERY = f"\x1b[>{DESIRED_KITTY_KEYBOARD_PROTOCOL_FLAGS}u\x1b[?u\x1b[c"

ENV_WRITE_LOG = "PIDREI_TUI_WRITE_LOG"

_KITTY_FLAGS_RE = re.compile(r"^\x1b\[\?(\d+)u$")
_DEVICE_ATTRIBUTES_RE = re.compile(r"^\x1b\[\?[\d;]*c$")
_NEGOTIATION_PREFIX_RE = re.compile(r"^\x1b\[\?[\d;]*$")

# KeyboardProtocolNegotiationSequence records:
#   {"type": "kitty-flags", "flags": int} | {"type": "device-attributes"}
KeyboardProtocolNegotiationSequence = dict[str, Any]


def parse_keyboard_protocol_negotiation_sequence(sequence: str) -> KeyboardProtocolNegotiationSequence | None:
    kitty_flags = _KITTY_FLAGS_RE.match(sequence)
    if kitty_flags:
        return {"type": "kitty-flags", "flags": int(kitty_flags.group(1))}
    if _DEVICE_ATTRIBUTES_RE.match(sequence):
        return {"type": "device-attributes"}
    return None


def _is_keyboard_protocol_negotiation_sequence_prefix(sequence: str) -> bool:
    return sequence == "\x1b[" or _NEGOTIATION_PREFIX_RE.match(sequence) is not None


def is_apple_terminal_session() -> bool:
    return sys.platform == "darwin" and os.environ.get("TERM_PROGRAM") == "Apple_Terminal"


def normalize_apple_terminal_input(data: str, is_apple_terminal: bool, is_shift_pressed: bool) -> str:
    if is_apple_terminal and data == "\r" and is_shift_pressed:
        return APPLE_TERMINAL_SHIFT_ENTER_SEQUENCE
    return data


def _is_native_modifier_pressed(key: str) -> bool:
    # pi loads a darwin-only native addon to read live modifier state; the
    # addon is not ported, so modifiers always read as released.
    return False


class Terminal(Protocol):
    """Minimal terminal interface for TUI."""

    async def start(self, on_input, on_resize) -> None:
        """Start the terminal with input and resize handlers."""
        ...

    async def stop(self) -> None:
        """Stop the terminal and restore state."""
        ...

    async def drain_input(self, max_ms: float = 1000, idle_ms: float = 50) -> None:
        """Drain stdin before exiting to prevent Kitty key release events from
        leaking to the parent shell over slow SSH connections."""
        ...

    async def write(self, data: str) -> None:
        """Write output to terminal."""
        ...

    @property
    def columns(self) -> int: ...

    @property
    def rows(self) -> int: ...

    @property
    def kitty_protocol_active(self) -> bool:
        """Whether Kitty keyboard protocol is active."""
        ...

    def move_by(self, lines: int) -> None:
        """Move cursor up (negative) or down (positive) by N lines."""
        ...

    def hide_cursor(self) -> None: ...

    def show_cursor(self) -> None: ...

    def clear_line(self) -> None:
        """Clear current line."""
        ...

    def clear_from_cursor(self) -> None:
        """Clear from cursor to end of screen."""
        ...

    def clear_screen(self) -> None:
        """Clear entire screen and move cursor to (0,0)."""
        ...

    def set_title(self, title: str) -> None:
        """Set terminal window title."""
        ...

    def set_progress(self, active: bool) -> None:
        """Progress indicator (OSC 9;4)."""
        ...


def _append_write_log(path: str, data: str) -> None:
    try:
        with open(path, "a", encoding="utf-8") as log:
            log.write(data)
    except OSError:
        pass  # Ignore logging errors, like pi


def _resolve_write_log_path() -> str:
    env = os.environ.get(ENV_WRITE_LOG) or ""
    if not env:
        return ""
    try:
        if os.path.isdir(env):
            ts = _time.strftime("%Y-%m-%d_%H-%M-%S")
            return os.path.join(env, f"tui-{ts}-{os.getpid()}.log")
    except OSError:
        # Not an existing directory - use as-is (file path)
        pass
    return env


DEFAULT_ESCAPE_TIMEOUT_MS = 10
DEFAULT_SSH_ESCAPE_TIMEOUT_MS = 100


def resolve_escape_timeout_ms(env: dict | None = None) -> float:
    """Resolve how long to wait for the rest of an escape sequence before
    dispatching a lone ESC as the Escape key. Legacy Alt+key input is ESC plus
    another byte, so high-latency transports need a longer reassembly window.
    """
    if env is None:
        env = os.environ
    try:
        configured = float(env.get("PIDREI_TUI_ESC_TIMEOUT") or "")
    except ValueError:
        configured = math.nan
    if math.isfinite(configured) and configured > 0:
        return configured
    if env.get("SSH_CONNECTION") or env.get("SSH_TTY"):
        return DEFAULT_SSH_ESCAPE_TIMEOUT_MS
    return DEFAULT_ESCAPE_TIMEOUT_MS


class ProcessTerminal:
    """Real terminal over the process stdin/stdout fds."""

    def __init__(self, *, input_fd: int = 0, output_fd: int = 1) -> None:
        self._input_fd = input_fd
        self._output_fd = output_fd
        self._lock = threading.RLock()
        # Output goes through one pump task (see `_output_pump`): writers
        # enqueue complete sequences and the pump emits them in FIFO order.
        self._out_tx = None
        self._out_rx = None
        self._out_sio = None
        self._out_scope = None
        self._saved_termios: list | None = None
        self._saved_blocking: bool | None = None
        self._saved_output_blocking: bool | None = None
        self._input_handler = None
        # Input reaches the handler from the stdin pump, the StdinBuffer flush
        # timer and the negotiation flush timer — three tasks. pi's handlers
        # (`editor.handle_input`, focus/overlay changes) never overlap on its
        # single thread; this lock restores that until an input-owner task
        # replaces the timers.
        self._input_dispatch_lock = tonio_sync.Lock()
        self._resize_handler = None
        self._kitty_protocol_active = False
        self._modify_other_keys_active = False
        self._keyboard_protocol_pushed = False
        self._negotiation_buffer = ""
        self._negotiation_flush_timer: Timeout | None = None
        self._stdin_buffer: StdinBuffer | None = None
        self._stdin_data_handler = None
        self._progress_interval: Interval | None = None
        self._write_log_path = _resolve_write_log_path()
        self._scope = None
        self._sio = None
        self._last_read_time = 0.0

    @property
    def kitty_protocol_active(self) -> bool:
        return self._kitty_protocol_active

    @property
    def modify_other_keys_active(self) -> bool:
        return self._modify_other_keys_active

    async def start(self, on_input, on_resize) -> None:
        self._input_handler = on_input
        self._resize_handler = on_resize

        # Save previous state and enable raw mode
        self._enter_raw_mode()

        # Output pump first, so every sequence from here on is queued behind
        # it in order. Its own scope: `stop()` drains it after the input side
        # is torn down, and the restore sequences must reach the fd before
        # termios is reset.
        self._saved_output_blocking = os.get_blocking(self._output_fd)
        os.set_blocking(self._output_fd, False)
        self._out_sio = tonio_io.register(self._output_fd)
        self._out_tx, self._out_rx = tonio_channel.unbounded()
        self._out_scope = tonio.scope()
        await self._out_scope.__aenter__()
        self._out_scope.spawn(self._output_pump())

        # Enable bracketed paste mode - terminal will wrap pastes in \x1b[200~ ... \x1b[201~
        self._write_stdout("\x1b[?2004h")

        # Set up the resize watcher and input pump; the pump plays the role of
        # node's process.stdin "data" listener.
        os.set_blocking(self._input_fd, False)
        # One registration per fd: a pty test (and a caller passing the same
        # fd for both sides) shares the output pump's.
        self._sio = self._out_sio if self._input_fd == self._output_fd else tonio_io.register(self._input_fd)
        self._scope = tonio.scope()
        await self._scope.__aenter__()
        self._scope.spawn(self._resize_watcher())

        # Query Kitty keyboard protocol and fall back to modifyOtherKeys when DA confirms no Kitty response.
        # See: https://sw.kovidgoyal.net/kitty/keyboard-protocol/
        self._query_and_enable_kitty_protocol()
        self._scope.spawn(self._input_pump())

    def _setup_stdin_buffer(self) -> None:
        """Set up StdinBuffer to split batched input into individual sequences.

        This ensures components receive single events, making matches_key/
        is_key_release work correctly.

        Also watches for the Kitty protocol response and enables it when
        detected. This is done here (after StdinBuffer parsing) rather than on
        raw stdin to handle the case where the response arrives split across
        multiple events.
        """
        self._stdin_buffer = StdinBuffer(escape_timeout=resolve_escape_timeout_ms())

        # Forward individual sequences to the input handler
        async def on_data(sequence: str) -> None:
            # `deferred` carries a buffered sequence that turned out not to be a
            # negotiation response; it must be forwarded before the current one,
            # but only after `_lock` is released.
            deferred: list[str] = []
            with self._lock:
                negotiation_sequence = self._read_keyboard_protocol_negotiation_sequence(sequence, deferred)
                if negotiation_sequence == "pending":
                    self._schedule_negotiation_buffer_flush()
                    consumed = True  # Wait briefly for the rest of a split Kitty response.
                else:
                    consumed = self._handle_keyboard_protocol_negotiation_sequence(negotiation_sequence)

            for buffered in deferred:
                await self._forward_input_sequence(buffered)
            if not consumed:
                await self._forward_input_sequence(sequence)

        self._stdin_buffer.on_data(on_data)

        # Re-wrap paste content with bracketed paste markers for existing editor handling
        async def on_paste(content: str) -> None:
            if self._input_handler is not None:
                async with self._input_dispatch_lock:
                    await self._input_handler(f"\x1b[200~{content}\x1b[201~")

        self._stdin_buffer.on_paste(on_paste)

        # Handler that pipes stdin data through the buffer
        self._stdin_data_handler = self._stdin_buffer.process

    def _query_and_enable_kitty_protocol(self) -> None:
        """Query terminal for Kitty keyboard protocol support and enable it if available.

        Kitty's progressive enhancement detection requires requesting the
        desired flags before querying them. The trailing DA query is a sentinel
        supported by terminals that do not know Kitty keyboard protocol;
        receiving DA before a Kitty response enables modifyOtherKeys fallback
        without a startup timeout.

        The requested flags are:
        - 1 = disambiguate escape codes
        - 2 = report event types (press/repeat/release)
        - 4 = report alternate keys (shifted key, base layout key)
        """
        self._setup_stdin_buffer()
        self._keyboard_protocol_pushed = True
        self._clear_negotiation_buffer()
        self._write_stdout(KITTY_KEYBOARD_PROTOCOL_QUERY)

    def _handle_keyboard_protocol_negotiation_sequence(
        self, negotiation_sequence: KeyboardProtocolNegotiationSequence | None
    ) -> bool:
        if not negotiation_sequence:
            return False
        self._clear_negotiation_buffer()
        if negotiation_sequence["type"] == "kitty-flags":
            if negotiation_sequence["flags"] != 0:
                self._disable_modify_other_keys()
                if not self._kitty_protocol_active:
                    self._kitty_protocol_active = True
                    set_kitty_protocol_active(True)
            else:
                self._enable_modify_other_keys()
            return True

        if not self._kitty_protocol_active:
            self._enable_modify_other_keys()
        return True

    def _read_keyboard_protocol_negotiation_sequence(
        self, sequence: str, deferred: list[str]
    ) -> KeyboardProtocolNegotiationSequence | str | None:
        """Runs under `_lock`. A buffered sequence that turns out not to be a
        negotiation response is appended to `deferred` for the caller to forward
        once the lock is released, rather than forwarded from here — forwarding
        is async now, and this runs inside the lock."""
        if self._negotiation_buffer:
            buffered_sequence = self._negotiation_buffer + sequence
            negotiation_sequence = parse_keyboard_protocol_negotiation_sequence(buffered_sequence)
            if negotiation_sequence:
                self._clear_negotiation_buffer()
                return negotiation_sequence
            if _is_keyboard_protocol_negotiation_sequence_prefix(buffered_sequence):
                self._set_negotiation_buffer(buffered_sequence)
                return "pending"
            buffered = self._take_negotiation_buffer()
            if buffered is not None:
                deferred.append(buffered)

        negotiation_sequence = parse_keyboard_protocol_negotiation_sequence(sequence)
        if negotiation_sequence:
            return negotiation_sequence
        if _is_keyboard_protocol_negotiation_sequence_prefix(sequence):
            self._set_negotiation_buffer(sequence)
            return "pending"
        return None

    def _set_negotiation_buffer(self, sequence: str) -> None:
        self._clear_negotiation_buffer_flush_timer()
        self._negotiation_buffer = sequence

    def _clear_negotiation_buffer(self) -> None:
        with self._lock:
            self._clear_negotiation_buffer_flush_timer()
            self._negotiation_buffer = ""

    def _take_negotiation_buffer(self) -> str | None:
        """Under `_lock`: claim the buffered sequence and reset the buffer."""
        if not self._negotiation_buffer:
            return None
        sequence = self._negotiation_buffer
        self._clear_negotiation_buffer_flush_timer()
        self._negotiation_buffer = ""
        return sequence

    def _schedule_negotiation_buffer_flush(self) -> None:
        if not self._negotiation_buffer or self._negotiation_flush_timer is not None:
            return
        timer: Timeout | None = None

        async def fire() -> None:
            with self._lock:
                # A clear/reschedule may have raced the firing callback past
                # its cancellation check; only the current timer may flush.
                if self._negotiation_flush_timer is not timer:
                    return
                self._negotiation_flush_timer = None
                sequence = self._take_negotiation_buffer()
            if sequence is not None:
                await self._forward_input_sequence(sequence)

        timer = Timeout(KEYBOARD_PROTOCOL_RESPONSE_FRAGMENT_TIMEOUT_MS, fire)
        self._negotiation_flush_timer = timer

    def _clear_negotiation_buffer_flush_timer(self) -> None:
        if self._negotiation_flush_timer is None:
            return
        self._negotiation_flush_timer.cancel()
        self._negotiation_flush_timer = None

    async def _forward_input_sequence(self, sequence: str) -> None:
        if self._input_handler is None:
            return
        is_apple_terminal = sequence == "\r" and is_apple_terminal_session()
        input_ = normalize_apple_terminal_input(
            sequence,
            is_apple_terminal,
            is_apple_terminal and _is_native_modifier_pressed("shift"),
        )
        async with self._input_dispatch_lock:
            await self._input_handler(input_)

    def _enable_modify_other_keys(self) -> None:
        if self._kitty_protocol_active or self._modify_other_keys_active:
            return
        self._write_stdout("\x1b[>4;2m")
        self._modify_other_keys_active = True

    def _disable_modify_other_keys(self) -> None:
        if not self._modify_other_keys_active:
            return
        self._write_stdout("\x1b[>4;0m")
        self._modify_other_keys_active = False

    async def _input_pump(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        sio = self._sio
        fd = self._input_fd
        while True:
            if (waiter := sio.arm_r()) is not None:
                await waiter
                continue
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                sio.consume_r()
                continue
            except InterruptedError:
                continue
            except OSError:
                return
            if not chunk:
                return
            self._last_read_time = _time.monotonic()
            data = decoder.decode(chunk)
            if data and (handler := self._stdin_data_handler) is not None:
                await handler(data)

    async def _resize_watcher(self) -> None:
        with tonio_signals.signal_receiver(signal_module.SIGWINCH) as receiver:
            async for _sig in receiver:
                handler = self._resize_handler
                if handler is None:
                    break  # stop() requested; exit through the receiver's cleanup
                handler()

    async def drain_input(self, max_ms: float = 1000, idle_ms: float = 50) -> None:
        should_disable_kitty_protocol = self._keyboard_protocol_pushed or self._kitty_protocol_active
        self._clear_negotiation_buffer()
        if should_disable_kitty_protocol:
            # Disable Kitty keyboard protocol first so any late key releases
            # do not generate new Kitty escape sequences.
            self._write_stdout("\x1b[<u")
            self._keyboard_protocol_pushed = False
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)
        self._disable_modify_other_keys()

        previous_handler = self._input_handler
        self._input_handler = None

        # The running input pump stamps _last_read_time on every read; pi
        # attaches a dedicated stdin listener for the same bookkeeping.
        last_data_time = _time.monotonic()
        end_time = _time.monotonic() + max_ms / 1000
        idle_s = idle_ms / 1000

        try:
            while True:
                now = _time.monotonic()
                time_left = end_time - now
                if time_left <= 0:
                    break
                if now - max(last_data_time, self._last_read_time) >= idle_s:
                    break
                await tonio.sleep(min(idle_s, time_left))
        finally:
            self._input_handler = previous_handler

    async def stop(self) -> None:
        if self._clear_progress_interval():
            self._write_stdout(TERMINAL_PROGRESS_CLEAR_SEQUENCE)

        # Disable bracketed paste mode
        self._write_stdout("\x1b[?2004l")

        should_disable_kitty_protocol = self._keyboard_protocol_pushed or self._kitty_protocol_active
        self._clear_negotiation_buffer()

        # Disable Kitty keyboard protocol if not already done by drain_input()
        if should_disable_kitty_protocol:
            self._write_stdout("\x1b[<u")
            self._keyboard_protocol_pushed = False
            self._kitty_protocol_active = False
            set_kitty_protocol_active(False)
        self._disable_modify_other_keys()

        # Clean up StdinBuffer
        if self._stdin_buffer is not None:
            self._stdin_buffer.destroy()
            self._stdin_buffer = None

        # Remove event handlers
        self._stdin_data_handler = None
        self._input_handler = None
        if self._resize_handler is not None:
            self._resize_handler = None
            # Nudge the resize watcher so it exits through the signal
            # receiver's cleanup instead of being aborted mid-park (best
            # effort: an abort would leak the SIGWINCH registration until the
            # process exits).
            os.kill(os.getpid(), signal_module.SIGWINCH)
            await tonio.yield_now()

        # Stop the pump before touching termios so buffered input (e.g.,
        # Ctrl+D) cannot be re-interpreted after raw mode is disabled (pi
        # pauses stdin for the same reason).
        if self._scope is not None:
            self._scope.cancel()
            await self._scope.__aexit__(None, None, None)
            self._scope = None
        if self._sio is not None:
            if self._sio is not self._out_sio:
                self._sio.close()
            self._sio = None
        if self._saved_blocking is not None:
            os.set_blocking(self._input_fd, self._saved_blocking)
            self._saved_blocking = None

        # Every restore sequence above is queued; wait for the pump to put
        # them on the wire, then retire it. Later writes (none expected) take
        # the direct path in `_write_stdout`.
        if self._out_tx is not None:
            await self.write("")
            self._out_tx.close()
            self._out_tx = None
        if self._out_scope is not None:
            await self._out_scope.__aexit__(None, None, None)
            self._out_scope = None
            self._out_rx = None
        if self._out_sio is not None:
            self._out_sio.close()
            self._out_sio = None
        if self._saved_output_blocking is not None:
            os.set_blocking(self._output_fd, self._saved_output_blocking)
            self._saved_output_blocking = None

        # Restore raw mode state
        self._restore_raw_mode()

    async def write(self, data: str) -> None:
        """Queue ``data`` and return once it is on the wire.

        The wait is what gives the render loop backpressure: a terminal that
        drains slowly (SSH) paces rendering instead of piling frames up in the
        queue. Writers that need ordering but not completion use the sync
        `_write_stdout`.
        """
        if self._out_tx is None:
            self._write_stdout(data)
            return
        done = tonio.Event()
        self._out_tx.send((data, done))
        await done.wait(None)

    @property
    def columns(self) -> int:
        return self._dimension("columns", "COLUMNS", 80)

    @property
    def rows(self) -> int:
        return self._dimension("lines", "LINES", 24)

    def _dimension(self, attr: str, env_name: str, fallback: int) -> int:
        try:
            value = getattr(os.get_terminal_size(self._output_fd), attr)
        except OSError, ValueError:
            value = 0
        if value:
            return value
        try:
            value = int(os.environ.get(env_name) or 0)
        except ValueError:
            value = 0
        return value or fallback

    def move_by(self, lines: int) -> None:
        if lines > 0:
            # Move down
            self._write_stdout(f"\x1b[{lines}B")
        elif lines < 0:
            # Move up
            self._write_stdout(f"\x1b[{-lines}A")
        # lines == 0: no movement

    def hide_cursor(self) -> None:
        self._write_stdout("\x1b[?25l")

    def show_cursor(self) -> None:
        self._write_stdout("\x1b[?25h")

    def clear_line(self) -> None:
        self._write_stdout("\x1b[K")

    def clear_from_cursor(self) -> None:
        self._write_stdout("\x1b[J")

    def clear_screen(self) -> None:
        self._write_stdout("\x1b[2J\x1b[H")  # Clear screen and move to home (1,1)

    def set_title(self, title: str) -> None:
        # OSC 0;title BEL - set terminal window title
        self._write_stdout(f"\x1b]0;{title}\x07")

    def set_progress(self, active: bool) -> None:
        with self._lock:
            if active:
                # OSC 9;4;3 - indeterminate progress
                self._write_stdout(TERMINAL_PROGRESS_ACTIVE_SEQUENCE)
                if self._progress_interval is None:
                    interval: Interval | None = None

                    async def fire() -> None:
                        with self._lock:
                            # set_progress(False) may have raced the firing
                            # callback past its cancellation check.
                            if self._progress_interval is not interval:
                                return
                            self._write_stdout(TERMINAL_PROGRESS_ACTIVE_SEQUENCE)

                    interval = Interval(TERMINAL_PROGRESS_KEEPALIVE_MS, fire)
                    self._progress_interval = interval
            else:
                self._clear_progress_interval()
                # OSC 9;4;0 - clear progress
                self._write_stdout(TERMINAL_PROGRESS_CLEAR_SEQUENCE)

    def _clear_progress_interval(self) -> bool:
        with self._lock:
            if self._progress_interval is None:
                return False
            self._progress_interval.cancel()
            self._progress_interval = None
            return True

    def _enter_raw_mode(self) -> None:
        fd = self._input_fd
        self._saved_blocking = os.get_blocking(fd)
        try:
            self._saved_termios = termios.tcgetattr(fd)
        except termios.error:
            self._saved_termios = None  # not a tty (pipes/pty tests)
            return
        attrs = termios.tcgetattr(fd)
        attrs[0] &= ~(termios.BRKINT | termios.ICRNL | termios.INPCK | termios.ISTRIP | termios.IXON)
        attrs[2] |= termios.CS8
        attrs[3] &= ~(termios.ECHO | termios.ICANON | termios.IEXTEN | termios.ISIG)
        attrs[6][termios.VMIN] = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)

    def _restore_raw_mode(self) -> None:
        if self._saved_termios is None:
            return
        try:
            termios.tcsetattr(self._input_fd, termios.TCSANOW, self._saved_termios)
        except termios.error:
            pass
        self._saved_termios = None

    def _write_stdout(self, data: str) -> None:
        """Queue ``data`` without waiting for it (sync; callable under `_lock`).

        Before `start()` / after `stop()` there is no pump and the fd is in
        its original blocking mode, so the bytes go straight out.
        """
        tx = self._out_tx
        if tx is not None:
            tx.send((data, None))
            return
        _write_all(self._output_fd, data.encode("utf-8"))

    async def _output_pump(self) -> None:
        """The only writer of the output fd while the terminal is started.

        Mirror of `_input_pump` on the write side: sequences are taken off the
        queue in order and written with `arm_w` readiness, so a full tty
        buffer parks this task instead of blocking a worker thread, and no
        two writers can ever interleave. pi has no counterpart — one JS thread
        and a synchronous `stdout.write` give it both properties for free.
        """
        rx = self._out_rx
        sio = self._out_sio
        fd = self._output_fd
        log_path = self._write_log_path
        broken = False
        while True:
            try:
                data, done = await rx.receive()
            except BrokenPipeError:
                return  # sender closed by stop(): queue drained
            buffer = memoryview(data.encode("utf-8"))
            while buffer and not broken:
                if (waiter := sio.arm_w()) is not None:
                    await waiter
                    continue
                try:
                    written = os.write(fd, buffer)
                except BlockingIOError:
                    sio.consume_w()
                    continue
                except InterruptedError:
                    continue
                except OSError:
                    # The terminal went away (EIO/EPIPE). Keep draining so no
                    # writer waits forever; the bytes have nowhere to go.
                    broken = True
                    break
                buffer = buffer[written:]
            if log_path and data:
                await tonio.spawn_blocking(_append_write_log, log_path, data)
            if done is not None:
                done.set()


def _write_all(fd: int, payload: bytes) -> None:
    """Direct write for the windows with no pump (before start / after stop)."""
    buffer = memoryview(payload)
    while buffer:
        try:
            written = os.write(fd, buffer)
        except BlockingIOError:
            # Only if the fd was inherited non-blocking; wait for room rather
            # than spin. Never reached while the pump is up.
            select.select([], [fd], [])
            continue
        except InterruptedError:
            continue
        buffer = buffer[written:]
