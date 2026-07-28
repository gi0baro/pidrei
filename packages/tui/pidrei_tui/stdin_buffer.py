"""Mirror of pi tui src/stdin-buffer.ts.

StdinBuffer buffers input and emits complete sequences.

This is necessary because stdin data events can arrive in partial chunks,
especially for escape sequences like mouse events. Without buffering,
partial sequences can be misinterpreted as regular keypresses.

For example, the mouse SGR sequence ``\\x1b[<35;20;5m`` might arrive as:

- Event 1: ``\\x1b``
- Event 2: ``[<35``
- Event 3: ``;20;5m``

The buffer accumulates these until a complete sequence is detected.
Call the ``process()`` method to feed input data.

pi's upstream note: based on code from OpenTUI
(https://github.com/anomalyco/opentui), MIT License, Copyright (c) 2025
opentui.

Port deviations (pi is single-threaded JS):

- Events: pi extends EventEmitter with "data"/"paste"; here listeners are
  registered via ``on_data``/``on_paste`` returning unsubscribe callables.
- The flush timeout is a tonio task; under tonio's multi-threaded runtime it
  may fire on a different worker thread than the ``process()`` caller, so all
  state transitions hold a re-entrant sync lock (never held across an await)
  and the timer callback re-checks its own identity under that lock to get
  clearTimeout's determinism back.
- Non-escape input is split per Unicode codepoint, not per UTF-16 unit, so
  astral-plane characters are never cut into surrogate halves.
"""

import re
import threading

from ._timers import Timeout


ESC = "\x1b"
BRACKETED_PASTE_START = "\x1b[200~"
BRACKETED_PASTE_END = "\x1b[201~"

_SGR_MOUSE_RE = re.compile(r"^<\d+;\d+;\d+[Mm]$")
_DIGITS_RE = re.compile(r"^\d+$")
_KITTY_PRINTABLE_RE = re.compile(r"^\x1b\[(\d+)(?::\d*)?(?::\d+)?u$")


def _is_complete_sequence(data: str) -> str:
    """Check if a string is a complete escape sequence or needs more data."""
    if not data.startswith(ESC):
        return "not-escape"

    if len(data) == 1:
        return "incomplete"

    after_esc = data[1:]

    # CSI sequences: ESC [
    if after_esc.startswith("["):
        # Check for old-style mouse sequence: ESC[M + 3 bytes
        if after_esc.startswith("[M"):
            # Old-style mouse needs ESC[M + 3 bytes = 6 total
            return "complete" if len(data) >= 6 else "incomplete"
        return _is_complete_csi_sequence(data)

    # OSC sequences: ESC ]
    if after_esc.startswith("]"):
        return _is_complete_osc_sequence(data)

    # DCS sequences: ESC P ... ESC \ (includes XTVersion responses)
    if after_esc.startswith("P"):
        return _is_complete_dcs_sequence(data)

    # APC sequences: ESC _ ... ESC \ (includes Kitty graphics responses)
    if after_esc.startswith("_"):
        return _is_complete_apc_sequence(data)

    # SS3 sequences: ESC O
    if after_esc.startswith("O"):
        # ESC O followed by a single character
        return "complete" if len(after_esc) >= 2 else "incomplete"

    # Meta key sequences: ESC followed by a single character
    if len(after_esc) == 1:
        return "complete"

    # Unknown escape sequence - treat as complete
    return "complete"


def _is_complete_csi_sequence(data: str) -> str:
    """CSI sequences: ESC [ ... followed by a final byte (0x40-0x7E)."""
    if not data.startswith(f"{ESC}["):
        return "complete"

    # Need at least ESC [ and one more character
    if len(data) < 3:
        return "incomplete"

    payload = data[2:]

    # CSI sequences end with a byte in the range 0x40-0x7E (@-~)
    # This includes all letters and several special characters
    last_char = payload[-1]
    last_char_code = ord(last_char)

    if 0x40 <= last_char_code <= 0x7E:
        # Special handling for SGR mouse sequences
        # Format: ESC[<B;X;Ym or ESC[<B;X;YM
        if payload.startswith("<"):
            # Must have format: <digits;digits;digits[Mm]
            if _SGR_MOUSE_RE.match(payload):
                return "complete"
            # If it ends with M or m but doesn't match the pattern, still incomplete
            if last_char in ("M", "m"):
                # Check if we have the right structure
                parts = payload[1:-1].split(";")
                if len(parts) == 3 and all(_DIGITS_RE.match(part) for part in parts):
                    return "complete"

            return "incomplete"

        return "complete"

    return "incomplete"


def _is_complete_osc_sequence(data: str) -> str:
    """OSC sequences: ESC ] ... ST (where ST is ESC \\ or BEL)."""
    if not data.startswith(f"{ESC}]"):
        return "complete"

    # OSC sequences end with ST (ESC \) or BEL (\x07)
    if data.endswith((f"{ESC}\\", "\x07")):
        return "complete"

    return "incomplete"


def _is_complete_dcs_sequence(data: str) -> str:
    """DCS sequences: ESC P ... ST (used for XTVersion responses)."""
    if not data.startswith(f"{ESC}P"):
        return "complete"

    # DCS sequences end with ST (ESC \)
    if data.endswith(f"{ESC}\\"):
        return "complete"

    return "incomplete"


def _is_complete_apc_sequence(data: str) -> str:
    """APC sequences: ESC _ ... ST (used for Kitty graphics responses)."""
    if not data.startswith(f"{ESC}_"):
        return "complete"

    # APC sequences end with ST (ESC \)
    if data.endswith(f"{ESC}\\"):
        return "complete"

    return "incomplete"


def _parse_unmodified_kitty_printable_codepoint(sequence: str) -> int | None:
    match = _KITTY_PRINTABLE_RE.match(sequence)
    if not match:
        return None

    codepoint = int(match.group(1))
    return codepoint if codepoint >= 32 else None


def _extract_complete_sequences(buffer: str) -> tuple[list[str], str]:
    """Split accumulated buffer into complete sequences and a remainder."""
    sequences: list[str] = []
    pos = 0

    while pos < len(buffer):
        remaining = buffer[pos:]

        # Try to extract a sequence starting at this position
        if remaining.startswith(ESC):
            # Find the end of this escape sequence
            seq_end = 1
            while seq_end <= len(remaining):
                candidate = remaining[:seq_end]
                status = _is_complete_sequence(candidate)

                if status == "complete":
                    # WezTerm with enable_kitty_keyboard sends the Escape key press as a
                    # raw '\x1b' byte (simple text path in encode_kitty, ignoring
                    # DISAMBIGUATE_ESCAPE_CODES) and the release as a full Kitty CSI-u
                    # sequence. These arrive concatenated as '\x1b\x1b[27;...u'.
                    # The buffer would normally treat '\x1b\x1b' as a complete meta-key
                    # sequence (ESC + single char), leaving '[27;...u' to be typed as
                    # plain text. If the character immediately following '\x1b\x1b'
                    # would begin a new escape sequence, emit only the first ESC and
                    # restart from the second.
                    if candidate == "\x1b\x1b":
                        next_char = remaining[seq_end : seq_end + 1]
                        if next_char in ("[", "]", "O", "P", "_"):  # CSI/OSC/SS3/DCS/APC
                            sequences.append(ESC)
                            pos += 1
                            break
                    sequences.append(candidate)
                    pos += seq_end
                    break
                if status == "incomplete":
                    seq_end += 1
                else:
                    # Should not happen when starting with ESC
                    sequences.append(candidate)
                    pos += seq_end
                    break

            if seq_end > len(remaining):
                return sequences, remaining
        else:
            # Not an escape sequence - take a single character
            sequences.append(remaining[0])
            pos += 1

    return sequences, ""


class StdinBuffer:
    """Buffers stdin input and emits complete sequences via data listeners.

    Handles partial escape sequences that arrive across multiple chunks.
    ``timeout`` is the maximum time in milliseconds to wait for sequence
    completion (default 10); after that the buffer is flushed even if
    incomplete.
    """

    def __init__(self, timeout: float = 10) -> None:
        self._timeout_ms = timeout
        self._lock = threading.RLock()
        self._buffer = ""
        self._timeout: Timeout | None = None
        self._paste_mode = False
        self._paste_buffer = ""
        self._pending_kitty_printable_codepoint: int | None = None
        self._data_listeners: list = []
        self._paste_listeners: list = []

    def on_data(self, listener):
        self._data_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._data_listeners:
                self._data_listeners.remove(listener)

        return unsubscribe

    def on_paste(self, listener):
        self._paste_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._paste_listeners:
                self._paste_listeners.remove(listener)

        return unsubscribe

    async def process(self, data: str | bytes | bytearray) -> None:
        """Parse `data` and deliver whatever it completes to the listeners.

        The parse runs under `_lock`; the delivery does not. Listeners are the
        head of the input chain and are now async (they end up persisting things
        like label edits), so awaiting them under a threading lock would be the
        very hazard this codebase forbids. `_process` therefore *collects*
        emissions instead of firing them, and the lock is released before any of
        them is delivered. Ordering across the recursive paste path is preserved
        because every branch appends to the same list.
        """
        with self._lock:
            emissions: list[tuple[str, str]] = []
            self._process(data, emissions)
        await self._dispatch(emissions)

    async def _dispatch(self, emissions: list[tuple[str, str]]) -> None:
        for kind, payload in emissions:
            listeners = self._data_listeners if kind == "data" else self._paste_listeners
            for listener in list(listeners):
                # Listeners are awaitable-returning (async-only policy): they
                # re-enter the async input chain.
                await listener(payload)

    def _process(self, data: str | bytes | bytearray, out: list[tuple[str, str]]) -> None:
        # Clear any pending timeout
        if self._timeout is not None:
            self._timeout.cancel()
            self._timeout = None

        # Handle high-byte conversion (for compatibility with parseKeypress)
        # If buffer has single byte > 127, convert to ESC + (byte - 128)
        if isinstance(data, (bytes, bytearray)):
            if len(data) == 1 and data[0] > 127:
                string = ESC + chr(data[0] - 128)
            else:
                string = bytes(data).decode("utf-8", "replace")
        else:
            string = data

        if len(string) == 0 and len(self._buffer) == 0:
            self._emit_data_sequence("", out)
            return

        self._buffer += string

        if self._paste_mode:
            self._paste_buffer += self._buffer
            self._buffer = ""

            end_index = self._paste_buffer.find(BRACKETED_PASTE_END)
            if end_index != -1:
                pasted_content = self._paste_buffer[:end_index]
                remaining = self._paste_buffer[end_index + len(BRACKETED_PASTE_END) :]

                self._paste_mode = False
                self._paste_buffer = ""
                self._pending_kitty_printable_codepoint = None

                out.append(("paste", pasted_content))

                if len(remaining) > 0:
                    self._process(remaining, out)
            return

        start_index = self._buffer.find(BRACKETED_PASTE_START)
        if start_index != -1:
            if start_index > 0:
                before_paste = self._buffer[:start_index]
                # pi drops any incomplete remainder before the paste marker
                sequences, _remainder = _extract_complete_sequences(before_paste)
                for sequence in sequences:
                    self._emit_data_sequence(sequence, out)

            self._pending_kitty_printable_codepoint = None
            self._buffer = self._buffer[start_index + len(BRACKETED_PASTE_START) :]
            self._paste_mode = True
            self._paste_buffer = self._buffer
            self._buffer = ""

            end_index = self._paste_buffer.find(BRACKETED_PASTE_END)
            if end_index != -1:
                pasted_content = self._paste_buffer[:end_index]
                remaining = self._paste_buffer[end_index + len(BRACKETED_PASTE_END) :]

                self._paste_mode = False
                self._paste_buffer = ""
                self._pending_kitty_printable_codepoint = None

                out.append(("paste", pasted_content))

                if len(remaining) > 0:
                    self._process(remaining, out)
            return

        sequences, remainder = _extract_complete_sequences(self._buffer)
        self._buffer = remainder

        for sequence in sequences:
            self._emit_data_sequence(sequence, out)

        if len(self._buffer) > 0:
            self._schedule_flush_timer()

    def _schedule_flush_timer(self) -> None:
        timer: Timeout | None = None

        async def fire() -> None:
            with self._lock:
                # A process()/flush()/clear() call may have raced the firing
                # callback past its cancellation check; only the current timer
                # is allowed to flush.
                if self._timeout is not timer:
                    return
                self._timeout = None
                emissions: list[tuple[str, str]] = []
                for sequence in self._flush():
                    self._emit_data_sequence(sequence, emissions)
            # Same rule as `process`: collected under the lock, delivered after.
            await self._dispatch(emissions)

        timer = Timeout(self._timeout_ms, fire)
        self._timeout = timer

    def _emit_data_sequence(self, sequence: str, out: list[tuple[str, str]]) -> None:
        """Under `_lock`: apply the Kitty codepoint de-duplication and queue the
        sequence for delivery. Does not touch listeners."""
        raw_codepoint = ord(sequence) if len(sequence) == 1 else None
        if raw_codepoint is not None and raw_codepoint == self._pending_kitty_printable_codepoint:
            self._pending_kitty_printable_codepoint = None
            return

        self._pending_kitty_printable_codepoint = _parse_unmodified_kitty_printable_codepoint(sequence)
        out.append(("data", sequence))

    def flush(self) -> list[str]:
        with self._lock:
            return self._flush()

    def _flush(self) -> list[str]:
        if self._timeout is not None:
            self._timeout.cancel()
            self._timeout = None

        if len(self._buffer) == 0:
            return []

        sequences = [self._buffer]
        self._buffer = ""
        self._pending_kitty_printable_codepoint = None
        return sequences

    def clear(self) -> None:
        with self._lock:
            if self._timeout is not None:
                self._timeout.cancel()
                self._timeout = None
            self._buffer = ""
            self._paste_mode = False
            self._paste_buffer = ""
            self._pending_kitty_printable_codepoint = None

    def get_buffer(self) -> str:
        with self._lock:
            return self._buffer

    def destroy(self) -> None:
        self.clear()
