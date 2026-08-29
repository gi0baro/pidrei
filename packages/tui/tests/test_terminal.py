"""Mirror of pi tui test/terminal.test.ts.

pi's harness patches the `process.stdout.write`/`process.stdin.on` globals
and reaches into the terminal's privates; here the same seams are instance
attributes (`_write_stdout`, `_input_handler`, `_stdin_data_handler`).
pi drives the split-response timers with mocked clocks; the ticks become
short real-time sleeps.
"""

import os

import pytest
import tonio.colored as tonio

from pidrei_tui.keys import set_kitty_protocol_active
from pidrei_tui.terminal import ProcessTerminal, normalize_apple_terminal_input, resolve_escape_timeout_ms

from .tui_helpers import env_var


STDIN_FLUSH_WAIT = 0.1  # pi ticks 50ms for the StdinBuffer sequence flush timer
NEGOTIATION_FLUSH_WAIT = 0.25  # pi ticks 150ms for the negotiation flush timer


# resolve_escape_timeout_ms


def test_uses_pidrei_tui_esc_timeout_when_configured():
    assert resolve_escape_timeout_ms({"PIDREI_TUI_ESC_TIMEOUT": "80"}) == 80
    assert resolve_escape_timeout_ms({"PIDREI_TUI_ESC_TIMEOUT": "80", "SSH_TTY": "/dev/pts/1"}) == 80


def test_ignores_invalid_pidrei_tui_esc_timeout_values():
    assert resolve_escape_timeout_ms({"PIDREI_TUI_ESC_TIMEOUT": "abc"}) == 10
    assert resolve_escape_timeout_ms({"PIDREI_TUI_ESC_TIMEOUT": "0"}) == 10
    assert resolve_escape_timeout_ms({"PIDREI_TUI_ESC_TIMEOUT": "-5"}) == 10
    assert resolve_escape_timeout_ms({"PIDREI_TUI_ESC_TIMEOUT": ""}) == 10


def test_defaults_to_100ms_over_ssh():
    assert resolve_escape_timeout_ms({"SSH_CONNECTION": "10.0.0.1 22"}) == 100
    assert resolve_escape_timeout_ms({"SSH_TTY": "/dev/pts/1"}) == 100


def test_defaults_to_10ms_otherwise():
    assert resolve_escape_timeout_ms({}) == 10


# normalize_apple_terminal_input


def test_rewrites_apple_terminal_return_to_csi_u_shift_enter_when_shift_pressed():
    assert normalize_apple_terminal_input("\r", True, True) == "\x1b[13;2u"


def test_leaves_apple_terminal_return_unchanged_when_shift_not_pressed():
    assert normalize_apple_terminal_input("\r", True, False) == "\r"


def test_leaves_non_apple_terminal_return_unchanged_when_shift_pressed():
    assert normalize_apple_terminal_input("\r", False, True) == "\r"


def test_leaves_non_return_input_unchanged():
    assert normalize_apple_terminal_input("\x1b[13;2u", True, True) == "\x1b[13;2u"
    assert normalize_apple_terminal_input("a", True, True) == "a"


# ProcessTerminal Kitty keyboard protocol negotiation


class _NegotiationHarness:
    def __init__(self):
        self.terminal = ProcessTerminal()
        self.writes = []
        self.input = None
        self._cleaned = False
        self.terminal._write_stdout = self.writes.append
        self.terminal._input_handler = self._on_input
        self.terminal._query_and_enable_kitty_protocol()

    async def _on_input(self, data):
        # Terminal awaits its input handler now, so the double must be async too.
        self.input = data

    async def send(self, data):
        await self.terminal._stdin_data_handler(data)

    async def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        try:
            await self.terminal.stop()
        finally:
            set_kitty_protocol_active(False)


@pytest.mark.tonio
async def test_queries_kitty_mode_before_enabling_modify_other_keys_fallback():
    harness = _NegotiationHarness()
    try:
        assert harness.writes[0] == "\x1b[>7u\x1b[?u\x1b[c"
        assert "\x1b[>4;2m" not in harness.writes
        assert harness.terminal.kitty_protocol_active is False
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_activates_kitty_mode_for_non_zero_negotiated_flags():
    harness = _NegotiationHarness()
    try:
        await harness.send("\x1b[?7u")

        assert harness.input is None
        assert harness.terminal.kitty_protocol_active is True
        assert "\x1b[>4;2m" not in harness.writes
        assert "\x1b[>4;0m" not in harness.writes

        await harness.cleanup()
        assert harness.writes.count("\x1b[<u") == 1
        assert "\x1b[>4;0m" not in harness.writes
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_falls_back_to_modify_other_keys_for_zero_kitty_flags():
    harness = _NegotiationHarness()
    try:
        await harness.send("\x1b[?0u")

        assert harness.input is None
        assert harness.terminal.kitty_protocol_active is False
        assert harness.writes.count("\x1b[>4;2m") == 1

        await harness.cleanup()
        assert harness.writes.count("\x1b[>4;0m") == 1
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_falls_back_to_modify_other_keys_for_device_attributes_without_kitty_flags():
    harness = _NegotiationHarness()
    try:
        await harness.send("\x1b[?62;4;52c")

        assert harness.input is None
        assert harness.terminal.kitty_protocol_active is False
        assert harness.writes.count("\x1b[>4;2m") == 1
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_forwards_normal_input_while_waiting_for_kitty_response():
    harness = _NegotiationHarness()
    try:
        await harness.send("a")

        assert harness.input == "a"
        assert harness.terminal.kitty_protocol_active is False
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_tracks_split_kitty_confirmation():
    harness = _NegotiationHarness()
    try:
        await harness.send("\x1b[?7")
        await tonio.sleep(STDIN_FLUSH_WAIT)

        assert harness.input is None

        await harness.send("u")

        assert harness.terminal.kitty_protocol_active is True
        assert "\x1b[>4;2m" not in harness.writes
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_replays_buffered_csi_prefix_input_when_it_is_not_a_kitty_response():
    harness = _NegotiationHarness()
    try:
        await harness.send("\x1b[")
        await tonio.sleep(STDIN_FLUSH_WAIT)

        assert harness.input is None

        await tonio.sleep(NEGOTIATION_FLUSH_WAIT)

        assert harness.input == "\x1b["
    finally:
        await harness.cleanup()


# ProcessTerminal progress


def test_writes_a_valid_osc_9_4_clear_sequence():
    terminal = ProcessTerminal()
    writes: list[str] = []
    terminal._write_stdout = writes.append

    terminal.set_progress(False)

    assert writes == ["\x1b]9;4;0\x07"]


# ProcessTerminal dimensions


@pytest.mark.tonio
async def test_falls_back_to_columns_and_lines_env_before_default_dimensions(tmp_path):
    # pi undefines process.stdout.columns/rows; a non-tty output fd is the
    # same seam here (os.get_terminal_size fails).
    fd = os.open(str(tmp_path / "not-a-tty"), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        with env_var("COLUMNS", "123"), env_var("LINES", "45"):
            terminal = ProcessTerminal(output_fd=fd)

            assert terminal.columns == 123
            assert terminal.rows == 45
    finally:
        os.close(fd)
