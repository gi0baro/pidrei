"""Mirror of pi tui test/terminal.test.ts.

pi's harness patches the `process.stdout.write`/`process.stdin.on` globals
and reaches into the terminal's privates; here the same seams are instance
attributes (`write_sync`, `_enqueue_input`, `_stdin_data_handler`).
pi drives the split-response timers with mocked clocks; here both (the
StdinBuffer sequence flush and the negotiation flush) are deadlines the
input reader expires, on `pidrei_tui.clock.monotonic`: `_ManualTimers.tick`
advances that clock and expires them, as pi's mocked timers fire.
"""

import os

import pytest

from pidrei_tui.keys import set_kitty_protocol_active
from pidrei_tui.program_status import ProgramStatus
from pidrei_tui.terminal import ProcessTerminal, normalize_apple_terminal_input, resolve_escape_timeout_ms
from pidrei_utils import clock as clock_module

from .tui_helpers import env_var


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


class _ManualTimers:
    """Stands in for `clock.monotonic` while installed: it holds still until
    `tick`, which then expires the reader's deadlines."""

    def __init__(self, terminal):
        self._terminal = terminal
        self.now = 0.0

    def monotonic(self):
        return self.now

    def tick(self, ms):
        self.now += ms / 1000
        self._terminal._expire_input_deadlines()


class _NegotiationHarness:
    def __init__(self):
        self.terminal = ProcessTerminal()
        self.writes = []
        self.input = None
        self.protocol_changes = []
        self._cleaned = False
        self.terminal.write_sync = self.writes.append
        # What the reader would queue for the input consumer.
        self.terminal._enqueue_input = self._on_input
        self.timers = _ManualTimers(self.terminal)
        self._monotonic = clock_module.monotonic
        clock_module.monotonic = self.timers.monotonic
        self.terminal._query_and_enable_kitty_protocol()

    def _on_input(self, data):
        # The Kitty activation is queued in input order, as an item of its own.
        if isinstance(data, str):
            self.input = data
        else:
            self.protocol_changes.append(data)

    async def send(self, data):
        self.terminal._stdin_data_handler(data)

    async def cleanup(self):
        if self._cleaned:
            return
        self._cleaned = True
        try:
            await self.terminal.stop()
        finally:
            clock_module.monotonic = self._monotonic
            set_kitty_protocol_active(False)


@pytest.mark.tonio
async def test_queries_kitty_mode_before_enabling_modify_other_keys_fallback():
    harness = _NegotiationHarness()
    try:
        assert harness.writes[0] == "\x1b[>7u\x1b[?u\x1b]7501;?\x1b\\\x1b[c"
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
async def test_forwards_device_attributes_replies_that_answer_other_queries():
    harness = _NegotiationHarness()
    try:
        await harness.send("\x1b[?7u")
        await harness.send("\x1b[?62;4;52c")
        assert harness.input is None

        # The TUI's color query uses DA1 as its own sentinel.
        await harness.send("\x1b[?62;4;52c")
        assert harness.input == "\x1b[?62;4;52c"
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
        harness.timers.tick(50)

        assert harness.input is None

        await harness.send("u")

        assert harness.terminal.kitty_protocol_active is True
        assert "\x1b[>4;2m" not in harness.writes
    finally:
        await harness.cleanup()


# Program status (OSC 7501), pi #10607

_WORKING = "\x1b]7501;state=working:app=pi\x1b\\"
_CLEAR = "\x1b]7501;state=clear\x1b\\"


@pytest.mark.tonio
async def test_reports_the_latest_program_status_once_the_terminal_answers_the_query_before_da(monkeypatch):
    monkeypatch.delenv("PIDREI_PROGRAM_STATUS", raising=False)
    harness = _NegotiationHarness()
    try:
        harness.terminal.set_program_status(ProgramStatus(state="working", app="pi"))
        assert _WORKING not in harness.writes

        await harness.send("\x1b]7501;?\x1b\\")
        assert harness.input is None
        assert harness.writes.count(_WORKING) == 1
        await harness.send("\x1b[?62;4;52c")

        harness.terminal.set_program_status(ProgramStatus(state="done"))
        assert harness.writes[-1] == "\x1b]7501;state=done\x1b\\"
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_reports_no_program_status_when_da_arrives_first_and_swallows_late_replies(monkeypatch):
    monkeypatch.delenv("PIDREI_PROGRAM_STATUS", raising=False)
    harness = _NegotiationHarness()
    try:
        await harness.send("\x1b[?62;4;52c")
        await harness.send("\x1b]7501;?\x07")
        harness.terminal.set_program_status(ProgramStatus(state="working", app="pi"))

        assert harness.input is None
        assert not any(write.startswith("\x1b]7501;state=") for write in harness.writes)
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_skips_the_program_status_query_when_pidrei_program_status_is_set(monkeypatch):
    monkeypatch.setenv("PIDREI_PROGRAM_STATUS", "1")
    harness = _NegotiationHarness()
    try:
        assert harness.writes[0] == "\x1b[>7u\x1b[?u\x1b[c"
        harness.terminal.set_program_status(ProgramStatus(state="working", app="pi"))
        assert harness.writes[-1] == _WORKING
    finally:
        await harness.cleanup()

    monkeypatch.setenv("PIDREI_PROGRAM_STATUS", "0")
    harness = _NegotiationHarness()
    try:
        assert harness.writes[0] == "\x1b[>7u\x1b[?u\x1b[c"
        harness.terminal.set_program_status(ProgramStatus(state="working", app="pi"))
        assert _WORKING not in harness.writes
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_does_not_let_a_da_reply_from_before_a_restart_end_the_new_program_status_query(monkeypatch):
    monkeypatch.delenv("PIDREI_PROGRAM_STATUS", raising=False)
    harness = _NegotiationHarness()
    try:
        harness.terminal.set_program_status(ProgramStatus(state="working", app="pi"))
        await harness.terminal.stop()
        harness.terminal._query_and_enable_kitty_protocol()

        # The first start's replies arrive late: its DA, then the second start's reply and DA.
        await harness.send("\x1b[?62;4;52c")
        await harness.send("\x1b]7501;?\x1b\\")
        await harness.send("\x1b[?62;4;52c")
        assert harness.writes[-1] == _WORKING
        assert harness.input is None
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_clears_the_program_status_on_stop_and_reports_it_again_after_restart(monkeypatch):
    monkeypatch.delenv("PIDREI_PROGRAM_STATUS", raising=False)
    harness = _NegotiationHarness()
    try:
        harness.terminal.set_program_status(ProgramStatus(state="working", app="pi"))
        await harness.send("\x1b]7501;?\x1b\\")
        await harness.terminal.stop()
        assert _CLEAR in harness.writes

        # Stopped: nothing is written until the restarted terminal confirms support again.
        writes_before_restart = len(harness.writes)
        harness.terminal.set_program_status(ProgramStatus(state="working", app="pi"))
        assert len(harness.writes) == writes_before_restart

        harness.terminal._query_and_enable_kitty_protocol()
        await harness.send("\x1b]7501;?\x1b\\")
        assert harness.writes[-1] == _WORKING
    finally:
        await harness.cleanup()


@pytest.mark.tonio
async def test_replays_buffered_csi_prefix_input_when_it_is_not_a_kitty_response():
    harness = _NegotiationHarness()
    try:
        await harness.send("\x1b[")
        harness.timers.tick(50)

        assert harness.input is None

        harness.timers.tick(150)

        assert harness.input == "\x1b["
    finally:
        await harness.cleanup()


# ProcessTerminal progress


def test_writes_a_valid_osc_9_4_clear_sequence():
    terminal = ProcessTerminal()
    writes: list[str] = []
    terminal.write_sync = writes.append

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
