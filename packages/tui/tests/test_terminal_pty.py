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

from pidrei_tui.keys import set_kitty_protocol_active
from pidrei_tui.terminal import ProcessTerminal


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

    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    try:
        await terminal.start(inputs.append, lambda: None)

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
        terminal.write("out")
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

    terminal = ProcessTerminal(input_fd=slave, output_fd=slave)
    try:
        await terminal.start(inputs.append, lambda: None)
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
