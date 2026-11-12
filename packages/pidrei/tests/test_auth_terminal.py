"""Mirror of pi coding-agent test/auth-terminal.test.ts.

pi swaps `process.stdin` for a `PassThrough`; here the terminal reads a
`StdinReader` over a pipe the test opens and closes. pi's interrupt callback
aborts the login's controller; here it cancels the command's token.

PiDrei-only: the secret prompt's raw-mode key handling and the restore of the
terminal mode, over a pty (pi's suite does not exercise raw mode).
"""

import os
import termios
from typing import Self

import pytest
import tonio.colored as tonio

from pidrei.cli.auth_terminal import create_process_login_terminal, read_secret
from pidrei.utils.fd_io import StdinReader
from pidrei_utils.cancel import AbortError, CancelToken, combine_cancel_tokens


class SwappedStdin:
    """A `StdinReader` over `fd` while the test body runs; `fd` is made
    non-blocking, as stdin is from process entry on, and closed afterwards."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        os.set_blocking(fd, False)
        self.reader = StdinReader(fd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.reader.close()
        os.close(self.fd)


class Interrupts:
    def __init__(self) -> None:
        self.controller = CancelToken()
        self.count = 0

    def __call__(self) -> None:
        self.count += 1
        self.controller.cancel()


@pytest.mark.tonio
async def test_rejects_and_cancels_the_login_when_stdin_closes_before_an_answer():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    interrupt = Interrupts()
    with SwappedStdin(read_fd) as stdin, pytest.raises(AbortError):
        terminal = create_process_login_terminal(stdin.reader, interrupt)
        await terminal.question("Enter domain: ", interrupt.controller)
    assert interrupt.count == 1
    assert interrupt.controller.cancelled


@pytest.mark.tonio
async def test_normal_cleanup_does_not_cancel_the_login_after_an_answer():
    read_fd, write_fd = os.pipe()
    interrupt = Interrupts()
    try:
        with SwappedStdin(read_fd) as stdin:
            terminal = create_process_login_terminal(stdin.reader, interrupt)
            os.write(write_fd, b"example.com\n")
            assert await terminal.question("Enter domain: ", interrupt.controller) == "example.com"
    finally:
        os.close(write_fd)
    assert interrupt.count == 0
    assert not interrupt.controller.cancelled


@pytest.mark.tonio
async def test_keeps_an_answer_delivered_immediately_before_eof():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"example.com\n")
    os.close(write_fd)
    interrupt = Interrupts()
    with SwappedStdin(read_fd) as stdin:
        terminal = create_process_login_terminal(stdin.reader, interrupt)
        assert await terminal.question("Enter domain: ", interrupt.controller) == "example.com"
    assert interrupt.count == 0


@pytest.mark.tonio
async def test_prompt_cancellation_does_not_cancel_the_whole_login():
    read_fd, write_fd = os.pipe()
    interrupt = Interrupts()
    prompt = CancelToken()
    combined = combine_cancel_tokens(interrupt.controller, prompt)
    try:
        with SwappedStdin(read_fd) as stdin:
            terminal = create_process_login_terminal(stdin.reader, interrupt)
            answer = tonio.spawn(terminal.question("Paste code: ", combined.token))
            prompt.cancel()
            with pytest.raises(ExceptionGroup) as raised:
                await answer
            assert isinstance(raised.value.exceptions[0], AbortError)
            # Stdin is free for the next question by the time this one settles.
            os.write(write_fd, b"next\n")
            assert await terminal.question("Enter domain: ", interrupt.controller) == "next"
    finally:
        combined.cleanup()
        os.close(write_fd)
    assert interrupt.count == 0
    assert not interrupt.controller.cancelled


class PtyStdin(SwappedStdin):
    """A pty's slave as stdin, in a non-canonical mode of the test's own whose
    `ICRNL` the secret prompt turns off, so the restore is observable."""

    def __init__(self) -> None:
        self.master, slave = os.openpty()
        attributes = termios.tcgetattr(slave)
        attributes[0] |= termios.ICRNL
        attributes[3] &= ~(termios.ICANON | termios.ECHO | termios.ISIG | termios.IEXTEN)
        attributes[6][termios.VMIN] = 1
        attributes[6][termios.VTIME] = 0
        termios.tcsetattr(slave, termios.TCSANOW, attributes)
        self.slave = slave
        self.baseline = termios.tcgetattr(slave)
        super().__init__(slave)

    def __exit__(self, *exc) -> None:
        super().__exit__(*exc)
        os.close(self.master)


@pytest.mark.tonio
async def test_secret_reads_keys_in_raw_mode_and_restores_the_terminal_mode():
    interrupt = Interrupts()
    with PtyStdin() as pty:
        os.write(pty.master, b"ab\x7fc\r")
        assert await read_secret(pty.reader, "Key: ", interrupt.controller, interrupt) == "ac"
        assert termios.tcgetattr(pty.slave) == pty.baseline
    assert interrupt.count == 0


@pytest.mark.tonio
async def test_secret_ctrl_c_interrupts_the_login_and_restores_the_terminal_mode():
    interrupt = Interrupts()
    with PtyStdin() as pty:
        os.write(pty.master, b"x\x03")
        with pytest.raises(Exception, match="^Login cancelled$"):
            await read_secret(pty.reader, "Key: ", interrupt.controller, interrupt)
        assert termios.tcgetattr(pty.slave) == pty.baseline
    assert interrupt.count == 1
