"""Mirror of pi coding-agent test/auth-terminal.test.ts.

pi swaps `process.stdin` for a `PassThrough`; here `sys.stdin` is swapped for
a pipe the test opens and closes. pi's interrupt callback aborts the login's
controller; here it cancels the command's token.

PiDrei-only: the secret prompt's raw-mode key handling and the restore of the
terminal mode, over a pty (pi's suite does not exercise raw mode).
"""

import os
import sys
import termios
from typing import Self

import pytest
import tonio.colored as tonio

from pidrei.cli.auth_terminal import create_process_login_terminal, read_secret
from pidrei_utils.cancel import AbortError, CancelToken, combine_cancel_tokens


class SwappedStdin:
    """`sys.stdin` replaced by a file over `fd` while the test body runs."""

    def __init__(self, fd: int) -> None:
        self._file = os.fdopen(fd, "r", closefd=True)
        self._saved = None

    def __enter__(self) -> Self:
        self._saved = sys.stdin
        sys.stdin = self._file
        return self

    def __exit__(self, *_exc) -> None:
        sys.stdin = self._saved
        self._file.close()


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
    terminal = create_process_login_terminal(interrupt)
    with SwappedStdin(read_fd), pytest.raises(AbortError):
        await terminal.question("Enter domain: ", interrupt.controller)
    assert interrupt.count == 1
    assert interrupt.controller.cancelled


@pytest.mark.tonio
async def test_normal_cleanup_does_not_cancel_the_login_after_an_answer():
    read_fd, write_fd = os.pipe()
    interrupt = Interrupts()
    terminal = create_process_login_terminal(interrupt)
    try:
        with SwappedStdin(read_fd):
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
    terminal = create_process_login_terminal(interrupt)
    with SwappedStdin(read_fd):
        assert await terminal.question("Enter domain: ", interrupt.controller) == "example.com"
    assert interrupt.count == 0


@pytest.mark.tonio
async def test_prompt_cancellation_does_not_cancel_the_whole_login():
    read_fd, write_fd = os.pipe()
    interrupt = Interrupts()
    terminal = create_process_login_terminal(interrupt)
    prompt = CancelToken()
    combined = combine_cancel_tokens(interrupt.controller, prompt)
    try:
        with SwappedStdin(read_fd):
            answer = tonio.spawn(terminal.question("Paste code: ", combined.token))
            prompt.cancel()
            with pytest.raises(ExceptionGroup) as raised:
                await answer
            assert isinstance(raised.value.exceptions[0], AbortError)
            # Stdin is released by the time the question settles.
            assert os.get_blocking(read_fd)
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
        assert await read_secret("Key: ", interrupt.controller, interrupt) == "ac"
        assert termios.tcgetattr(pty.slave) == pty.baseline
    assert interrupt.count == 0


@pytest.mark.tonio
async def test_secret_ctrl_c_interrupts_the_login_and_restores_the_terminal_mode():
    interrupt = Interrupts()
    with PtyStdin() as pty:
        os.write(pty.master, b"x\x03")
        with pytest.raises(Exception, match="^Login cancelled$"):
            await read_secret("Key: ", interrupt.controller, interrupt)
        assert termios.tcgetattr(pty.slave) == pty.baseline
        assert os.get_blocking(pty.slave)
    assert interrupt.count == 1
