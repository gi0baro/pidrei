"""Mirror of pi coding-agent test/terminal-question.test.ts.

pi swaps `process.stdin` for a `PassThrough`; here `sys.stdin` is swapped for
a pipe the test opens and closes, and pi's listener and pause checks become
the descriptor's blocking flag, which the reader puts back when it is done.

Not mirrored: "Ctrl+C rejects the question and reports an interrupt". It
drives Node readline's raw-mode key handling; PiDrei reads the line in the
terminal's canonical mode, where Ctrl+C is SIGINT for the process.
"""

import os
import sys
from typing import Self

import pytest
import tonio.colored as tonio

from pidrei.utils.terminal_question import question_terminal
from pidrei_utils.cancel import AbortError, CancelToken


class PipeStdin:
    """A pipe standing in for stdin while the test body runs."""

    def __init__(self) -> None:
        self.read_fd, self.write_fd = os.pipe()
        self._file = os.fdopen(self.read_fd, "r", closefd=True)
        self._saved = None

    def __enter__(self) -> Self:
        self._saved = sys.stdin
        sys.stdin = self._file
        return self

    def __exit__(self, *_exc) -> None:
        sys.stdin = self._saved
        self.close_input()
        self._file.close()

    def write(self, data: bytes) -> None:
        os.write(self.write_fd, data)

    def close_input(self) -> None:
        if self.write_fd != -1:
            os.close(self.write_fd)
            self.write_fd = -1


class Recorder:
    def __init__(self) -> None:
        self.output: list[str] = []
        self.written = tonio.Event()
        self.interrupts = 0

    def write(self, text: str) -> None:
        self.output.append(text)
        self.written.set()

    def on_interrupt(self) -> None:
        self.interrupts += 1


@pytest.mark.tonio
async def test_writes_to_the_selected_output_and_cleans_up_after_an_answer():
    recorder = Recorder()
    with PipeStdin() as stdin:
        stdin.write(b"https://example.test/callback\n")
        answer = await question_terminal(
            "Paste URL: ", output=recorder.write, cancel=CancelToken(), on_interrupt=recorder.on_interrupt
        )
        assert answer == "https://example.test/callback"
        assert recorder.output == ["Paste URL: "]
        assert recorder.interrupts == 0
        assert os.get_blocking(stdin.read_fd)


@pytest.mark.tonio
async def test_rejects_on_eof_without_requiring_an_interrupt_callback():
    recorder = Recorder()
    with PipeStdin() as stdin:
        stdin.close_input()
        with pytest.raises(AbortError):
            await question_terminal("Paste URL: ", output=recorder.write, cancel=CancelToken())
        assert os.get_blocking(stdin.read_fd)


@pytest.mark.tonio
async def test_caller_cancellation_does_not_report_an_interrupt():
    recorder = Recorder()
    cancel = CancelToken()

    async def cancel_once_asked() -> None:
        await recorder.written.wait(5)
        cancel.cancel()

    with PipeStdin() as stdin:
        canceller = tonio.spawn(cancel_once_asked())
        with pytest.raises(AbortError):
            await question_terminal(
                "Paste URL: ", output=recorder.write, cancel=cancel, on_interrupt=recorder.on_interrupt
            )
        await canceller
        assert recorder.interrupts == 0
        assert os.get_blocking(stdin.read_fd)


@pytest.mark.tonio
async def test_rejects_an_already_aborted_prompt_and_cleans_up():
    recorder = Recorder()
    cancel = CancelToken()
    cancel.cancel()
    with PipeStdin() as stdin:
        with pytest.raises(AbortError):
            await question_terminal(
                "Paste URL: ", output=recorder.write, cancel=cancel, on_interrupt=recorder.on_interrupt
            )
        assert recorder.interrupts == 0
        assert recorder.output == []
        assert os.get_blocking(stdin.read_fd)
