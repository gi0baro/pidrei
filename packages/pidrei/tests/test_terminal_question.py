"""Mirror of pi coding-agent test/terminal-question.test.ts.

pi swaps `process.stdin` for a `PassThrough`; here the question reads a
`StdinReader` over a pipe the test opens and closes. pi's listener and pause
checks have no counterpart: the reader is the run's, and a question leaves
nothing behind but what it did not take.

Not mirrored: "Ctrl+C rejects the question and reports an interrupt". It
drives Node readline's raw-mode key handling; PiDrei reads the line in the
terminal's canonical mode, where Ctrl+C is SIGINT for the process.

PiDrei-only: what the run's one reader keeps between questions, and its
one-question-at-a-time guard.
"""

import os
from typing import Self

import pytest
import tonio.colored as tonio
from tonio.colored.exceptions import WouldBlock

from pidrei.utils.fd_io import StdinReader
from pidrei.utils.terminal_question import question_terminal
from pidrei_utils.cancel import AbortError, CancelToken


class PipeStdin:
    """A pipe standing in for stdin while the test body runs, read by a
    `StdinReader`; its read end is non-blocking, as stdin is from process
    entry on."""

    def __init__(self) -> None:
        self.read_fd, self.write_fd = os.pipe()
        os.set_blocking(self.read_fd, False)
        self.reader = StdinReader(self.read_fd)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close_input()
        self.reader.close()
        os.close(self.read_fd)

    def write(self, data: bytes) -> None:
        os.write(self.write_fd, data)

    def close_input(self) -> None:
        if self.write_fd != -1:
            os.close(self.write_fd)
            self.write_fd = -1


class WaitedToken(CancelToken):
    """A token that reports each time a read starts waiting on it: the reader
    then holds its guard, has nothing buffered to answer with, and is parked."""

    __slots__ = ("_waits", "waiting")

    def __init__(self, waits: int = 1) -> None:
        super().__init__()
        self._waits = 0
        # `waiting[n]` is set by the (n+1)th wait.
        self.waiting = [tonio.Event() for _ in range(waits)]

    def wait(self, timeout: float | None = None):
        if self._waits < len(self.waiting):
            self.waiting[self._waits].set()
        self._waits += 1
        return super().wait(timeout)


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
            "Paste URL: ",
            stdin=stdin.reader,
            output=recorder.write,
            cancel=CancelToken(),
            on_interrupt=recorder.on_interrupt,
        )
        assert answer == "https://example.test/callback"
        assert recorder.output == ["Paste URL: "]
        assert recorder.interrupts == 0


@pytest.mark.tonio
async def test_rejects_on_eof_without_requiring_an_interrupt_callback():
    recorder = Recorder()
    with PipeStdin() as stdin:
        stdin.close_input()
        with pytest.raises(AbortError):
            await question_terminal("Paste URL: ", stdin=stdin.reader, output=recorder.write, cancel=CancelToken())


@pytest.mark.tonio
async def test_caller_cancellation_does_not_report_an_interrupt():
    recorder = Recorder()
    cancel = CancelToken()

    async def cancel_once_asked() -> None:
        await recorder.written.wait(5)
        cancel.cancel()

    settled = tonio.Event()
    outcome: list[BaseException] = []

    async def ask() -> None:
        try:
            await question_terminal(
                "Paste URL: ",
                stdin=stdin.reader,
                output=recorder.write,
                cancel=cancel,
                on_interrupt=recorder.on_interrupt,
            )
        except Exception as error:
            outcome.append(error)
        finally:
            settled.set()

    with PipeStdin() as stdin:
        try:
            canceller = tonio.spawn(cancel_once_asked())
            asking = tonio.spawn(ask())
            await settled.wait(5)
            assert settled.is_set(), "the cancelled question kept waiting for input"
            assert len(outcome) == 1 and isinstance(outcome[0], AbortError)
            await asking
            await canceller
        finally:
            # EOF ends a question still parked after a failed assertion.
            stdin.close_input()
            await settled.wait(5)
        assert recorder.interrupts == 0


@pytest.mark.tonio
async def test_rejects_an_already_aborted_prompt_and_cleans_up():
    recorder = Recorder()
    cancel = CancelToken()
    cancel.cancel()
    with PipeStdin() as stdin:
        with pytest.raises(AbortError):
            await question_terminal(
                "Paste URL: ",
                stdin=stdin.reader,
                output=recorder.write,
                cancel=cancel,
                on_interrupt=recorder.on_interrupt,
            )
        assert recorder.interrupts == 0
        assert recorder.output == []


@pytest.mark.tonio
async def test_two_lines_in_one_chunk_answer_two_questions():
    recorder = Recorder()
    with PipeStdin() as stdin:
        stdin.write(b"first\nsecond\n")
        stdin.close_input()  # a lost second line ends the second question instead of parking it
        for expected in ("first", "second"):
            answer = await question_terminal(
                "Answer: ", stdin=stdin.reader, output=recorder.write, cancel=CancelToken()
            )
            assert answer == expected


@pytest.mark.tonio
async def test_a_cancelled_question_keeps_its_half_line_for_the_next_question():
    """The token fires while the first question waits for the rest of a line it
    has started reading: the wait ends at once, and the next question answers
    with the whole line once the rest arrives."""
    recorder = Recorder()
    cancel = WaitedToken(waits=2)
    first_settled = tonio.Event()
    first_outcome: list[BaseException] = []

    async def first_question() -> None:
        try:
            await question_terminal("First: ", stdin=stdin.reader, output=recorder.write, cancel=cancel)
        except Exception as error:
            first_outcome.append(error)
        finally:
            first_settled.set()

    with PipeStdin() as stdin:
        try:
            first = tonio.spawn(first_question())
            # Parked on the empty pipe, then fed half a line.
            await cancel.waiting[0].wait(5)
            assert cancel.waiting[0].is_set()
            stdin.write(b"first ha")
            # Parked again: the half line has been read.
            await cancel.waiting[1].wait(5)
            assert cancel.waiting[1].is_set()
            cancel.cancel()
            await first_settled.wait(5)
            assert first_settled.is_set(), "the cancelled question kept waiting for input"
            assert len(first_outcome) == 1 and isinstance(first_outcome[0], AbortError)
            await first

            stdin.write(b"lf\n")
            answer = await question_terminal(
                "Second: ", stdin=stdin.reader, output=recorder.write, cancel=CancelToken()
            )
            assert answer == "first half"
        finally:
            # EOF ends a question still parked after a failed assertion.
            stdin.close_input()
            await first_settled.wait(5)


@pytest.mark.tonio
async def test_a_second_question_while_one_waits_fails_loudly():
    recorder = Recorder()
    cancel = WaitedToken()
    settled = {"first": tonio.Event(), "second": tonio.Event()}
    outcomes: dict[str, object] = {}

    async def question(name: str, token: CancelToken) -> None:
        try:
            outcomes[name] = await question_terminal(
                f"{name}: ", stdin=stdin.reader, output=recorder.write, cancel=token
            )
        except Exception as error:
            outcomes[name] = error
        finally:
            settled[name].set()

    with PipeStdin() as stdin:
        try:
            first = tonio.spawn(question("first", cancel))
            await cancel.waiting[0].wait(5)
            assert cancel.waiting[0].is_set()
            second = tonio.spawn(question("second", CancelToken()))
            await settled["second"].wait(5)
            assert settled["second"].is_set(), "the second question waited beside the first"
            assert isinstance(outcomes["second"], WouldBlock)
            await second
            stdin.write(b"for the first\n")
            await first
            assert outcomes["first"] == "for the first"
        finally:
            # EOF ends a question still parked after a failed assertion.
            stdin.close_input()
            for event in settled.values():
                await event.wait(5)
