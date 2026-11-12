"""Port of pi's utils/terminal-question.ts: one line read from the terminal.

pi runs Node's readline over stdin. Here the line is taken from the run's
`StdinReader`, which reads on demand and keeps what it read: the caller's
token ends the wait for input, and a line it interrupts is kept for the next
reader. Ctrl+C at this canonical-mode read is SIGINT for the whole process,
which PiDrei leaves to CPython's default handling, so only end of input
reports an interrupt.
"""

from collections.abc import Callable

from pidrei_utils.cancel import AbortError, CancelToken

from ..core.output_guard import drain_output
from .fd_io import StdinReader


async def question_terminal(
    message: str,
    *,
    stdin: StdinReader,
    output: Callable[[str], None],
    cancel: CancelToken | None,
    on_interrupt: Callable[[], None] | None = None,
) -> str:
    """Read one line from `stdin` after writing `message` with `output`. Raises
    when `cancel` fires or at end of input; `on_interrupt` is called at end of
    input, not when the caller cancels just this prompt."""
    if cancel is not None:
        cancel.raise_if_cancelled()
    output(message)
    await drain_output()
    line = await _read_line(stdin, cancel)
    if line is None:
        if on_interrupt is not None:
            on_interrupt()
        raise AbortError("The operation was aborted")
    return line


async def _read_line(stdin: StdinReader, cancel: CancelToken | None) -> str | None:
    """The next line of stdin; a line cut short by end of input is returned as
    is, and None means end of input before any byte."""
    while (line := stdin.take_line()) is None:
        if not await stdin.fill(cancel):
            line = stdin.take()
            if not line:
                return None
            break
    lines = line.decode("utf-8", "replace").splitlines()
    return lines[0] if lines else ""
