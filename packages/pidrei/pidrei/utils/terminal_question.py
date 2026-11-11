"""Port of pi's utils/terminal-question.ts: one line read from the terminal.

pi runs Node's readline over stdin. Here the line is read on readiness through
`FdReader`, as a scope child (`run_cancellable`) that the caller's token
cancels (`read_stdin`). Ctrl+C at this canonical-mode read is SIGINT for the whole process,
which PiDrei leaves to CPython's default handling, so only end of input
reports an interrupt.
"""

import sys
import threading
from collections.abc import Awaitable, Callable

import tonio.colored as tonio

from pidrei_utils.cancel import AbortError, CancelToken, run_cancellable

from ..core.output_guard import drain_output
from .fd_io import FdReader


async def read_stdin[T](read: Callable[[FdReader], Awaitable[T]], cancel: CancelToken | None) -> T:
    """Run `read` on a reader of stdin until it returns or `cancel` fires, and
    return or raise only once stdin is released, as pi's readline is closed
    when its question settles. A cancelled read unwinds after `run_cancellable`
    has returned (leaving a cancelled scope does not wait for the child), so
    the cancel path waits for the reader's release. Whether the read started
    is settled under a lock: one that never started never takes stdin."""
    claim_guard = threading.Lock()
    claim = {"started": False, "abandoned": False}
    released = tonio.Event()

    async def operation() -> T | None:
        with claim_guard:
            if claim["abandoned"]:
                return None
            claim["started"] = True
        reader: FdReader | None = None
        try:
            reader = FdReader(sys.stdin.fileno())
            return await read(reader)
        finally:
            if reader is not None:
                reader.close()
            released.set()

    try:
        return await run_cancellable(operation(), cancel)
    except Exception:
        with claim_guard:
            claim["abandoned"] = not claim["started"]
            started = claim["started"]
        if started:
            await released.wait()
        raise


async def question_terminal(
    message: str,
    *,
    output: Callable[[str], None],
    cancel: CancelToken | None,
    on_interrupt: Callable[[], None] | None = None,
) -> str:
    """Read one line from stdin after writing `message` with `output`. Raises when
    `cancel` fires or at end of input; `on_interrupt` is called at end of input,
    not when the caller cancels just this prompt."""
    if cancel is not None:
        cancel.raise_if_cancelled()
    output(message)
    await drain_output()
    line = await read_stdin(_read_line, cancel)
    if line is None:
        if on_interrupt is not None:
            on_interrupt()
        raise AbortError("The operation was aborted")
    return line


async def _read_line(reader: FdReader) -> str | None:
    """The first line of stdin, read on readiness; a line cut short by end of
    input is returned as is, and None means end of input before any byte."""
    buffer = b""
    while b"\n" not in buffer:
        chunk = await reader.read()
        if not chunk:
            if not buffer:
                return None
            break
        buffer += chunk
    lines = buffer.decode("utf-8", "replace").splitlines()
    return lines[0] if lines else ""
