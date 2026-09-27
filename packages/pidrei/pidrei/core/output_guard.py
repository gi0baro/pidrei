"""Mirror of pi coding-agent src/core/output-guard.ts, and every stdio write.

Protects protocol stdout (JSON/JSONL modes) from stray writes: after
take_over_stdout(), anything using sys.stdout (print(), libraries) is
rerouted to stderr, while write_raw_stdout() writes to the real stdout.

Every write to fds 1 and 2 goes through here, not only the protocol stream:
`write_stdout` (what `print()` was: stdout, or stderr under the takeover),
`write_stderr` and `write_raw_stdout` send the text down one channel and
return, and a single writer task consumes it — stdout and stderr keep one
order, and a slow reader parks the writer instead of the caller. pi gets that
from node's async stdio. `drain_output` is the wait pi's callers get from write
callbacks: it sends a flush ticket and waits for the writer to reach it. The
writer runs between `start_output_writer` and `stop_output_writer`
(`pidrei.__main__` brackets the run with them); outside that window there is
no runtime to protect, and writes go straight to the fd.

Each fd is written with `FdWriter`, in a mode fixed by its first write. Under
the takeover that is `arm_w` readiness, holding `O_NONBLOCK` until the writer
stops: print and RPC modes never hand the terminal to a child (the takeover
starts before either runs and ends only as print mode exits). The exception is
an fd that shares its open file description with stdin, which is where the
flag lives: stdin's readers (`FdReader`, the TUI terminal) set and restore it
on their own schedule, so that fd is written from the blocking pool instead.
Outside the takeover every fd is written from the pool and the flag is left
alone, so the terminal is blocking again whenever the TUI terminal (which holds
the flag between its start and stop) lets go of it — which is what a child
inheriting the terminal, such as the external editor, needs.

A failed stdout write exits the process (pi: process.exit(1)); a failed stderr
write is dropped.

While a `ProcessTerminal` owns the tty (interactive mode, the startup
dialogs: `attach_terminal` to `detach_terminal`), writes to the fds that share
its output description go through the terminal's own queue instead: the tty
then has one writer, so a stderr line never lands inside a frame, and it is
held with the rest of the terminal's output while a child owns the tty.

pidrei's own code writes through the functions here only (lint bans `print`
and `sys.stdout`/`sys.stderr`). `install_stdio_streams` points `sys.stdout`/
`sys.stderr` at stand-ins that do the same, for the code pidrei does not own:
warnings, exception hooks, libraries, extensions.
"""

import os
import sys
from typing import Any

import tonio.colored as tonio
from tonio.colored.sync import channel

from pidrei_tui.terminal import ProcessTerminal, Terminal

from ..utils.fd_io import FdWriter, hard_exit, write_all_blocking


STDOUT_FD = 1
STDERR_FD = 2

_takeover_state: dict[str, Any] | None = None

# The channel's sender and the writer task, while it runs. Only the process
# entry (and tests) start and stop it.
_sender: Any = None
_writer: Any = None

# The attached terminal and the fds it takes, while attached.
_terminal_route: tuple[ProcessTerminal, frozenset[int]] | None = None


class _StdioStream:
    """A text stream that writes through here: `sys.stdout`/`sys.stderr` for
    code pidrei does not own, and stdout's stand-in under the takeover."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def write(self, text: str) -> int:
        _write(self._fd, text)
        return len(text)

    def flush(self) -> None:
        pass  # writes are queued in order; `drain_output` waits for delivery

    def isatty(self) -> bool:
        try:
            return os.isatty(self._fd)
        except OSError:
            return False

    def fileno(self) -> int:
        return self._fd

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return "utf-8"

    @property
    def errors(self) -> str:
        return _errors(self._fd)


def install_stdio_streams() -> None:
    """Process entry only: `sys.stdout`/`sys.stderr` write through here."""
    sys.stdout = _StdioStream(STDOUT_FD)  # noqa: TID251
    sys.stderr = _StdioStream(STDERR_FD)  # noqa: TID251


def take_over_stdout() -> None:
    global _takeover_state
    if _takeover_state is not None:
        return

    _takeover_state = {"original_stdout": sys.stdout}  # noqa: TID251
    sys.stdout = _StdioStream(STDERR_FD)  # noqa: TID251


def restore_stdout() -> None:
    global _takeover_state
    if _takeover_state is None:
        return

    sys.stdout = _takeover_state["original_stdout"]  # noqa: TID251
    _takeover_state = None


def is_stdout_taken_over() -> bool:
    return _takeover_state is not None


def stdout_isatty() -> bool:
    """Whether stdout is a terminal, as `sys.stdout` reports it (under the
    takeover, its stand-in reports stderr's)."""
    try:
        return sys.stdout.isatty()  # noqa: TID251
    except Exception:
        return False


def write_stdout(text: str) -> None:
    """What `print()` was: stdout, or stderr under the takeover."""
    _write(STDERR_FD if _takeover_state is not None else STDOUT_FD, text)


def write_stderr(text: str) -> None:
    _write(STDERR_FD, text)


def write_raw_stdout(text: str) -> None:
    """The real stdout, takeover or not: protocol output."""
    _write(STDOUT_FD, text)


async def drain_output() -> None:
    """Wait until everything written so far is on its fd, the attached
    terminal's queue included (which takes the tty for the moment it needs:
    before a hand-off to a child, not during one)."""
    sender = _sender
    if sender is not None:
        ticket = tonio.Event()
        try:
            sender.send((STDOUT_FD, "", ticket))
        except BrokenPipeError:
            pass  # stopping: the writer delivers what it has before it ends
        else:
            await ticket.wait()
    route = _terminal_route
    if route is not None:
        await route[0].flush()


def start_output_writer() -> None:
    """Start the writer; from here on writes are queued."""
    global _sender, _writer
    if _sender is not None:
        return
    _sender, receiver = channel.unbounded()
    _writer = tonio.spawn(_writer_loop(receiver))


async def stop_output_writer() -> None:
    """Deliver everything queued, then stop the writer, restoring the flags it
    set. Writes from here on go straight to the fd."""
    global _sender, _writer
    sender, writer = _sender, _writer
    if sender is None:
        return
    _sender = _writer = None
    sender.close()  # what is already queued is still delivered
    await writer


async def attach_terminal(terminal: Terminal) -> None:
    """Route writes to the fds sharing `terminal`'s output description
    through its queue, from before its first `arm()` until
    `detach_terminal()`. Writes made earlier are delivered first. Only a
    `ProcessTerminal` writes the process's own fds; any other terminal takes
    nothing."""
    global _terminal_route
    if not isinstance(terminal, ProcessTerminal):
        return
    fds = frozenset(fd for fd in (STDOUT_FD, STDERR_FD) if _shares_description(fd, terminal.output_fd))
    _terminal_route = (terminal, fds)
    # What was queued before the switch goes out first; later writes wait in
    # the terminal's queue behind it.
    await drain_output()


def detach_terminal() -> None:
    """Stop routing to the attached terminal; right before closing it, which
    puts out what its queue still holds."""
    global _terminal_route
    _terminal_route = None


def _errors(fd: int) -> str:
    # Python's own choice for its standard streams.
    return "backslashreplace" if fd == STDERR_FD else "strict"


def _write(fd: int, text: str) -> None:
    if not text:
        return
    route = _terminal_route
    if route is not None and fd in route[1]:
        route[0].write_sync(text)
        return
    sender = _sender
    if sender is not None:
        try:
            sender.send((fd, text, None))
            return
        except BrokenPipeError:
            pass  # stopped meanwhile (see `stop_output_writer`)
    # No writer: before or after the run, where there is no runtime to protect.
    try:
        write_all_blocking(fd, text.encode("utf-8", _errors(fd)))
    except Exception:
        if fd == STDOUT_FD:
            hard_exit(1)


def _shares_description(fd: int, other_fd: int) -> bool:
    """Same device+inode — almost certainly the same open file description."""
    try:
        ours, theirs = os.fstat(fd), os.fstat(other_fd)
    except OSError:
        return False  # one of them is closed: nothing to protect
    return (ours.st_dev, ours.st_ino) == (theirs.st_dev, theirs.st_ino)


def _close_writers(writers: dict[int, FdWriter]) -> None:
    # Newest first: two fds on one description (`2>&1`) restore the flag in the
    # reverse order they set it.
    for writer in reversed(list(writers.values())):
        writer.close()
    writers.clear()


async def _deliver(writers: dict[int, FdWriter], fd: int, text: str) -> None:
    try:
        writer = writers.get(fd)
        if writer is None:
            readiness = _takeover_state is not None and not _shares_description(fd, 0)
            writer = writers[fd] = FdWriter(fd, readiness=readiness)
        await writer.write_all(text.encode("utf-8", _errors(fd)))
    except Exception:
        if fd == STDOUT_FD:
            hard_exit(1)


async def _writer_loop(receiver: Any) -> None:
    writers: dict[int, FdWriter] = {}
    try:
        while True:
            try:
                fd, text, done = await receiver.receive()
            except BrokenPipeError:
                return  # closed by `stop_output_writer`, once drained
            if text:
                await _deliver(writers, fd, text)
            if done is not None:
                done.set()  # a `drain_output` ticket: everything before it is out
    finally:
        _close_writers(writers)
