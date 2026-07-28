"""Mirror of pi coding-agent src/core/output-guard.ts.

Protects protocol stdout (JSON/JSONL modes) from stray writes: after
take_over_stdout(), anything using sys.stdout (print(), libraries) is
rerouted to stderr, while write_raw_stdout() writes to the real stdout.

pi serializes raw writes on an async promise chain with EAGAIN retries, and this
is now the same shape: a single colored writer task drains a channel, writing
each chunk with the primitive that matches where stdout points (`FdWriter`) —
`arm_w` readiness for a pipe or socket, the blocking pool for a regular file or
a stream a test substituted. Any write failure exits the process
(pi: process.exit(1)). An earlier revision used a dedicated writer thread here;
the readiness API made it unnecessary.

The readiness branch requires setting `O_NONBLOCK` on stdout, which lives on the
open file description. That is only safe when the description is exclusively
ours, so it is declined whenever stdout appears to share its description with
stdin or stderr (`fstat` device+inode equality): under `2>&1` both names are one
pipe, and on a plain terminal fds 0/1/2 are one pty — making stderr non-blocking
there would break every `print(file=sys.stderr)` in the process, including the
takeover redirect itself. Declining is conservative: two separate opens of the
same FIFO also match, and lose nothing but the readiness fast path.
"""

import os
import sys
import threading
import time
from typing import Any

import tonio.colored as tonio
from tonio.colored.sync import channel
from tonio.exceptions import RuntimeNotInitializedError

from ..utils.fd_io import FdWriter, is_pollable


RAW_STDOUT_RETRY_DELAY_S = 0.010

_takeover_state: dict[str, Any] | None = None

_pending_lock = threading.Lock()
_pending_count = 0
# Runtime tasks waiting for the queue to drain. Registered under `_pending_lock`
# so the writer task cannot cross zero between the check and the registration.
_drain_waiters: list[tonio.Event] = []

_sender, _receiver = channel.unbounded()
# One writer task per process. The single-runtime assumption is project-wide:
# the tonio pytest plugin builds one runtime per session, and `tonio.run` is
# called once per process in production.
_writer_started = False
_fd_writer: FdWriter | None = None
_fd_writer_fd: int | None = None


class _StderrRedirect:
    """Replacement sys.stdout that forwards every write to stderr."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()

    def isatty(self) -> bool:
        try:
            return sys.stderr.isatty()
        except Exception:
            return False

    @property
    def encoding(self) -> str:
        return getattr(sys.stderr, "encoding", "utf-8")

    def fileno(self) -> int:
        return sys.stderr.fileno()


def _get_raw_stream() -> Any:
    if _takeover_state is not None:
        return _takeover_state["raw_stdout"]
    return sys.stdout


def _write_chunk_sync(text: str) -> None:
    """Blocking write through the stream object; the pool-side fallback."""
    while True:
        try:
            stream = _get_raw_stream()
            stream.write(text)
            stream.flush()
            return
        except BlockingIOError:
            time.sleep(RAW_STDOUT_RETRY_DELAY_S)
        except Exception:
            os._exit(1)


def _shares_description(fd: int, other_fd: int) -> bool:
    """Same device+inode — almost certainly the same open file description."""
    try:
        ours, theirs = os.fstat(fd), os.fstat(other_fd)
    except OSError:
        return False  # one of them is closed: nothing to protect
    return (ours.st_dev, ours.st_ino) == (theirs.st_dev, theirs.st_ino)


def _readiness_fd(stream: Any) -> int | None:
    """The fd to drive with `arm_w`, or None to stay on the pool."""
    try:
        fd = stream.fileno()
    except Exception:
        return None  # test double or pseudo-stream: no descriptor to drive
    try:
        if not is_pollable(fd):
            return None
    except OSError:
        return None
    if _shares_description(fd, 2) or _shares_description(fd, 0):
        return None
    return fd


def _drop_fd_writer() -> None:
    global _fd_writer, _fd_writer_fd
    if _fd_writer is not None:
        _fd_writer.close()  # restores the O_NONBLOCK it set
        _fd_writer = None
        _fd_writer_fd = None


async def _deliver(text: str) -> None:
    global _fd_writer, _fd_writer_fd
    stream = _get_raw_stream()
    fd = _readiness_fd(stream)
    if fd is None:
        _drop_fd_writer()
        await tonio.spawn_blocking(_write_chunk_sync, text)
        return

    if fd != _fd_writer_fd:
        _drop_fd_writer()
        # Anything a pre-takeover print() left in the stream's buffer must land
        # before the first fd-level write, or output inverts.
        await tonio.spawn_blocking(stream.flush)
        _fd_writer = FdWriter(fd)
        _fd_writer_fd = fd

    try:
        data = text.encode(getattr(stream, "encoding", None) or "utf-8", getattr(stream, "errors", None) or "strict")
        await _fd_writer.write_all(data)
    except Exception:
        os._exit(1)


async def _writer_loop() -> None:
    global _pending_count
    while True:
        text = await _receiver.receive()
        await _deliver(text)
        drained: list[tonio.Event] = []
        with _pending_lock:
            _pending_count -= 1
            if _pending_count == 0:
                drained = _drain_waiters[:]
                _drain_waiters.clear()
        # Decide under the lock, wake outside it.
        for waiter in drained:
            waiter.set()


def _ensure_writer_task() -> bool:
    """Start the writer task once. False if there is no runtime to start it on."""
    global _writer_started
    if _writer_started:
        return True
    try:
        tonio.spawn.without_tracking(_writer_loop())
    except RuntimeNotInitializedError:
        return False
    _writer_started = True
    return True


def take_over_stdout() -> None:
    global _takeover_state
    if _takeover_state is not None:
        return

    _takeover_state = {
        "raw_stdout": sys.stdout,
        "original_stdout": sys.stdout,
    }
    sys.stdout = _StderrRedirect()  # type: ignore[assignment]


def restore_stdout() -> None:
    global _takeover_state
    if _takeover_state is None:
        return

    sys.stdout = _takeover_state["original_stdout"]
    _takeover_state = None


def is_stdout_taken_over() -> bool:
    return _takeover_state is not None


def write_raw_stdout(text: str) -> None:
    global _pending_count
    if len(text) == 0:
        return
    if not _ensure_writer_task():
        # No runtime, so no worker to protect — the boundary condition that
        # exempts import-time code. Write through synchronously.
        _write_chunk_sync(text)
        return
    with _pending_lock:
        _pending_count += 1
    _sender.send(text)


async def wait_for_raw_stdout_backpressure() -> None:
    waiter = tonio.Event()
    with _pending_lock:
        if _pending_count == 0:
            return
        _drain_waiters.append(waiter)
    # A cancelled wait leaves its event registered; the writer sets it once more
    # and drops it, which costs nothing.
    await waiter.wait()


async def flush_raw_stdout() -> None:
    await wait_for_raw_stdout_backpressure()

    def _flush() -> None:
        try:
            _get_raw_stream().flush()
        except Exception:
            os._exit(1)

    await tonio.spawn_blocking(_flush)
