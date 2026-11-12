"""Read or write a descriptor with the tonio primitive that matches its kind.

tonio's descriptor APIs are not interchangeable, and choosing between them is
the caller's job rather than something to discover by trial:

* **socket, pipe, tty** — `io.register` plus `arm_r`/`arm_w`, driving readiness
  by hand. This is what `pidrei_tui/terminal.py`'s input reader does for the
  TUI's stdin.
* **regular file** — `fs.wrap_file`. Readiness is meaningless for a file (the
  kernel refuses to poll one), and reads genuinely block, so they belong on the
  blocking pool. That is not a consolation prize; it is the right answer, and
  it is what libuv does internally, which is why Node appears to treat the two
  transparently.

Driving readiness needs the descriptor to be non-blocking, otherwise a spurious
wakeup turns `os.read` into a blocking call on a *runtime worker* — strictly
worse than leaving the whole read on the pool. `O_NONBLOCK` lives on the open
file description, which is shared with whoever handed us the fd; `os.dup`
shares it too, and reopening via `/proc/self/fd/N` is Linux-only.

For stdio the flag is set once: `snapshot_stdio_flags()` records the inherited
blocking flags of fds 0-2 at process start, registers an `atexit` restore, and
then `take_stdio()` makes every pollable one non-blocking for the run.
`StdinReader` and `FdWriter` rely on that and never touch the flag.

The exit restore is the stdio teardown policy: `hard_exit()` replaces
`os._exit()` everywhere so even the paths that skip interpreter shutdown put
the shell's descriptors back. It covers every in-process flipper —
`take_stdio`, the TUI terminal — because the restore is a blanket "put fds 0-2
back how we found them", not per-owner bookkeeping. SIGKILL is explicitly out
of scope.
"""

import atexit
import os
import select
import stat
from typing import Any, NoReturn

from tonio.colored import fs, io as tonio_io, sync

from pidrei_utils.cancel import CancelToken


READ_SIZE = 65536

_stdio_flags: dict[int, bool] = {}
_restore_registered = False

# The flags fds 0-2 had when the process was suspended, until it resumes.
_suspended_flags: dict[int, bool] | None = None


def snapshot_stdio_flags() -> None:
    """Record the inherited blocking flags of stdio, before anything flips them,
    then take stdio (`take_stdio`).

    Call once at process entry, before any fd registration. Idempotent; a
    second call never overwrites the first snapshot.
    """
    global _restore_registered
    for fd in (0, 1, 2):
        try:
            _stdio_flags.setdefault(fd, os.get_blocking(fd))
        except OSError:
            continue
    if not _restore_registered:
        atexit.register(restore_stdio_flags)
        _restore_registered = True
    take_stdio()


def take_stdio() -> None:
    """Make every pollable fd among the snapshotted fds 0-2 non-blocking.

    Readiness needs it, and `FdWriter` relies on it. A regular file or
    `/dev/null` keeps its flag, which means nothing to it; an fd closed since
    the snapshot is skipped.
    """
    for fd in _stdio_flags:
        try:
            if is_pollable(fd):
                os.set_blocking(fd, False)
        except OSError:
            continue


def restore_stdio_flags() -> None:
    """Put fds 0-2 back to their inherited blocking state. Safe to call twice."""
    for fd, blocking in _stdio_flags.items():
        try:
            os.set_blocking(fd, blocking)
        except OSError:
            continue


def suspend_stdio() -> None:
    """Before the process stops: fds 0-2 get their inherited blocking flags,
    since the shell owns the terminal while the process is stopped.

    The flags they had are kept for `resume_stdio()`: not all are
    non-blocking, a terminal handed off to a child is blocking and has to be
    again. A second call before a resume keeps the first call's flags.
    """
    global _suspended_flags
    if _suspended_flags is None:
        flags: dict[int, bool] = {}
        for fd in _stdio_flags:
            try:
                flags[fd] = os.get_blocking(fd)
            except OSError:
                continue
        _suspended_flags = flags
    restore_stdio_flags()


def resume_stdio() -> None:
    """Once the process continues: fds 0-2 as `suspend_stdio()` found them.
    Nothing to do when no suspend came before."""
    global _suspended_flags
    flags, _suspended_flags = _suspended_flags, None
    if flags is None:
        return
    for fd, blocking in flags.items():
        try:
            os.set_blocking(fd, blocking)
        except OSError:
            continue


def hard_exit(status: int) -> NoReturn:
    """`os._exit` with the shell's stdio blocking flags put back first.

    `os._exit` skips `atexit`, so every immediate-exit path must go through
    here or it leaks `O_NONBLOCK` onto the parent shell's descriptors (the
    shell then sees EAGAIN on its own reads). Do not call `os._exit` directly.
    """
    restore_stdio_flags()
    os._exit(status)


def write_all_blocking(fd: int, data: bytes) -> None:
    """Write every byte of `data` with blocking calls: for a pool thread, or for
    code running with no runtime.

    The descriptor may be non-blocking: a pollable stdio fd is from
    `take_stdio()` on, and `O_NONBLOCK` lives on the open file description, so
    whoever shares it (the TUI terminal) can set it at any time. EAGAIN then
    means "no room yet", so wait until there is and retry.
    """
    buffer = memoryview(data)
    while buffer:
        try:
            buffer = buffer[os.write(fd, buffer) :]
        except BlockingIOError:
            select.select([], [fd], [])
        except InterruptedError:
            pass


def is_pollable(fd: int) -> bool:
    """Whether readiness notification applies to this descriptor at all."""
    mode = os.fstat(fd).st_mode
    if stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
        return True
    # Of the character devices, only terminals can be registered for readiness:
    # epoll refuses /dev/null, /dev/zero and friends.
    return stat.S_ISCHR(mode) and os.isatty(fd)


class StdinReader:
    """The one reader of stdin for the run, and the holder of what it reads.

    Bytes read from `fd` go into this object's buffer, never into a local of
    the coroutine that asked for them, and stay there until a consumer takes
    them (`take_line`, `take`). So a consumer whose token fires gives up its
    wait, never data: a line half read when one prompt is cancelled is
    completed and answered by the next one, and input after the first line of
    a chunk waits for whoever reads next. Reads happen on demand, on the
    consumer's coroutine: nothing reads stdin ahead of a consumer, so input
    meant for the shell or for a child is left alone.

    Built once inside the runtime (on fd 0) and handed down to every consumer.
    One consumer at a time: a call while another consumer's `fill` is in
    progress raises `WouldBlock`, as a bug, rather than interleave two readers.

    Does not own `fd`. A pollable fd is read by readiness through a `dup` of
    it, which this object registers and owns: an fd number can be registered
    only once, and the TUI terminal registers fd 0 itself while it runs. The
    fd must already be non-blocking, as stdio is from `take_stdio()` on; the
    dup shares the flag with `fd`, and this object never touches it. A regular
    file or `/dev/null` is read through `fs.wrap_file`. `close()` drops the
    registration and closes the dup, never `fd`.
    """

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self._buffer = bytearray()
        # Held across a `fill`'s wait, and only ever taken with `or_raise()`:
        # it detects a second consumer, it never queues one. Its release is
        # synchronous, so it also happens on a cancelled chain.
        self._guard = sync.Lock()
        self._dup: int | None = None
        self._sio: Any = None
        self._handle: Any = None

        if not is_pollable(fd):
            # closefd=False: the descriptor stays the caller's.
            self._handle = fs.wrap_file(os.fdopen(fd, "rb", buffering=0, closefd=False))
            return
        dup = os.dup(fd)
        try:
            self._sio = tonio_io.register(dup)
        except BaseException:
            os.close(dup)
            raise
        self._dup = dup

    async def fill(self, cancel: CancelToken | None = None) -> bool:
        """Read one chunk into the buffer; False at end of input.

        Raises the token's reason once `cancel` fires. The token is part of the
        wait for input, never a cancellation of this coroutine. On a pollable
        fd the read and the append to the buffer have no suspension point
        between them, so what was read is kept for the next consumer; a file
        is read on the blocking pool, through `fs.wrap_file`.
        """
        watch = cancel is not None and not cancel.never
        with self._guard.or_raise():
            if self._sio is None:
                # A file read returns promptly, so the token is checked before
                # it, not raced with it.
                if watch:
                    cancel.raise_if_cancelled()
                chunk = await self._handle.read(READ_SIZE)
            else:
                while True:
                    if watch:
                        cancel.raise_if_cancelled()
                    if (waiter := self._sio.arm_r()) is not None:
                        await ((waiter | cancel.wait()) if watch else waiter)
                        continue
                    try:
                        chunk = os.read(self._dup, READ_SIZE)
                        break
                    except BlockingIOError:
                        # Tick-guarded: an edge landing between the failed read
                        # and the clear must survive, or `arm_r` parks with data
                        # already buffered (`consume_r` would eat it).
                        self._sio.clear_r()
                    except InterruptedError:
                        pass
            self._buffer += chunk
            return bool(chunk)

    def take_line(self, ends: bytes = b"\n") -> bytes | None:
        """The buffered bytes through the first of the bytes in `ends`, or None
        while none of them is buffered."""
        with self._guard.or_raise():
            found = [index for index in map(self._buffer.find, ends) if index >= 0]
            if not found:
                return None
            end = min(found) + 1
            line = bytes(self._buffer[:end])
            del self._buffer[:end]
            return line

    def take(self) -> bytes:
        """Everything buffered (`b""` when nothing is)."""
        with self._guard.or_raise():
            data = bytes(self._buffer)
            self._buffer.clear()
            return data

    def close(self) -> None:
        if self._sio is not None:
            self._sio.close()
            self._sio = None
        if self._dup is not None:
            os.close(self._dup)
            self._dup = None
        self._handle = None


class FdWriter:
    """Complete writes to `fd`, off the runtime, whatever kind it is.

    Does not own `fd`. A pollable fd is written by readiness through a `dup` of
    it, which this object registers and owns: an fd number can be registered
    only once, and the TUI terminal registers its output fd itself while armed.
    `close()` drops the registration and closes the dup, never `fd`.

    The fd must already be non-blocking, as stdio is from `take_stdio()` on;
    the dup shares the flag with `fd`, and this object never touches it. The
    readiness loop is pi's EAGAIN retry chain expressed with `arm_w`; the file
    branch loops over `fs` handle writes because a raw (unbuffered) write may
    be partial.
    """

    def __init__(self, fd: int) -> None:
        self._dup: int | None = None
        self._sio: Any = None
        self._handle: Any = None

        if not is_pollable(fd):
            self._handle = fs.wrap_file(os.fdopen(fd, "wb", buffering=0, closefd=False))
            return
        dup = os.dup(fd)
        try:
            self._sio = tonio_io.register(dup)
        except BaseException:
            os.close(dup)
            raise
        self._dup = dup

    async def write_all(self, data: bytes) -> None:
        """Write every byte of `data`, however many rounds that takes."""
        sent = 0
        if self._sio is not None:
            while sent < len(data):
                if (waiter := self._sio.arm_w()) is not None:
                    await waiter
                    continue
                try:
                    sent += os.write(self._dup, data[sent:])
                except BlockingIOError:
                    # Tick-guarded: a writable edge landing between the failed
                    # write and the clear must survive, or `arm_w` parks
                    # forever on an already-writable fd (edge-triggered).
                    self._sio.clear_w()
                except InterruptedError:
                    pass
            return
        while sent < len(data):
            sent += await self._handle.write(data[sent:])

    def close(self) -> None:
        if self._sio is not None:
            self._sio.close()
            self._sio = None
        if self._dup is not None:
            os.close(self._dup)
            self._dup = None
        self._handle = None
