"""Read or write a descriptor with the tonio primitive that matches its kind.

tonio's descriptor APIs are not interchangeable, and choosing between them is
the caller's job rather than something to discover by trial:

* **socket, pipe, tty** — `io.register` plus `arm_r`/`arm_w`, driving readiness
  by hand. This is what `pidrei_tui/terminal.py`'s input pump does for the
  TUI's stdin.
* **regular file** — `fs.wrap_file`. Readiness is meaningless for a file (the
  kernel refuses to poll one), and reads genuinely block, so they belong on the
  blocking pool. That is not a consolation prize; it is the right answer, and
  it is what libuv does internally, which is why Node appears to treat the two
  transparently.

Driving readiness needs the descriptor to be non-blocking, otherwise a spurious
wakeup turns `os.read` into a blocking call on a *runtime worker* — strictly
worse than leaving the whole read on the pool. `O_NONBLOCK` lives on the open
file description, which is shared with whoever handed us the fd, so `close()`
restores it. `os.dup` would not help (dup shares the description) and reopening
via `/proc/self/fd/N` is Linux-only.

`readiness=False` declines that trade and keeps a pollable descriptor on the
pool, while still using `fs.wrap_file` for a regular file. It exists for callers
whose parent is an interactive shell, where a missed restore is user-visible.
"""

import os
import stat
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs, io as tonio_io


DEFAULT_READ_SIZE = 65536


def is_pollable(fd: int) -> bool:
    """Whether readiness notification applies to this descriptor at all."""
    mode = os.fstat(fd).st_mode
    return stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode) or stat.S_ISCHR(mode)


class FdReader:
    """Chunked reads from `fd`, off the runtime, whatever kind it is.

    Does not own `fd`: `close()` releases what this object registered and
    restores the blocking flag it changed, but never closes the descriptor.
    """

    def __init__(self, fd: int, *, size: int = DEFAULT_READ_SIZE, readiness: bool = True) -> None:
        self._fd = fd
        self._size = size
        self._sio: Any = None
        self._handle: Any = None
        self._saved_blocking: bool | None = None

        if not is_pollable(fd):
            # closefd=False: the descriptor stays the caller's.
            self._handle = fs.wrap_file(os.fdopen(fd, "rb", buffering=0, closefd=False))
        elif readiness:
            self._saved_blocking = os.get_blocking(fd)
            os.set_blocking(fd, False)
            self._sio = tonio_io.register(fd)

    async def read(self) -> bytes:
        """Next chunk, or `b""` at end of input."""
        if self._sio is not None:
            while True:
                if (waiter := self._sio.arm_r()) is not None:
                    await waiter
                    continue
                try:
                    return os.read(self._fd, self._size)
                except BlockingIOError:
                    self._sio.consume_r()
                except InterruptedError:
                    pass
        if self._handle is not None:
            return await self._handle.read(self._size)
        return await tonio.spawn_blocking(os.read, self._fd, self._size)

    def close(self) -> None:
        if self._sio is not None:
            self._sio.close()
            self._sio = None
        if self._saved_blocking is not None:
            os.set_blocking(self._fd, self._saved_blocking)
            self._saved_blocking = None
        self._handle = None


class FdWriter:
    """`FdReader`'s write-side twin: complete writes to `fd`, off the runtime.

    Same dispatch, same non-ownership. The readiness loop is pi's EAGAIN retry
    chain expressed with `arm_w`; the file branch loops over `fs` handle writes
    because a raw (unbuffered) write may be partial.
    """

    def __init__(self, fd: int, *, readiness: bool = True) -> None:
        self._fd = fd
        self._sio: Any = None
        self._handle: Any = None
        self._saved_blocking: bool | None = None

        if not is_pollable(fd):
            self._handle = fs.wrap_file(os.fdopen(fd, "wb", buffering=0, closefd=False))
        elif readiness:
            self._saved_blocking = os.get_blocking(fd)
            os.set_blocking(fd, False)
            self._sio = tonio_io.register(fd)

    async def write_all(self, data: bytes) -> None:
        """Write every byte of `data`, however many rounds that takes."""
        sent = 0
        if self._sio is not None:
            while sent < len(data):
                if (waiter := self._sio.arm_w()) is not None:
                    await waiter
                    continue
                try:
                    sent += os.write(self._fd, data[sent:])
                except BlockingIOError:
                    self._sio.consume_w()
                except InterruptedError:
                    pass
            return
        if self._handle is not None:
            while sent < len(data):
                sent += await self._handle.write(data[sent:])
            return
        while sent < len(data):
            sent += await tonio.spawn_blocking(os.write, self._fd, data[sent:])

    def close(self) -> None:
        if self._sio is not None:
            self._sio.close()
            self._sio = None
        if self._saved_blocking is not None:
            os.set_blocking(self._fd, self._saved_blocking)
            self._saved_blocking = None
        self._handle = None
