"""Non-blocking temp-file writer — the port of Node's `createWriteStream`.

pi builds its bash/tool overflow files with `createWriteStream`, whose `write()`
buffers and returns immediately while Node flushes on its own I/O thread
(`core/tools/output-accumulator.ts`, `core/bash-executor.ts`). The first pidrei
port turned that into a blocking `open()` + `write()`, which put a filesystem
write on a tonio runtime worker for every chunk of subprocess output — invisible
to the blocking-fs detector, since writes through an already-open handle raise no
audit event.

This restores pi's shape: `write()` is a channel send with no I/O, and a writer
task performs the actual writes on the blocking pool.

Deferring here is faithful rather than a divergence — pi defers too, which is
what the "a queue is correct only where pi itself defers" rule asks for. Two
details follow from matching `createWriteStream`:

* the queue is **unbounded**, because Node's `write()` never blocks the producer
  (it returns a backpressure hint that pi ignores);
* a write error is captured and re-raised from `close()`, mirroring pi's `error`
  event on the stream, so failures cannot vanish into a detached task.

`close()` drains before returning. pi awaits the `finish` event in the
accumulator but not in bash-executor; pidrei drains in both, deliberately —
Node gets away with the un-awaited `end()` because its flush lands within
microseconds, whereas a pool-backed writer can lag, and the path this produces
exists precisely to be read afterwards.
"""

import contextlib
import os
from typing import BinaryIO

import tonio.colored as tonio
from tonio.colored import fs
from tonio.colored.exceptions import RuntimeNotInitializedError
from tonio.colored.sync import channel


# Output can carry private data, so only the user may read the files.
_PRIVATE_FILE_MODE = 0o600


async def discard_temp_file(path: fs.Path) -> None:
    """Remove a temp file, ignoring errors (it may never have been created).

    From a cancelled task, hand it to `tonio.spawn.without_tracking` instead of
    awaiting it: an await there is not served."""
    with contextlib.suppress(OSError):
        await path.unlink()


def create_private_file_blocking(path: str) -> BinaryIO:
    """Create `path` for writing, readable only by the user. The exclusive
    create (`O_EXCL`) never follows a link someone else placed at the path."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, _PRIVATE_FILE_MODE)
    return os.fdopen(fd, "wb")


class TempFileWriter:
    """Owns a temp file; `write()` never touches the filesystem."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._sender, self._receiver = channel.unbounded()
        self._error: BaseException | None = None
        self._finished = tonio.Event()
        self._closed = False
        self._sync_handle = None

        try:
            tonio.spawn.without_tracking(self._run())
        except RuntimeNotInitializedError:
            # No runtime, so no worker to protect — same boundary condition as
            # import-time code. Write straight through.
            self._sync_handle = create_private_file_blocking(path)
            self._finished.set()

    def write(self, data: bytes) -> None:
        """Buffer `data`. Non-blocking, like `WriteStream.write`."""
        if self._closed:
            return
        if self._sync_handle is not None:
            self._sync_handle.write(data)
            return
        self._sender.send(data)

    async def close(self) -> None:
        """Flush everything already written, then close. Re-raises a write error."""
        if self._sync_handle is not None:
            if not self._closed:
                self._closed = True
                self._sync_handle.close()
            return

        if not self._closed:
            self._closed = True
            self._sender.close()
        await self._finished.wait()
        if self._error is not None:
            raise self._error

    async def _run(self) -> None:
        handle = None
        try:
            # Eagerly, so the file exists as soon as the path is handed out —
            # `createWriteStream` creates it on construction.
            handle = await tonio.spawn_blocking(create_private_file_blocking, self._path)
            while True:
                try:
                    chunk = await self._receiver.receive()
                except BrokenPipeError:
                    break  # sender closed: everything buffered has been drained
                await tonio.spawn_blocking(handle.write, chunk)
        except Exception as error:
            self._error = error
        finally:
            if handle is not None:
                try:
                    await tonio.spawn_blocking(handle.close)
                except Exception as error:
                    if self._error is None:
                        self._error = error
            self._finished.set()
