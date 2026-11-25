"""Non-blocking temp-file writer — the port of Node's `createWriteStream`.

pi builds its bash/tool overflow files with `createWriteStream`, whose `write()`
buffers and returns immediately while Node flushes on its own I/O thread
(`core/tools/output-accumulator.ts`, `core/bash-executor.ts`). The first pidrei
port turned that into a blocking `open()` + `write()`, which put a filesystem
write on a tonio runtime worker for every chunk of subprocess output — invisible
to the blocking-fs detector, since writes through an already-open handle raise no
audit event.

This restores pi's shape: `write()` is a channel send with no I/O, and a writer
coroutine performs the actual writes on the blocking pool.

Deferring here is faithful rather than a divergence — pi defers too, which is
what the "a queue is correct only where pi itself defers" rule asks for. Two
details follow from matching `createWriteStream`:

* the queue is **unbounded**, because Node's `write()` never blocks the producer
  (it returns a backpressure hint that pi ignores);
* the writer is pi's `OutputFileStream` (1.6.1): open, write and close failures
  (full disk, quota, unwritable temp dir) never raise. The first one becomes the
  writer's `result`, one immutable value (`SavedOutputFile` while nothing
  failed, else `FailedOutputFile` with `str(error)`) that the writer coroutine
  rebinds once and other threads read; later writes are ignored, and a file the
  writer created is removed, a file that already sat at the path is not.

`close()` drains, closes the file and returns the result; it never raises. Both
pi callers wait for it (since 1.6.1 also bash-executor), and so does pidrei.
"""

import contextlib
import os
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import BinaryIO, Literal

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


@dataclass(frozen=True, slots=True)
class SavedOutputFile:
    """A complete output file. While output streams in, the file is still being written."""

    path: str
    kind: Literal["saved"] = "saved"


@dataclass(frozen=True, slots=True)
class FailedOutputFile:
    """An output file that could not be saved, and why (`str` of the error)."""

    error: str
    kind: Literal["failed"] = "failed"


#: Where streamed output ended up: a complete file, or why it could not be saved.
OutputFileResult = SavedOutputFile | FailedOutputFile


def _discard_partial_file_blocking(handle: BinaryIO, path: str) -> None:
    with contextlib.suppress(OSError):
        handle.close()
    with contextlib.suppress(OSError):
        os.unlink(path)


class TempFileWriter:
    """Owns a temp file; `write()` never touches the filesystem."""

    def __init__(self, path: str) -> None:
        self._path = path
        self._sender, self._receiver = channel.unbounded()
        # Rebound once, by the writer coroutine, on the first failure.
        self._result: OutputFileResult = SavedOutputFile(path)
        self._finished = tonio.Event()
        self._closed = False
        self._sync = False
        self._sync_handle: BinaryIO | None = None

        try:
            tonio.spawn.without_tracking(self._run())
        except RuntimeNotInitializedError:
            # No runtime, so no worker to protect — same boundary condition as
            # import-time code. Write straight through.
            self._sync = True
            try:
                self._sync_handle = create_private_file_blocking(path)
            except Exception as error:
                self._fail(error)
            self._finished.set()

    @property
    def result(self) -> OutputFileResult:
        """`saved` while writing has not failed; the file is complete only once `close()` returns."""
        return self._result

    def write(self, data: bytes) -> None:
        """Buffer `data`. Non-blocking, like `WriteStream.write`. Ignored after a
        failure and after `close()`: nothing would write it."""
        if self._closed or self._result.kind == "failed":
            return
        if self._sync:
            handle = self._sync_handle
            if handle is None:
                return
            try:
                handle.write(data)
            except Exception as error:
                self._sync_handle = None
                self._fail_off_runtime(handle, error)
            return
        self._sender.send(data)

    def close(self) -> Awaitable[OutputFileResult]:
        """Flush everything already written, then close. Never raises: a failure
        is the returned `FailedOutputFile`, and the partial file is gone by then.
        Writes are ignored from the call on, before the result is awaited (pi's
        `end()` runs when `close()` is called)."""
        if not self._closed:
            self._closed = True
            if self._sync:
                handle = self._sync_handle
                self._sync_handle = None
                if handle is not None:
                    try:
                        handle.close()
                    except Exception as error:
                        self._fail_off_runtime(handle, error)
            else:
                self._sender.close()
        return self._closed_result()

    async def _closed_result(self) -> OutputFileResult:
        await self._finished.wait()
        return self._result

    def _fail(self, error: Exception) -> None:
        # The first failure is the reason; later ones (a close after a failed
        # write) follow from it.
        if self._result.kind == "saved":
            self._result = FailedOutputFile(str(error))

    def _fail_off_runtime(self, handle: BinaryIO, error: Exception) -> None:
        # The no-runtime path (see `__init__`): fail and drop the partial file in place.
        self._fail(error)
        _discard_partial_file_blocking(handle, self._path)

    async def _run(self) -> None:
        handle = None
        try:
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
                self._fail(error)
            if handle is not None:
                try:
                    # Closing can still fail (a delayed ENOSPC or EDQUOT on
                    # quota, NFS or FUSE file systems): the file is not saved then.
                    await tonio.spawn_blocking(handle.close)
                except Exception as error:
                    self._fail(error)
                if self._result.kind == "failed":
                    # A partial file would pass for the full output. A failed
                    # exclusive create leaves the path alone: whatever is
                    # there is not ours.
                    await tonio.spawn_blocking(_discard_partial_file_blocking, handle, self._path)
        finally:
            # Set last, so `close()` returns after the removal.
            self._finished.set()
