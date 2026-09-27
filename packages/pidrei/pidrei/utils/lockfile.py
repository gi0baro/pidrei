"""Inter-process file locking (equivalent of pi's proper-lockfile usage).

proper-lockfile takes a mkdir-based lock: it creates `<path>.lock` as a
directory (atomic on POSIX) and considers a lock stale after 10 seconds.
pi wraps `lockSync` in its own retry loop (10 attempts, 20ms apart, a busy
wait); `FileLock.acquire` is that loop, waiting on the runtime. Every
filesystem step goes through `tonio.colored.fs`, except refreshing a stale
lock's mtime, which `fs` has no call for (`Path.touch` would create a file if
the directory vanished meanwhile) and so goes to the pool as `os.utime`.
"""

import os
import time
from typing import Self

import tonio.colored as tonio
from tonio.colored import fs


STALE_SECONDS = 10.0


class LockedError(Exception):
    def __init__(self, path: str):
        super().__init__(f"Lock file is already being held: {path}")
        self.code = "ELOCKED"


class FileLock:
    """The lock for `path`: `async with` it, or `acquire()`/`release()` it by
    hand where the lock is taken conditionally. One acquisition at a time, not
    reentrant."""

    def __init__(
        self,
        path: str,
        *,
        lockfile_path: str | None = None,
        stale: float = STALE_SECONDS,
        max_attempts: int = 10,
        delay: float = 0.02,
    ) -> None:
        self._dir = fs.Path(lockfile_path if lockfile_path is not None else f"{path}.lock")
        self._stale = stale
        self._max_attempts = max_attempts
        self._delay = delay
        self._held = False

    @property
    def held(self) -> bool:
        return self._held

    async def try_acquire(self) -> None:
        """One attempt. Raises LockedError (code ELOCKED) when the lock is
        held by someone else."""
        try:
            await self._dir.mkdir()
        except FileExistsError:
            try:
                mtime = (await self._dir.stat()).st_mtime
            except OSError:
                mtime = None
            if mtime is None or time.time() - mtime <= self._stale:
                raise LockedError(str(self._dir)) from None
            # Stale lock left behind by a dead process: steal it.
            try:
                await tonio.spawn_blocking(os.utime, self._dir)
            except OSError:
                raise LockedError(str(self._dir)) from None
        self._held = True

    async def acquire(self) -> None:
        """Mirror of pi's acquireLockSyncWithRetry: retry ELOCKED up to
        `max_attempts`, `delay` apart."""
        for _ in range(self._max_attempts - 1):
            try:
                await self.try_acquire()
                return
            except LockedError:
                await tonio.time.sleep(self._delay)
        await self.try_acquire()

    async def release(self) -> None:
        """Remove the lock; nothing to do unless it is held."""
        if not self._held:
            return
        self._held = False
        try:
            await self._dir.rmdir()
        except OSError:
            pass

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()
