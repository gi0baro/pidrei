"""Inter-process file locking (equivalent of pi's proper-lockfile usage).

proper-lockfile takes a mkdir-based lock: it creates `<path>.lock` as a
directory (atomic on POSIX) and considers a lock stale after 10 seconds.
pi wraps `lockSync` in its own retry loop (10 attempts, 20ms apart, a busy
wait); `FileLock.acquire` is that loop, waiting on the runtime. Every
filesystem step goes through `tonio.colored.fs`, except refreshing a stale
lock's mtime, which `fs` has no call for (`Path.touch` would create a file if
the directory vanished meanwhile) and so goes to the pool as `os.utime`.

`renew=True` keeps a long hold from going stale, as proper-lockfile does for
every lock: the holder touches the lock's mtime every `stale / 2`. The MCP
refresh lock is the only lock held across network requests, and the only one
that asks for it. Like proper-lockfile, a renewal that finds an mtime it did
not set (another process took the lock over) stops renewing and gives the lock
up without raising (pi passes a no-op `onCompromised`).
"""

import os
from typing import Self

import tonio.colored as tonio
from tonio.colored import fs
from tonio.colored.sync import Lock

from pidrei_utils import clock
from pidrei_utils.timers import Interval


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
        renew: bool = False,
    ) -> None:
        self._dir = fs.Path(lockfile_path if lockfile_path is not None else f"{path}.lock")
        self._stale = stale
        self._max_attempts = max_attempts
        self._delay = delay
        self._held = False
        self._renew = renew
        # Renewal state: the timer, the mtime (ns) the holder last set, and the
        # lock that keeps a renewal and the release apart, so no touch lands
        # after the rmdir, on a lock another process has just created.
        self._interval: Interval | None = None
        self._mtime_ns = 0
        self._renewal_lock = Lock()

    @property
    def held(self) -> bool:
        return self._held

    async def try_acquire(self) -> None:
        """One attempt. Raises LockedError (code ELOCKED) when the lock is
        held by someone else."""
        try:
            if self._renew:
                # The mtime is read in the same pool job, for the renewals to compare against.
                self._mtime_ns = await tonio.spawn_blocking(self._create_blocking)
            else:
                await self._dir.mkdir()
        except FileExistsError:
            try:
                mtime = (await self._dir.stat()).st_mtime
            except OSError:
                mtime = None
            if mtime is None or clock.now_ms() / 1000 - mtime <= self._stale:
                raise LockedError(str(self._dir)) from None
            # Stale lock left behind by a dead process: steal it.
            try:
                if self._renew:
                    self._mtime_ns = await tonio.spawn_blocking(self._touch_blocking)
                else:
                    await tonio.spawn_blocking(os.utime, self._dir)
            except OSError:
                raise LockedError(str(self._dir)) from None
        self._held = True
        if self._renew:
            self._interval = Interval(self._stale * 500, self._schedule_renewal)

    def _create_blocking(self) -> int:
        os.mkdir(self._dir)
        return os.stat(self._dir).st_mtime_ns

    def _touch_blocking(self) -> int:
        """Set the lock's mtime to now; returns it as the filesystem stored it."""
        now = clock.now_ms() / 1000
        os.utime(self._dir, (now, now))
        return os.stat(self._dir).st_mtime_ns

    def _schedule_renewal(self) -> None:
        # The timer callback is synchronous; the touch is pool I/O.
        tonio.spawn.without_tracking(self._renew_once())

    async def _renew_once(self) -> None:
        async with self._renewal_lock:
            if not self._held:
                return
            try:
                mtime_ns = await tonio.spawn_blocking(self._renew_blocking, self._mtime_ns)
            except FileNotFoundError:
                mtime_ns = None
            except OSError:
                # Tried again on the next tick; the lock goes stale if every
                # touch fails.
                return
            if mtime_ns is None:
                # Compromised: another process holds the lock now. Release
                # must not remove it.
                self._held = False
                self._stop_renewing()
                return
            self._mtime_ns = mtime_ns

    def _renew_blocking(self, expected_ns: int) -> int | None:
        """The new mtime, or None when the lock's mtime is not the one this
        holder set."""
        if os.stat(self._dir).st_mtime_ns != expected_ns:
            return None
        return self._touch_blocking()

    def _stop_renewing(self) -> None:
        interval, self._interval = self._interval, None
        if interval is not None:
            interval.cancel()

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
        # Synchronous, before the first await: a release on a cancelled chain
        # fails at its first await, and must not leave a lock renewed forever.
        self._stop_renewing()
        if not self._held:
            return
        if not self._renew:
            self._held = False
            await self._remove()
            return
        async with self._renewal_lock:
            # A renewal that ran meanwhile may have found the lock taken over.
            if not self._held:
                return
            self._held = False
            await self._remove()

    async def _remove(self) -> None:
        try:
            await self._dir.rmdir()
        except OSError:
            pass

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.release()
