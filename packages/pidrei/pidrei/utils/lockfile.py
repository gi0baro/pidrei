"""Sync inter-process file locking (equivalent of pi's proper-lockfile usage).

proper-lockfile takes a mkdir-based lock: it creates `<path>.lock` as a
directory (atomic on POSIX) and considers a lock stale after 10 seconds.
pi wraps `lockSync` in its own retry loop (10 attempts, 20ms apart); the
retry loop lives with the callers here too, mirroring that structure.
"""

import os
import time
from collections.abc import Callable


STALE_SECONDS = 10.0


class LockedError(Exception):
    def __init__(self, path: str):
        super().__init__(f"Lock file is already being held: {path}")
        self.code = "ELOCKED"


def lock_sync(path: str, *, lockfile_path: str | None = None) -> Callable[[], None]:
    """Acquire the lock for `path`, returning a release callable.

    Raises LockedError (code ELOCKED) when the lock is held by someone else.
    """
    lock_dir = lockfile_path if lockfile_path is not None else f"{path}.lock"
    try:
        os.mkdir(lock_dir)
    except FileExistsError:
        try:
            mtime = os.stat(lock_dir).st_mtime
        except OSError:
            mtime = None
        if mtime is not None and time.time() - mtime > STALE_SECONDS:
            # Stale lock left behind by a dead process: steal it.
            try:
                os.utime(lock_dir)
            except OSError:
                raise LockedError(lock_dir) from None
        else:
            raise LockedError(lock_dir) from None

    def release() -> None:
        try:
            os.rmdir(lock_dir)
        except OSError:
            pass

    return release


def acquire_lock_sync_with_retry(
    path: str,
    *,
    lockfile_path: str | None = None,
    max_attempts: int = 10,
    delay: float = 0.02,
) -> Callable[[], None]:
    """Mirror of pi's acquireLockSyncWithRetry: retry ELOCKED up to max_attempts."""
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return lock_sync(path, lockfile_path=lockfile_path)
        except LockedError as error:
            if attempt == max_attempts:
                raise
            last_error = error
            time.sleep(delay)
    raise last_error if last_error is not None else Exception("Failed to acquire lock")
