"""The Monty worker pool scripts run in.

Scripts never run in pidrei's process: each one checks out a session from a
pool of `monty` worker subprocesses. `CodemodePool` owns one pool; its owner
(the codemode extension) enters it at `session_start` and closes it at
`session_shutdown`.

Importing this module primes Monty's process-wide runtime (see
`_prime_monty_runtime_blocking`), so it is imported at program start, before
`tonio.run`: importing it lazily from a running program opens a window of a few
milliseconds in which a subprocess spawned on another thread inherits
`TOKIO_WORKER_THREADS=1`.
"""

import os
import pathlib
import sysconfig

import tonio.colored as tonio
from pydantic_monty import Monty


# The binary installed by `pydantic-monty-runtime` in this environment's scripts
# directory. Pinned rather than resolved through `MONTY_BIN` or `PATH`, which
# Monty's docs advise against for untrusted code. Computed, not probed: a
# missing binary fails the first checkout, which is reported as a sandbox
# error.
MONTY_BINARY = str(pathlib.Path(sysconfig.get_path("scripts")) / "monty")

# How long the parent waits past a feed's execution limit before killing the
# worker itself, giving the sandbox time to raise its own `TimeoutError`.
FEED_DURATION_LIMIT_GRACE_SECS = 1.0

_TOKIO_WORKER_THREADS = "TOKIO_WORKER_THREADS"


def _prime_monty_runtime_blocking() -> None:
    """Build Monty's tokio runtime with one thread.

    The runtime is process-wide, built when the first pool is entered and never
    torn down, and it starts one thread per CPU unless `TOKIO_WORKER_THREADS`
    says otherwise; its threads only shuttle messages to the workers. The
    variable must not stay in the environment, where every subprocess would
    inherit it, so it is set around one throwaway pool (which spawns no
    worker) and put back as it was. A value the user set is not honoured for
    Monty's runtime.
    """
    previous = os.environ.get(_TOKIO_WORKER_THREADS)
    os.environ[_TOKIO_WORKER_THREADS] = "1"
    try:
        with Monty(binary_path=MONTY_BINARY, min_processes=0):
            pass
    finally:
        if previous is None:
            del os.environ[_TOKIO_WORKER_THREADS]
        else:
            os.environ[_TOKIO_WORKER_THREADS] = previous


# Module-level code runs at import, before `tonio.run` (see the module docstring).
_prime_monty_runtime_blocking()


class CodemodePool:
    """One Monty pool, with no warm workers: the first script's checkout spawns
    one. `pool = await CodemodePool()`; `await pool.close()` when done.

    Closing does not wait for scripts still running: they finish normally.
    """

    def __init__(self, *, binary_path: str = MONTY_BINARY, max_processes: int | None = None) -> None:
        self._entered = False
        self.monty = Monty(
            binary_path=binary_path,
            min_processes=0,
            max_processes=max_processes,
            feed_duration_limit_grace=FEED_DURATION_LIMIT_GRACE_SECS,
        )

    def __await__(self):
        return self._start().__await__()

    async def _start(self) -> CodemodePool:
        await tonio.spawn_blocking(self.monty.__enter__)
        self._entered = True
        return self

    async def close(self) -> None:
        """Close the pool. Safe on a pool that was never entered, and twice."""
        if self._entered:
            await tonio.spawn_blocking(self.monty.__exit__, None, None, None)
