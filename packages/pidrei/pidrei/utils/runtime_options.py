"""tonio runtime sizing for pidrei entry points (no pi counterpart).

tonio's own defaults are cpu_count() workers and a fixed 128-thread blocking
pool cap. pidrei's workload is I/O-bound: worker threads beyond a handful buy
nothing on large machines, and the blocking pool (on-demand, 30s idle TTL)
only needs a cap proportional to the workers feeding it.

`PIDREI_THREADS` and `PIDREI_BLOCKING_THREADS` override either value; explicit
values are taken as-is, the [2, 8] clamp only shapes the computed default.
"""

import os

_THREADS_MIN = 2
_THREADS_MAX = 8
_BLOCKING_PER_THREAD = 8


def _env_positive_int(name: str) -> int | None:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return None
    return value if value > 0 else None


def runtime_threads() -> int:
    """Worker thread count: process CPU count clamped to [2, 8].

    process_cpu_count() respects affinity masks and cgroup limits, and the
    lower clamp doubles as its None fallback.
    """
    if (override := _env_positive_int("PIDREI_THREADS")) is not None:
        return override
    return min(max(os.process_cpu_count() or 0, _THREADS_MIN), _THREADS_MAX)


def runtime_options() -> dict:
    """Keyword arguments for `tonio.run` sizing the runtime for pidrei."""
    threads = runtime_threads()
    blocking = _env_positive_int("PIDREI_BLOCKING_THREADS")
    return {
        "threads": threads,
        "blocking_threadpool_size": blocking if blocking is not None else threads * _BLOCKING_PER_THREAD,
    }
