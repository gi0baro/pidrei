"""Mirror of pi coding-agent src/core/output-guard.ts.

Protects protocol stdout (JSON/JSONL modes) from stray writes: after
take_over_stdout(), anything using sys.stdout (print(), libraries) is
rerouted to stderr, while write_raw_stdout() writes to the real stdout.

pi serializes raw writes on an async promise chain with EAGAIN retries;
here a dedicated writer thread drains a queue, retrying on BlockingIOError
and exiting the process on any other write failure (pi: process.exit(1)).
"""

import os
import queue
import sys
import threading
import time
from typing import Any

import tonio.colored as tonio


RAW_STDOUT_RETRY_DELAY_S = 0.010

_takeover_state: dict[str, Any] | None = None

_write_queue: queue.SimpleQueue[str] = queue.SimpleQueue()
_pending_cond = threading.Condition()
_pending_count = 0
_writer_thread: threading.Thread | None = None


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


def _write_chunk(text: str) -> None:
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


def _writer_loop() -> None:
    global _pending_count
    while True:
        text = _write_queue.get()
        _write_chunk(text)
        with _pending_cond:
            _pending_count -= 1
            if _pending_count == 0:
                _pending_cond.notify_all()


def _ensure_writer_thread() -> None:
    global _writer_thread
    if _writer_thread is None or not _writer_thread.is_alive():
        _writer_thread = threading.Thread(target=_writer_loop, name="pidrei-raw-stdout", daemon=True)
        _writer_thread.start()


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
    with _pending_cond:
        _pending_count += 1
    _ensure_writer_thread()
    _write_queue.put(text)


def _wait_drained() -> None:
    with _pending_cond:
        while _pending_count > 0:
            _pending_cond.wait()


async def wait_for_raw_stdout_backpressure() -> None:
    with _pending_cond:
        drained = _pending_count == 0
    if drained:
        return
    await tonio.spawn_blocking(_wait_drained)


async def flush_raw_stdout() -> None:
    await wait_for_raw_stdout_backpressure()

    def _flush() -> None:
        try:
            _get_raw_stream().flush()
        except Exception:
            os._exit(1)

    await tonio.spawn_blocking(_flush)
