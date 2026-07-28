"""Backpressure and delivery contract of `core/output_guard`.

pi serializes raw writes on a promise chain, and pidrei now does the same with a
colored writer task. These tests pin the contract, not the mechanism, so they
survived the writer-thread era unchanged: callers must not resume until every
queued chunk has reached the stream, whichever path delivered it. A test double
without a real fd exercises the pool path; a real pipe exercises the `arm_w`
readiness path.

No yield fixtures (the tonio pytest plugin cannot wrap them), so each test does
its own setup/teardown around `take_over_stdout`.
"""

import os
import sys
import threading

import pytest
import tonio.colored as tonio

from pidrei.core.output_guard import (
    flush_raw_stdout,
    restore_stdout,
    take_over_stdout,
    wait_for_raw_stdout_backpressure,
    write_raw_stdout,
)
from pidrei.utils.fd_io import FdReader


class _RecordingStream:
    """Stands in for the real stdout, optionally stalling the writer thread."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.chunks: list[str] = []
        self.flushes = 0
        self._gate = gate
        self._lock = threading.Lock()

    def write(self, text: str) -> int:
        if self._gate is not None:
            self._gate.wait(5)
        with self._lock:
            self.chunks.append(text)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self.flushes += 1


def _install(stream: _RecordingStream) -> object:
    original = sys.stdout
    sys.stdout = stream  # type: ignore[assignment]
    take_over_stdout()
    return original


def _uninstall(original: object) -> None:
    restore_stdout()
    sys.stdout = original  # type: ignore[assignment]


@pytest.mark.tonio
async def test_backpressure_returns_immediately_when_nothing_is_queued():
    stream = _RecordingStream()
    original = _install(stream)
    try:
        _, completed = await tonio.time.timeout(wait_for_raw_stdout_backpressure(), 1.0)
        assert completed
        assert stream.chunks == []
    finally:
        _uninstall(original)


@pytest.mark.tonio
async def test_backpressure_waits_until_every_queued_chunk_reached_the_stream():
    gate = threading.Event()
    stream = _RecordingStream(gate)
    original = _install(stream)
    try:
        for index in range(5):
            write_raw_stdout(f"chunk-{index}\n")

        # Writer thread is stalled inside write(), so the caller must not resume.
        _, completed = await tonio.time.timeout(wait_for_raw_stdout_backpressure(), 0.05)
        assert not completed, "resumed while writes were still queued"

        gate.set()
        _, completed = await tonio.time.timeout(wait_for_raw_stdout_backpressure(), 5.0)
        assert completed
        assert stream.chunks == [f"chunk-{index}\n" for index in range(5)]
    finally:
        gate.set()
        _uninstall(original)


@pytest.mark.tonio
async def test_concurrent_waiters_all_wake_on_the_same_drain():
    gate = threading.Event()
    stream = _RecordingStream(gate)
    original = _install(stream)
    try:
        write_raw_stdout("payload\n")
        waiters = tonio.spawn(
            wait_for_raw_stdout_backpressure(),
            wait_for_raw_stdout_backpressure(),
            wait_for_raw_stdout_backpressure(),
        )
        gate.set()
        _, completed = await tonio.time.timeout(waiters, 5.0)
        assert completed, "a registered waiter was never woken"
    finally:
        gate.set()
        _uninstall(original)


@pytest.mark.tonio
async def test_flush_drains_then_flushes_the_stream():
    stream = _RecordingStream()
    original = _install(stream)
    try:
        write_raw_stdout("first\n")
        await flush_raw_stdout()
        assert stream.chunks == ["first\n"]
        assert stream.flushes >= 1
    finally:
        _uninstall(original)


@pytest.mark.tonio
async def test_readiness_path_delivers_through_a_real_pipe_in_order():
    """Raw stdout pointing at an exclusive pipe takes the `arm_w` branch: no
    dedicated thread, and the writes land on the pipe in order. The drain runs
    concurrently as its own runtime task, so this would deadlock if the writer
    held a worker instead of parking on readiness."""
    read_fd, write_fd = os.pipe()
    stream = os.fdopen(write_fd, "w", encoding="utf-8", closefd=False)
    original = sys.stdout
    sys.stdout = stream  # type: ignore[assignment]
    take_over_stdout()
    received = bytearray()

    async def drain() -> None:
        drain_reader = FdReader(read_fd)
        try:
            while chunk := await drain_reader.read():
                received.extend(chunk)
                if received.endswith(b"chunk-99\n"):
                    return
        finally:
            drain_reader.close()

    try:
        drain_join = tonio.spawn(drain())
        for index in range(100):
            write_raw_stdout(f"chunk-{index}\n")
        await wait_for_raw_stdout_backpressure()
        await drain_join
        assert received.decode() == "".join(f"chunk-{index}\n" for index in range(100))
        assert not any(t.name == "pidrei-raw-stdout" for t in threading.enumerate()), (
            "the dedicated writer thread should no longer exist"
        )
    finally:
        restore_stdout()
        sys.stdout = original  # type: ignore[assignment]
        # Hand a write to the pool path so the writer task drops its FdWriter
        # for the now-restored stdout before the pipe fds go away.
        write_raw_stdout(" ")
        await wait_for_raw_stdout_backpressure()
        stream.close()
        os.close(read_fd)
        os.close(write_fd)
