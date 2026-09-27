"""Delivery contract of `core/output_guard`: every stdio write.

pi serializes raw writes on a promise chain, and pidrei does the same with one
writer task for fds 1 and 2. The writer only ever writes those two fds, so each
test points them at a pipe it owns (`dup2`, put back afterwards) and starts and
stops the writer itself; the conftest guard fails loud if one is left running.
The pipes are driven the way the writer drives them, which is part of what is
pinned: readiness under the takeover, the pool with the flag left alone outside
it, and the pool again for a stdout that shares its description with stdin.

Each test does its own setup/teardown (predates tonio 0.9.14 yield-fixture
support).
"""

import contextlib
import os
import sys

import pytest
import tonio.colored as tonio

from pidrei.core.output_guard import (
    drain_output,
    restore_stdout,
    start_output_writer,
    stop_output_writer,
    take_over_stdout,
    write_raw_stdout,
    write_stderr,
    write_stdout,
)
from pidrei.utils.fd_io import FdReader


@contextlib.contextmanager
def _fds_onto(target_fd: int, *fds: int):
    """Point `fds` at `target_fd`, putting the originals back afterwards."""
    saved = {fd: os.dup(fd) for fd in fds}
    try:
        for fd in fds:
            os.dup2(target_fd, fd)
        yield
    finally:
        for fd, copy in saved.items():
            os.dup2(copy, fd)
            os.close(copy)


@contextlib.contextmanager
def _pipe():
    read_fd, write_fd = os.pipe()
    try:
        yield read_fd, write_fd
    finally:
        os.close(read_fd)
        os.close(write_fd)


async def _read_exactly(read_fd: int, size: int) -> bytes:
    reader = FdReader(read_fd)
    received = bytearray()
    try:
        while len(received) < size:
            chunk = await reader.read()
            if not chunk:
                break
            received.extend(chunk)
    finally:
        reader.close()
    return bytes(received)


def _read_available(read_fd: int) -> bytes:
    """What is in the pipe right now; nothing there is `b""`, never a hang."""
    os.set_blocking(read_fd, False)
    try:
        return os.read(read_fd, 1 << 20)
    except BlockingIOError:
        return b""
    finally:
        os.set_blocking(read_fd, True)


# Far past any pipe buffer (64 KiB on Linux and macOS), so the writer has to
# park until the test reads.
_STALLING_CHUNKS = [f"chunk-{index:04d}-{'x' * 1000}\n" for index in range(300)]
_STALLING_BYTES = "".join(_STALLING_CHUNKS).encode()


@pytest.mark.tonio
async def test_drain_waits_until_every_queued_write_is_on_the_fd():
    with _pipe() as (read_fd, write_fd), _fds_onto(write_fd, 1):
        take_over_stdout()
        start_output_writer()
        try:
            for chunk in _STALLING_CHUNKS:
                write_raw_stdout(chunk)
            resumed = tonio.Event()

            async def wait_for_drain() -> None:
                await drain_output()
                resumed.set()

            waiting = tonio.spawn(wait_for_drain())
            # The writer is parked on the full pipe until the reads below, which
            # run either way: the writer cannot stop with the pipe full.
            await resumed.wait(0.2)
            resumed_early = resumed.is_set()
            received = await _read_exactly(read_fd, len(_STALLING_BYTES))
            await waiting
            assert not resumed_early, "resumed with the pipe full"
            assert received == _STALLING_BYTES
        finally:
            await stop_output_writer()
            restore_stdout()


@pytest.mark.tonio
async def test_stdout_and_stderr_keep_one_order():
    """`2>&1`: both fds on one pipe, the writes interleaved."""
    expected = "".join(f"line-{index}\n" for index in range(50)).encode()
    with _pipe() as (read_fd, write_fd), _fds_onto(write_fd, 1, 2):
        start_output_writer()
        try:
            for index in range(50):
                (write_stdout if index % 2 == 0 else write_stderr)(f"line-{index}\n")
            await drain_output()
        finally:
            await stop_output_writer()
        assert _read_available(read_fd) == expected


@pytest.mark.tonio
async def test_stop_delivers_what_is_still_queued():
    with _pipe() as (read_fd, write_fd), _fds_onto(write_fd, 2):
        start_output_writer()
        for index in range(20):
            write_stderr(f"late-{index}\n")
        await stop_output_writer()
        assert _read_available(read_fd) == "".join(f"late-{index}\n" for index in range(20)).encode()


@pytest.mark.tonio
async def test_without_the_writer_a_write_lands_before_it_returns():
    with _pipe() as (read_fd, write_fd), _fds_onto(write_fd, 2):
        write_stderr("in place\n")
        assert _read_available(read_fd) == b"in place\n"


@pytest.mark.tonio
async def test_takeover_sends_stdout_writes_to_stderr_and_raw_writes_to_stdout():
    with _pipe() as (out_read, out_write), _pipe() as (err_read, err_write), _fds_onto(out_write, 1):
        with _fds_onto(err_write, 2):
            take_over_stdout()
            start_output_writer()
            try:
                write_raw_stdout("protocol\n")
                write_stdout("stray\n")
                sys.stdout.write("foreign\n")  # the takeover's stand-in
                await drain_output()
            finally:
                await stop_output_writer()
                restore_stdout()
        assert _read_available(out_read) == b"protocol\n"
        assert _read_available(err_read) == b"stray\nforeign\n"


@pytest.mark.tonio
async def test_takeover_drives_an_exclusive_stdout_by_readiness_until_the_writer_stops():
    with _pipe() as (_read_fd, write_fd), _fds_onto(write_fd, 1):
        take_over_stdout()
        start_output_writer()
        try:
            write_raw_stdout("x\n")
            await drain_output()
            assert not os.get_blocking(write_fd), "the writer should hold O_NONBLOCK under the takeover"
        finally:
            await stop_output_writer()
            restore_stdout()
        assert os.get_blocking(write_fd), "stopping the writer should give the flag back"


@pytest.mark.tonio
async def test_outside_the_takeover_the_flag_is_left_alone():
    """A child inheriting the terminal (the external editor) needs it blocking."""
    with _pipe() as (_read_fd, write_fd), _fds_onto(write_fd, 1):
        start_output_writer()
        try:
            write_stdout("x\n")
            await drain_output()
            assert os.get_blocking(write_fd)
        finally:
            await stop_output_writer()


@pytest.mark.tonio
async def test_under_the_takeover_a_stdout_shared_with_stdin_is_left_alone():
    """stdin's readers own the flag on that description."""
    with _pipe() as (_read_fd, write_fd), _fds_onto(write_fd, 0, 1):
        take_over_stdout()
        start_output_writer()
        try:
            write_raw_stdout("x\n")
            await drain_output()
            assert os.get_blocking(write_fd)
        finally:
            await stop_output_writer()
            restore_stdout()
