"""`utils/fd_io` — `FdReader`/`FdWriter`, one dispatch, three descriptor kinds.

What is pinned here is the dispatch and the descriptor hygiene, not tonio's
primitives: a regular file goes through `fs.wrap_file`, a pipe can be driven by
readiness or left on the pool, and in every case the descriptor comes back the
way it was handed over — same blocking flag, still open.

No yield fixtures (the tonio plugin cannot wrap them), so pipes and temp files
are made and torn down by hand.
"""

import os
import tempfile

import pytest
import tonio.colored as tonio

from pidrei.utils.fd_io import FdReader, FdWriter, is_pollable


def _file_with(content: bytes) -> str:
    path = os.path.join(tempfile.mkdtemp(), "input")
    with open(path, "wb") as handle:
        handle.write(content)
    return path


async def _read_all(reader: FdReader) -> bytes:
    chunks: list[bytes] = []
    while chunk := await reader.read():
        chunks.append(chunk)
    return b"".join(chunks)


def test_classifies_descriptor_kinds():
    path = _file_with(b"x")
    fd = os.open(path, os.O_RDONLY)
    try:
        assert not is_pollable(fd)
    finally:
        os.close(fd)

    read_fd, write_fd = os.pipe()
    try:
        assert is_pollable(read_fd)
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.tonio
async def test_reads_a_regular_file_to_eof():
    fd = os.open(_file_with(b"hello from a file\n"), os.O_RDONLY)
    reader = FdReader(fd, size=4)  # small size: several reads, then b""
    try:
        assert await _read_all(reader) == b"hello from a file\n"
    finally:
        reader.close()
        os.close(fd)


@pytest.mark.tonio
async def test_reads_a_pipe_by_readiness_until_the_writer_closes():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"first\n")
    os.write(write_fd, b"second\n")
    os.close(write_fd)  # EOF, so the read loop terminates

    reader = FdReader(read_fd)
    try:
        assert await _read_all(reader) == b"first\nsecond\n"
    finally:
        reader.close()
        os.close(read_fd)


@pytest.mark.tonio
async def test_reads_a_pipe_on_the_pool_when_readiness_is_declined():
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"pooled\n")
    os.close(write_fd)

    reader = FdReader(read_fd, readiness=False)
    try:
        # The whole point of declining: the descriptor is never made
        # non-blocking, so nothing has to be restored later.
        assert os.get_blocking(read_fd)
        assert await _read_all(reader) == b"pooled\n"
        assert os.get_blocking(read_fd)
    finally:
        reader.close()
        os.close(read_fd)


@pytest.mark.tonio
async def test_readiness_restores_the_blocking_flag_it_changed():
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    assert os.get_blocking(read_fd)

    reader = FdReader(read_fd)
    try:
        # `O_NONBLOCK` lives on the shared open file description, so leaving it
        # set would change stdin for whoever handed it to us.
        assert not os.get_blocking(read_fd)
        await reader.read()
    finally:
        reader.close()
    assert os.get_blocking(read_fd)
    os.close(read_fd)


@pytest.mark.tonio
async def test_close_does_not_close_the_descriptor():
    fd = os.open(_file_with(b"still mine\n"), os.O_RDONLY)
    try:
        reader = FdReader(fd)
        await reader.read()
        reader.close()
        # Would raise EBADF if `FdReader` had taken ownership.
        os.fstat(fd)
    finally:
        os.close(fd)


@pytest.mark.skip(
    reason="HANG — TONIO_BUGS #10: deterministic victim of the intermittent lost-wake wedge under full-suite load; "
    "the file alone is green. Deselecting per run kept being forgotten; unskip only to chase the runtime bug."
)
@pytest.mark.tonio
async def test_writer_fills_a_pipe_past_its_buffer_without_stalling_the_runtime():
    """A write far beyond the pipe buffer forces the `arm_w` path to actually
    wait for the reader; a concurrent runtime task drains it. If the writer
    blocked a worker instead of parking on readiness, drain and write would
    deadlock each other."""
    read_fd, write_fd = os.pipe()
    payload = b"x" * 400_000
    received = bytearray()

    async def drain() -> None:
        drain_reader = FdReader(read_fd)
        try:
            while chunk := await drain_reader.read():
                received.extend(chunk)
        finally:
            drain_reader.close()

    writer = FdWriter(write_fd)
    write_end_closed = False

    async def produce() -> None:
        nonlocal write_end_closed
        await writer.write_all(payload)
        writer.close()  # before os.close: restoring the flag needs a live fd
        os.close(write_fd)
        write_end_closed = True

    try:
        assert not os.get_blocking(write_fd)
        await tonio.spawn(produce(), drain())
        assert bytes(received) == payload
    finally:
        if not write_end_closed:
            writer.close()
            os.close(write_fd)
        os.close(read_fd)


@pytest.mark.tonio
async def test_writer_handles_a_regular_file_and_restores_nothing():
    path = os.path.join(tempfile.mkdtemp(), "out")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    writer = FdWriter(fd)
    try:
        await writer.write_all(b"written through fs.wrap_file\n")
    finally:
        writer.close()
        os.close(fd)
    with open(path, "rb") as handle:
        assert handle.read() == b"written through fs.wrap_file\n"


@pytest.mark.tonio
async def test_writer_close_restores_the_blocking_flag_and_leaves_the_fd_open():
    read_fd, write_fd = os.pipe()
    try:
        writer = FdWriter(write_fd)
        assert not os.get_blocking(write_fd)
        await writer.write_all(b"hygiene\n")
        writer.close()
        assert os.get_blocking(write_fd)
        os.fstat(write_fd)  # EBADF here would mean ownership was taken
        assert os.read(read_fd, 64) == b"hygiene\n"
    finally:
        os.close(read_fd)
        os.close(write_fd)


# --- stdio teardown policy (task #92) -----------------------------------------
#
# `O_NONBLOCK` lives on the open file description, which parent and child
# share — so the parent observes on its own pipe end whether the child's exit
# path restored the flag. Sync tests: they drive a child process, not the
# runtime.


def _run_child(code: str, *, expect_blocking_after: bool) -> None:
    import subprocess
    import sys

    read_fd, write_fd = os.pipe()
    try:
        result = subprocess.run(  # noqa: S603 - fixed interpreter, test-authored code
            [sys.executable, "-c", code],
            stdin=read_fd,
            capture_output=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert os.get_blocking(read_fd) is expect_blocking_after
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_hard_exit_restores_the_inherited_blocking_flag():
    _run_child(
        "import os\n"
        "from pidrei.utils.fd_io import snapshot_std_blocking, hard_exit\n"
        "snapshot_std_blocking()\n"
        "os.set_blocking(0, False)\n"
        "hard_exit(0)\n",
        expect_blocking_after=True,
    )


def test_plain_os_exit_leaks_the_flag_which_is_why_hard_exit_exists():
    # Negative control: without the policy, the parent's descriptor stays
    # non-blocking after the child dies. If this ever starts passing with a
    # restored flag, the OS semantics changed and the policy can be retired.
    _run_child(
        "import os\nos.set_blocking(0, False)\nos._exit(0)\n",
        expect_blocking_after=False,
    )
