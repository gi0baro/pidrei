"""`utils/fd_io` — `StdinReader`/`FdWriter`, one dispatch, three descriptor kinds.

What is pinned here is the dispatch and the descriptor hygiene, not tonio's
primitives: a regular file goes through `fs.wrap_file`, a pipe is driven by
readiness through a dup, and in every case the descriptor comes back the way
it was handed over — same blocking flag, still open, its number free for
another registration. A pipe handed to `StdinReader` or `FdWriter` is made
non-blocking first, as stdio is from process entry on. What the reader keeps
across prompts is pinned in `test_terminal_question.py`.

Pipes and temp files are made and torn down by hand (predates tonio 0.9.14
yield-fixture support).
"""

import os
import tempfile

import pytest
import tonio.colored as tonio
from tonio.colored import io as tonio_io

from pidrei.utils.fd_io import FdWriter, StdinReader, is_pollable, write_all_blocking


def _file_with(content: bytes) -> str:
    path = os.path.join(tempfile.mkdtemp(), "input")
    with open(path, "wb") as handle:
        handle.write(content)
    return path


def _nonblocking_pipe() -> tuple[int, int]:
    """A pipe whose read end is non-blocking, as stdio is from process entry on."""
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    return read_fd, write_fd


async def _read_all(reader: StdinReader) -> bytes:
    while await reader.fill():
        pass
    return reader.take()


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
    fd = os.open(_file_with(b"hello from a file\nand more\n"), os.O_RDONLY)
    reader = StdinReader(fd)
    try:
        assert reader.take_line() is None
        assert await reader.fill()
        assert reader.take_line() == b"hello from a file\n"
        assert await _read_all(reader) == b"and more\n"
    finally:
        reader.close()
        os.close(fd)


@pytest.mark.tonio
async def test_reads_a_pipe_by_readiness_until_the_writer_closes():
    read_fd, write_fd = _nonblocking_pipe()
    os.write(write_fd, b"first\n")
    os.write(write_fd, b"second\n")
    os.close(write_fd)  # EOF, so the read loop terminates

    reader = StdinReader(read_fd)
    try:
        assert await _read_all(reader) == b"first\nsecond\n"
    finally:
        reader.close()
        os.close(read_fd)


@pytest.mark.tonio
async def test_reader_close_leaves_the_fd_open_and_its_number_free_for_another_registration():
    """The TUI terminal registers fd 0 itself while it runs, and a number can
    be registered only once: the reader registers a dup, and closes only that."""
    read_fd, write_fd = _nonblocking_pipe()
    reader = StdinReader(read_fd)
    try:
        terminal_registration = tonio_io.register(read_fd)
        terminal_registration.close()
        os.write(write_fd, b"still mine\n")
        assert await reader.fill()
        reader.close()
        # Would raise EBADF if the reader had taken ownership.
        os.fstat(read_fd)
        assert not os.get_blocking(read_fd)
    finally:
        reader.close()
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.tonio
async def test_writer_fills_a_pipe_past_its_buffer_without_stalling_the_runtime():
    """A write far beyond the pipe buffer forces the `arm_w` path to actually
    wait for the reader; a concurrent runtime task drains it. If the writer
    blocked a worker instead of parking on readiness, drain and write would
    deadlock each other."""
    read_fd, write_fd = _nonblocking_pipe()
    os.set_blocking(write_fd, False)
    payload = b"x" * 400_000
    received = bytearray()

    async def drain() -> None:
        drain_reader = StdinReader(read_fd)
        try:
            while await drain_reader.fill():
                received.extend(drain_reader.take())
        finally:
            drain_reader.close()

    writer = FdWriter(write_fd)
    write_end_closed = False

    async def produce() -> None:
        nonlocal write_end_closed
        await writer.write_all(payload)
        writer.close()  # its dup is a write end too: the drain's EOF needs both closed
        os.close(write_fd)
        write_end_closed = True

    try:
        await tonio.spawn(produce(), drain())
        assert bytes(received) == payload
    finally:
        if not write_end_closed:
            writer.close()
            os.close(write_fd)
        os.close(read_fd)


@pytest.mark.tonio
async def test_blocking_write_waits_for_room_on_a_non_blocking_descriptor():
    """A pollable stdio fd is non-blocking for the whole run, so a write outside
    the output writer's window meets a full pipe as EAGAIN: it waits for room
    instead of failing."""
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    try:
        filled = 0
        try:
            while True:
                filled += os.write(write_fd, b"f" * 65536)
        except BlockingIOError:
            pass
        payload = b"p" * 100_000
        done = tonio.Event()
        failures: list[BaseException] = []

        async def produce() -> None:
            try:
                await tonio.spawn_blocking(write_all_blocking, write_fd, payload)
            except Exception as error:
                failures.append(error)
            finally:
                done.set()

        producing = tonio.spawn(produce())
        await done.wait(0.2)
        assert not done.is_set(), f"gave up on the full pipe: {failures}"
        os.set_blocking(read_fd, False)
        reader = StdinReader(read_fd)
        received = bytearray()
        try:
            while len(received) < filled + len(payload):
                await reader.fill()
                received.extend(reader.take())
        finally:
            reader.close()
        await producing
        assert failures == []
        assert bytes(received[filled:]) == payload
    finally:
        os.close(read_fd)
        os.close(write_fd)


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
async def test_writer_close_leaves_the_fd_open_and_its_flag_as_it_found_it():
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    try:
        writer = FdWriter(write_fd)
        await writer.write_all(b"hygiene\n")
        writer.close()
        assert not os.get_blocking(write_fd)
        os.fstat(write_fd)  # EBADF here would mean ownership was taken
        assert os.read(read_fd, 64) == b"hygiene\n"
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.tonio
async def test_writer_leaves_the_fd_number_free_for_another_registration():
    """The TUI terminal registers its output fd itself while armed, and a
    number can be registered only once: the writer registers a dup."""
    read_fd, write_fd = os.pipe()
    os.set_blocking(write_fd, False)
    writer = FdWriter(write_fd)
    try:
        terminal_registration = tonio_io.register(write_fd)
        try:
            await writer.write_all(b"beside the terminal\n")
        finally:
            terminal_registration.close()
        assert os.read(read_fd, 64) == b"beside the terminal\n"
    finally:
        writer.close()
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
        "from pidrei.utils.fd_io import snapshot_stdio_flags, hard_exit\n"
        "snapshot_stdio_flags()\n"
        "os.set_blocking(0, False)\n"
        "hard_exit(0)\n",
        expect_blocking_after=True,
    )


def test_snapshot_takes_a_pollable_stdio_fd_and_leaves_a_regular_file_alone():
    import subprocess
    import sys

    # `os._exit` skips the exit restore: the parent sees what the entry set.
    code = "import os\nfrom pidrei.utils.fd_io import snapshot_stdio_flags\nsnapshot_stdio_flags()\nos._exit(0)\n"
    read_fd, write_fd = os.pipe()
    file_fd = os.open(os.path.join(tempfile.mkdtemp(), "out"), os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        result = subprocess.run(  # noqa: S603 - fixed interpreter, test-authored code
            [sys.executable, "-c", code],
            stdin=read_fd,
            stdout=file_fd,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr.decode()
        assert not os.get_blocking(read_fd), "the pipe on stdin should be non-blocking from entry"
        assert os.get_blocking(file_fd), "a regular file's flag means nothing and stays as inherited"
    finally:
        os.close(read_fd)
        os.close(write_fd)
        os.close(file_fd)


def test_a_suspend_gives_stdio_its_inherited_flags_and_a_resume_puts_them_back():
    """While pidrei is stopped (Ctrl+Z) the shell owns the terminal. The
    resume puts back what the suspend found, which is not always non-blocking:
    a terminal handed off to a child was blocking and is again."""
    # The child checks its own stdin at each step: a failed assert ends it with
    # a non-zero status and its message on stderr. `os._exit` skips the exit
    # restore, so the parent sees the flag the last resume left.
    code = (
        "import os\n"
        "from pidrei.utils.fd_io import resume_stdio, snapshot_stdio_flags, suspend_stdio\n"
        "snapshot_stdio_flags()\n"
        "assert not os.get_blocking(0), 'entry'\n"
        "suspend_stdio()\n"
        "assert os.get_blocking(0), 'suspended'\n"
        "resume_stdio()\n"
        "assert not os.get_blocking(0), 'resumed'\n"
        "os.set_blocking(0, True)\n"
        "suspend_stdio()\n"
        "resume_stdio()\n"
        "assert os.get_blocking(0), 'resumed while handed off'\n"
        "os._exit(0)\n"
    )
    _run_child(code, expect_blocking_after=True)


def test_plain_os_exit_leaks_the_flag_which_is_why_hard_exit_exists():
    # Negative control: without the policy, the parent's descriptor stays
    # non-blocking after the child dies. If this ever starts passing with a
    # restored flag, the OS semantics changed and the policy can be retired.
    _run_child(
        "import os\nos.set_blocking(0, False)\nos._exit(0)\n",
        expect_blocking_after=False,
    )
