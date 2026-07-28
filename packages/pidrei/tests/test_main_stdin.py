"""`main._read_piped_stdin` / `main._prompt_confirm` — readiness-driven stdin.

Both moved off the blocking pool under the stdio teardown policy (task #92):
a pipe is driven by readiness with `O_NONBLOCK` restored by `FdReader.close`
on the orderly path and by `snapshot_std_blocking`/`hard_exit` on the rest.
These tests swap `sys.stdin` for a pipe-backed file by hand (no yield
fixtures: tonio) and check both the answer and the descriptor hygiene.
"""

import os
import sys

import pytest

from pidrei.main import _prompt_confirm, _read_piped_stdin


async def _with_pipe_stdin(payload: bytes, run):
    read_fd, write_fd = os.pipe()
    if payload:
        os.write(write_fd, payload)
    os.close(write_fd)
    stdin = os.fdopen(read_fd, "r", closefd=True)
    saved = sys.stdin
    sys.stdin = stdin
    try:
        result = await run()
        # The orderly path restored the shell-shared flag.
        assert os.get_blocking(read_fd)
        return result
    finally:
        sys.stdin = saved
        stdin.close()


@pytest.mark.tonio
async def test_read_piped_stdin_drains_a_pipe_via_readiness():
    result = await _with_pipe_stdin(b"hello from a pipe\n", _read_piped_stdin)
    assert result == "hello from a pipe"


@pytest.mark.tonio
async def test_read_piped_stdin_returns_none_for_empty_input():
    assert await _with_pipe_stdin(b"", _read_piped_stdin) is None


@pytest.mark.tonio
async def test_prompt_confirm_accepts_yes():
    assert await _with_pipe_stdin(b"y\n", lambda: _prompt_confirm("Fork?")) is True


@pytest.mark.tonio
async def test_prompt_confirm_defaults_to_no():
    assert await _with_pipe_stdin(b"nope\n", lambda: _prompt_confirm("Fork?")) is False


@pytest.mark.tonio
async def test_prompt_confirm_treats_eof_as_no():
    assert await _with_pipe_stdin(b"", lambda: _prompt_confirm("Fork?")) is False
