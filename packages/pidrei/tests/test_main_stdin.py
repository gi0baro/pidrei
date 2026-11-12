"""`main._read_piped_stdin` / `main._prompt_confirm` — stdin through the run's
`StdinReader`.

Each test hands the function a reader over a pipe it made itself, non-blocking
as stdin is from process entry on, and closes both in the test body.
"""

import os

import pytest

from pidrei.main import _prompt_confirm, _read_piped_stdin
from pidrei.utils.fd_io import StdinReader


async def _with_pipe_stdin(payload: bytes, run):
    read_fd, write_fd = os.pipe()
    os.set_blocking(read_fd, False)
    if payload:
        os.write(write_fd, payload)
    os.close(write_fd)
    stdin = StdinReader(read_fd)
    try:
        return await run(stdin)
    finally:
        stdin.close()
        os.close(read_fd)


@pytest.mark.tonio
async def test_read_piped_stdin_drains_a_pipe_via_readiness():
    result = await _with_pipe_stdin(b"hello from a pipe\n", _read_piped_stdin)
    assert result == "hello from a pipe"


@pytest.mark.tonio
async def test_read_piped_stdin_returns_none_for_empty_input():
    assert await _with_pipe_stdin(b"", _read_piped_stdin) is None


@pytest.mark.tonio
async def test_prompt_confirm_accepts_yes():
    assert await _with_pipe_stdin(b"y\n", lambda stdin: _prompt_confirm(stdin, "Fork?")) is True


@pytest.mark.tonio
async def test_prompt_confirm_defaults_to_no():
    assert await _with_pipe_stdin(b"nope\n", lambda stdin: _prompt_confirm(stdin, "Fork?")) is False


@pytest.mark.tonio
async def test_prompt_confirm_treats_eof_as_no():
    assert await _with_pipe_stdin(b"", lambda stdin: _prompt_confirm(stdin, "Fork?")) is False
