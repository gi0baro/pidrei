"""Mirror of pi coding-agent test/exec.test.ts.

pi's child is `node -e`; here it is `python -c`, writing through `os.write`
so nothing is buffered. Not mirrored: the `maxBuffer` above V8's string limit
(no counterpart; the limit has no upper bound here) and `Infinity`/`NaN`
(floats, rejected with the other non-integers).
"""

import os
import signal
import sys

import pytest

from pidrei.core import exec as exec_module
from pidrei.core.exec import ExecResult, exec_command


def _python(script: str) -> tuple[str, list[str]]:
    return sys.executable, ["-c", f"import os, signal, sys, time\n{script}"]


@pytest.mark.tonio
async def test_reports_spawn_errors_in_stderr():
    # Regression test for https://github.com/earendil-works/pi/issues/11085
    result = await exec_command("pi-command-that-does-not-exist-11085", [], os.getcwd())

    assert (result.code, result.killed, result.truncated, result.stdout) == (1, False, False, "")
    # pi's stderr is Node's "spawn ... ENOENT"; here the OSError's message.
    assert "No such file or directory" in result.stderr


@pytest.mark.tonio
async def test_keeps_stderr_empty_when_a_command_exits_silently():
    result = await exec_command(*_python("sys.exit(1)"), os.getcwd())

    assert result == ExecResult(stdout="", stderr="", code=1, killed=False, truncated=False)


# Regression tests for https://github.com/earendil-works/pi/issues/11360


@pytest.mark.parametrize("max_buffer", [-1, 0.5, float("nan"), float("inf"), True])
@pytest.mark.tonio
async def test_rejects_invalid_max_buffer_instead_of_attempting_execution(max_buffer):
    # A missing command normally resolves to a failed result; invalid options must raise instead.
    with pytest.raises(ValueError, match="max_buffer"):
        await exec_command("pi-command-that-does-not-exist-11360", [], os.getcwd(), max_buffer=max_buffer)


@pytest.mark.tonio
async def test_accepts_max_buffer_boundary_zero():
    result = await exec_command(*_python(""), os.getcwd(), max_buffer=0)

    assert result == ExecResult(stdout="", stderr="", code=0, killed=False, truncated=False)


@pytest.mark.tonio
async def test_truncates_all_output_when_max_buffer_is_zero():
    result = await exec_command(*_python("os.write(1, b'x')"), os.getcwd(), max_buffer=0)

    assert (result.stdout, result.stderr, result.truncated, result.killed) == ("", "", True, True)
    assert result.code != 0


@pytest.mark.tonio
async def test_caps_combined_output_at_max_buffer_terminates_the_command_and_reports_truncation():
    script = "os.write(2, b'e' * 600)\nos.write(1, b'o' * 600)\nwhile True:\n    os.write(1, b'x' * 1000)\n    time.sleep(0.01)"
    result = await exec_command(*_python(script), os.getcwd(), max_buffer=1000)

    assert (result.truncated, result.killed) == (True, True)
    assert result.code != 0
    assert len(result.stdout.encode()) + len(result.stderr.encode()) == 1000


# #11360: a byte cap must not turn a partial UTF-8 character into replacement output.
@pytest.mark.parametrize(
    ("fd", "output", "max_buffer", "expected"),
    [(1, "€", 1, ""), (2, "€", 1, ""), (1, "a€🙂", 5, "a€"), (2, "a€🙂", 5, "a€")],
)
@pytest.mark.tonio
async def test_keeps_complete_utf8_characters_from_capped_output(fd, output, max_buffer, expected):
    result = await exec_command(*_python(f"os.write({fd}, {output.encode()!r})"), os.getcwd(), max_buffer=max_buffer)

    assert (result.stdout if fd == 1 else result.stderr) == expected
    assert (result.truncated, result.killed) == (True, True)
    assert result.code != 0
    assert len(result.stdout.encode()) + len(result.stderr.encode()) <= max_buffer


@pytest.mark.tonio
async def test_still_flushes_incomplete_utf8_at_normal_eof_when_output_was_not_truncated():
    result = await exec_command(*_python("os.write(1, b'\\xe2')\nos.write(2, b'\\xf0')"), os.getcwd())

    assert result == ExecResult(stdout="\ufffd", stderr="\ufffd", code=0, killed=False, truncated=False)


@pytest.mark.tonio
async def test_decodes_a_character_split_across_two_writes():
    # PiDrei: the case pi's #11312 fix describes (its commit adds no test for it).
    script = "data = '你好'.encode()\nfor fd in (1, 2):\n    os.write(fd, data[:1])\ntime.sleep(0.1)\nfor fd in (1, 2):\n    os.write(fd, data[1:])"
    result = await exec_command(*_python(script), os.getcwd())

    assert result == ExecResult(stdout="你好", stderr="你好", code=0, killed=False, truncated=False)


@pytest.mark.tonio
async def test_leaves_output_up_to_max_buffer_unchanged():
    result = await exec_command(*_python("os.write(1, b'a' * 1000)"), os.getcwd(), max_buffer=1000)

    assert result == ExecResult(stdout="a" * 1000, stderr="", code=0, killed=False, truncated=False)


@pytest.mark.tonio
async def test_returns_when_a_shells_child_keeps_writing_after_the_shell_is_terminated():
    # SIGTERM kills only `sh`; `yes` keeps the output pipes open and writing.
    result = await exec_command("sh", ["-c", "yes; echo done"], os.getcwd(), max_buffer=1000)

    assert (result.killed, result.truncated) == (True, True)
    assert result.stdout == "y\n" * 500


@pytest.mark.tonio
async def test_returns_once_the_command_exits_while_a_background_process_holds_its_pipes():
    # PiDrei: pi's waitForChildProcess behaviour (its own tests cover it through other callers).
    result = await exec_command("sh", ["-c", "sleep 30 & echo $!"], os.getcwd())
    pid = int(result.stdout)
    try:
        os.kill(pid, 0)  # still running: the call did not wait for it
        assert (result.code, result.killed, result.truncated) == (0, False, False)
    finally:
        os.kill(pid, signal.SIGKILL)


@pytest.mark.tonio
async def test_reports_code_0_for_a_command_its_timeout_terminated():
    # PiDrei: pi reads Node's null exit code of a signal-ended child as 0; `killed` says it was terminated.
    result = await exec_command("sleep", ["10"], os.getcwd(), timeout=100)

    assert result == ExecResult(stdout="", stderr="", code=0, killed=True, truncated=False)


@pytest.mark.tonio
async def test_lays_env_over_the_inherited_environment(monkeypatch):
    # PiDrei: pi's package manager spawns with `{ ...baseEnv, ...options.env }`.
    monkeypatch.setenv("PIDREI_EXEC_TEST_INHERITED", "inherited")
    monkeypatch.setenv("PIDREI_EXEC_TEST_LAID", "inherited")
    script = "os.write(1, ' '.join(os.environ[f'PIDREI_EXEC_TEST_{name}'] for name in ('INHERITED', 'LAID')).encode())"
    result = await exec_command(*_python(script), os.getcwd(), env={"PIDREI_EXEC_TEST_LAID": "laid"})

    assert result == ExecResult(stdout="inherited laid", stderr="", code=0, killed=False, truncated=False)


@pytest.mark.tonio
async def test_force_kills_a_command_that_ignores_sigterm(monkeypatch):
    # pi waits its 5 s; the grace is shortened here.
    monkeypatch.setattr(exec_module, "_FORCE_KILL_DELAY_S", 0.2)
    # The handler is installed before the first write, so the kill triggered by max_buffer
    # cannot arrive before the child ignores SIGTERM.
    script = "signal.signal(signal.SIGTERM, signal.SIG_IGN)\nos.write(1, b'xx')\nwhile True:\n    time.sleep(1)"
    result = await exec_command(*_python(script), os.getcwd(), max_buffer=1)

    assert (result.stdout, result.killed, result.truncated) == ("x", True, True)
    assert result.code != 0
