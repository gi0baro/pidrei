"""`utils/process.run_command` — the async `subprocess.run` with a real deadline.

pi has no direct equivalent: it reaches these commands through `spawnSync` or a
`spawn` plus `setTimeout`. What is pinned here is our contract — the same
exceptions `subprocess.run` raises, and a deadline that is a *ceiling*, with the
child dead afterwards rather than merely signalled.

No yield fixtures (the tonio pytest plugin cannot wrap them), so temp dirs are
made by hand and module state is restored in `finally`.

**No test here asserts on elapsed time or waits a fixed margin.** The deadline
properties are expressed structurally instead: a child that sleeps for a minute
is either still unfinished when we return (so we did not wait for it) or gone
when we look (so we killed it), and neither statement depends on how fast the
machine is. The only durations are inputs to the code under test — the timeout
we pass, the escalation delay we set — plus a safety bound on the polling
helpers, set well below the child's own lifetime so that exceeding it can only
mean the condition never came true, never that the machine was slow.

Clocks come from `tonio.time.time()`, the runtime's own, rather than
`time.perf_counter`.
"""

import os
import subprocess
import tempfile

import pytest
import tonio.colored as tonio

from pidrei.utils import process as process_module
from pidrei.utils.process import run_command


#: Upper bound for "this should have happened by now" polls. Not a margin: a
#: healthy run leaves it almost entirely unused — the escalation it waits on is
#: set to 0.05s here, a hundredth of this — and exceeding it means the condition
#: never came true rather than that the machine was slow.
#:
#: It is 5s rather than something roomier for an ugly reason, established by
#: mutation testing: with a 30s bound, a *failing* run of the escalation test was
#: reported as **passed** about half the time, its assertion lost while the body
#: was apparently still running. At 5s the same mutation failed correctly every
#: time. See TONIO_BUGS #9 — until that is understood, a wait in a test has to
#: stay short enough to keep its own assertions alive.
_NEVER = 5.0

#: Poll cadence. `interval` ticks on a fixed schedule rather than sleeping for
#: the period after each iteration, so the loop does not drift by however long
#: the check itself took.
_POLL_S = 0.01


def _scratch() -> tuple[str, str]:
    """A fresh (pid file, marker) pair in their own directory."""
    directory = tempfile.mkdtemp()
    return os.path.join(directory, "pid"), os.path.join(directory, "ran")


async def _child_pid(pid_path: str) -> int:
    """The pid the child wrote for itself, once it exists."""
    ticker = tonio.time.interval(_POLL_S)
    deadline = tonio.time.time() + _NEVER
    while tonio.time.time() < deadline:
        try:
            with open(pid_path) as handle:
                recorded = handle.read().strip()
        except OSError:
            recorded = ""
        if recorded:
            return int(recorded)
        await ticker.tick()
    raise AssertionError("child never recorded its pid")


async def _wait_gone(pid: int) -> bool:
    """Poll until `pid` no longer exists. A zombie still answers, so this also
    waits out the gap between the kill and `reap` collecting the child."""
    ticker = tonio.time.interval(_POLL_S)
    deadline = tonio.time.time() + _NEVER
    while tonio.time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await ticker.tick()
    return False


@pytest.mark.tonio
async def test_captures_stdout_stderr_and_exit_code():
    result = await run_command(["sh", "-c", "echo out; echo err >&2; exit 3"], capture_output=True)
    assert result.returncode == 3
    assert result.stdout == b"out\n"
    assert result.stderr == b"err\n"
    assert result.args == ["sh", "-c", "echo out; echo err >&2; exit 3"]


@pytest.mark.tonio
async def test_feeds_stdin_and_closes_it():
    # `cat` only exits once stdin reaches EOF, so this hanging would mean the
    # writer never closed the pipe.
    result = await run_command(["cat"], input=b"piped\n", capture_output=True)
    assert result.returncode == 0
    assert result.stdout == b"piped\n"


@pytest.mark.tonio
async def test_large_output_does_not_deadlock_on_the_pipe_buffer():
    # Well past a 64 KiB pipe buffer: readers have to run alongside the wait.
    result = await run_command(["sh", "-c", "yes abcdefghij | head -c 400000"], capture_output=True)
    assert len(result.stdout) == 400_000


@pytest.mark.tonio
async def test_check_raises_called_process_error_with_output_attached():
    with pytest.raises(subprocess.CalledProcessError) as raised:
        await run_command(["sh", "-c", "echo boom >&2; exit 4"], capture_output=True, check=True)
    assert raised.value.returncode == 4
    assert raised.value.stderr == b"boom\n"


@pytest.mark.tonio
async def test_missing_binary_raises_oserror():
    # Call sites catch OSError to mean "tool not installed".
    with pytest.raises(OSError):
        await run_command(["pidrei-definitely-not-a-real-binary"], capture_output=True)


@pytest.mark.tonio
async def test_timeout_returns_without_waiting_for_the_child_to_die():
    """The deadline is a ceiling, not a floor.

    The child ignores SIGTERM and sleeps for a minute, so an implementation that
    waited for exit would sit here until it finished. No elapsed-time assertion
    is needed: `_NEVER` turns that into a clean failure well before then.
    """
    pid_path, marker = _scratch()
    script = f"echo $$ > {pid_path}; trap '' TERM; sleep 60; touch {marker}"

    async def attempt() -> None:
        with pytest.raises(subprocess.TimeoutExpired):
            await run_command(["sh", "-c", script], capture_output=True, timeout=0.05)

    _, completed = await tonio.time.timeout(attempt(), _NEVER)
    assert completed, "run_command waited for the child instead of giving up"
    assert not os.path.exists(marker), "the child cannot have finished yet"


@pytest.mark.tonio
async def test_a_timed_out_child_does_not_survive():
    pid_path, marker = _scratch()
    script = f"echo $$ > {pid_path}; sleep 60; touch {marker}"
    with pytest.raises(subprocess.TimeoutExpired):
        await run_command(["sh", "-c", script], capture_output=True, timeout=0.05)

    pid = await _child_pid(pid_path)
    assert await _wait_gone(pid), "child outlived its deadline"
    assert not os.path.exists(marker)


@pytest.mark.tonio
async def test_a_child_ignoring_sigterm_is_escalated_to_sigkill():
    pid_path, marker = _scratch()
    script = f"echo $$ > {pid_path}; trap '' TERM; sleep 60; touch {marker}"
    previous = process_module._KILL_ESCALATION_S
    process_module._KILL_ESCALATION_S = 0.05
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            await run_command(["sh", "-c", script], capture_output=True, timeout=0.05)

        pid = await _child_pid(pid_path)
        # SIGTERM is trapped away, so only the escalation can end this.
        assert await _wait_gone(pid), "SIGTERM was ignored and never escalated"
        assert not os.path.exists(marker)
    finally:
        process_module._KILL_ESCALATION_S = previous


@pytest.mark.tonio
async def test_timeout_reports_the_command_and_drains_what_it_has():
    with pytest.raises(subprocess.TimeoutExpired) as raised:
        await run_command(["sh", "-c", "echo early; sleep 60"], capture_output=True, timeout=0.05)
    assert raised.value.cmd == ["sh", "-c", "echo early; sleep 60"]
    assert raised.value.timeout == 0.05
    # The contract is that the drains are joined rather than dropped, so the
    # fields are populated. Whether `early` won the race against a 50 ms
    # deadline is not the contract, and asserting it would be a timing bet.
    assert raised.value.output is not None
    assert raised.value.stderr is not None


@pytest.mark.tonio
async def test_rejects_contradictory_redirection_arguments():
    with pytest.raises(ValueError):
        await run_command(["true"], capture_output=True, stdout=subprocess.DEVNULL)
    with pytest.raises(ValueError):
        await run_command(["true"], input=b"x", stdin=subprocess.DEVNULL)
