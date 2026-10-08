"""Partial mirror of pi's suite/agent-session-bash-persistence.test.ts.

Only the concurrent-bash cases added in 0.83.0 (#7103) and the split escape
sequence cases added in 1.1.0 (#10504) are mirrored here; the rest of pi's
bash/persistence characterization suite is an open parity gap (see
scripts/upstream_diff.py TEST_HOMES).
"""

import threading
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from .agent_session_helpers import abortable_stream_fn, create_agent_session


class ControlledBashInvocation:
    """pi's `ControlledBashInvocation`: exposes the abort signal and a `finish`
    that lets the test settle the exec at will."""

    def __init__(self, cancel):
        self.cancel = cancel
        self._done = tonio.Event()

    def finish(self) -> None:
        self._done.set()


class ControlledBashOperations:
    def __init__(self):
        self.invocations: list[ControlledBashInvocation] = []
        self._lock = threading.Lock()
        self._waiters: list[tuple[int, tonio.Event]] = []

    async def exec(self, _command, _cwd, *, on_data=None, cancel=None):
        invocation = ControlledBashInvocation(cancel)
        with self._lock:
            self.invocations.append(invocation)
            reached = [waiter for waiter in self._waiters if len(self.invocations) >= waiter[0]]
        for _count, event in reached:
            event.set()
        await invocation._done.wait()
        return SimpleNamespace(exit_code=0)


async def _wait_for_invocations(operations: ControlledBashOperations, count: int) -> None:
    """Wait until the stub exec has been reached `count` times (each exec
    signals as it records itself)."""
    reached = tonio.Event()
    with operations._lock:
        if len(operations.invocations) >= count:
            return
        operations._waiters.append((count, reached))
    await reached.wait(5)
    assert reached.is_set(), f"stub exec reached {len(operations.invocations)} of {count} times"


@pytest.mark.tonio
async def test_keeps_newer_bash_execution_tracked_when_an_older_execution_finishes(tmp_path):
    session = await create_agent_session(tmp_path, stream_fn=abortable_stream_fn)
    operations = ControlledBashOperations()

    # pi's executeBash reaches the stub exec synchronously, so invocation order
    # matches call order; spawned tasks race here, so each start is awaited.
    first_bash = tonio.spawn(session.execute_bash("first", None, {"operations": operations}))
    await _wait_for_invocations(operations, 1)
    second_bash = tonio.spawn(session.execute_bash("second", None, {"operations": operations}))
    await _wait_for_invocations(operations, 2)

    operations.invocations[0].finish()
    first_result = await first_bash
    running_after_first_settles = session.is_bash_running

    session.abort_bash()
    second_was_aborted = operations.invocations[1].cancel.cancelled
    operations.invocations[1].finish()
    second_result = await second_bash

    assert first_result.cancelled is False
    assert running_after_first_settles is True
    assert second_was_aborted is True
    assert second_result.cancelled is True
    assert session.is_bash_running is False


@pytest.mark.tonio
async def test_aborts_all_active_bash_executions(tmp_path):
    session = await create_agent_session(tmp_path, stream_fn=abortable_stream_fn)
    operations = ControlledBashOperations()

    first_bash = tonio.spawn(session.execute_bash("first", None, {"operations": operations}))
    await _wait_for_invocations(operations, 1)
    second_bash = tonio.spawn(session.execute_bash("second", None, {"operations": operations}))
    await _wait_for_invocations(operations, 2)

    session.abort_bash()
    aborted_signals = [invocation.cancel.cancelled for invocation in operations.invocations]
    for invocation in operations.invocations:
        invocation.finish()
    results = [await first_bash, await second_bash]

    assert aborted_signals == [True, True]
    assert [result.cancelled for result in results] == [True, True]
    assert session.is_bash_running is False


# Regression tests for pi #10504: escape sequences split across output chunks.


async def _run_chunks(tmp_path, chunks: list[str | bytes], before_exit=None) -> SimpleNamespace:
    """Run a user bash command whose fake shell emits the given chunks; `before_exit` sees what was streamed by then."""
    session = await create_agent_session(tmp_path, stream_fn=abortable_stream_fn)
    deltas: list[str] = []

    class _Operations:
        async def exec(self, _command, _cwd, *, on_data=None, cancel=None):
            for chunk in chunks:
                on_data(chunk.encode() if isinstance(chunk, str) else chunk)
            if before_exit is not None:
                before_exit("".join(deltas))
            return SimpleNamespace(exit_code=0)

    result = await session.execute_bash("custom", deltas.append, {"operations": _Operations()})
    recorded = session.messages[-1]
    return SimpleNamespace(
        output=result.output,
        streamed="".join(deltas),
        recorded=recorded.output if recorded.role == "bashExecution" else None,
    )


@pytest.mark.tonio
async def test_strips_a_color_reset_split_inside_its_parameters(tmp_path):
    run = await _run_chunks(tmp_path, ["\x1b[31mERROR: file.py:1\x1b[0", "m\n"])
    assert run.output == "ERROR: file.py:1\n"
    assert run.streamed == "ERROR: file.py:1\n"
    assert run.recorded == "ERROR: file.py:1\n"


@pytest.mark.tonio
async def test_strips_a_color_code_split_right_after_esc(tmp_path):
    run = await _run_chunks(tmp_path, ["before\x1b", "[32mafter\n"])
    assert run.output == "beforeafter\n"
    assert run.streamed == "beforeafter\n"


@pytest.mark.tonio
async def test_strips_an_osc_sequence_split_before_its_terminator(tmp_path):
    run = await _run_chunks(tmp_path, ["a\x1b]0;window ", "title\x1b", "\\b\n"])
    assert run.output == "ab\n"


@pytest.mark.tonio
async def test_flushes_an_incomplete_multi_byte_character_at_the_end_of_output(tmp_path):
    run = await _run_chunks(tmp_path, ["ok", "\u00e9".encode()[:1]])
    assert run.output == "ok\ufffd"
    assert run.streamed == "ok\ufffd"


@pytest.mark.tonio
async def test_does_not_hold_back_output_behind_a_long_unterminated_sequence(tmp_path):
    long = "x" * 300
    streamed_before_exit: list[str] = []
    await _run_chunks(tmp_path, [f"\x1b]{long}"], streamed_before_exit.append)
    assert streamed_before_exit == [f"]{long}"]
