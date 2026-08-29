"""Partial mirror of pi's suite/agent-session-bash-persistence.test.ts.

Only the concurrent-bash cases added in 0.83.0 (#7103) are mirrored here; the
rest of pi's bash/persistence characterization suite is an open parity gap
(see scripts/upstream_diff.py TEST_HOMES).
"""

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

    async def exec(self, _command, _cwd, *, on_data=None, cancel=None):
        invocation = ControlledBashInvocation(cancel)
        self.invocations.append(invocation)
        await invocation._done.wait()
        return SimpleNamespace(exit_code=0)


async def _wait_for_invocations(operations: ControlledBashOperations, count: int) -> None:
    while len(operations.invocations) < count:
        await tonio.time.sleep(0.005)


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
