"""pidrei-only: every exit of a started run hands the session back.

pi claims the run before its preparation (`_startRun`), and leaves the session
busy when a flush or `onStarted` throws after the preparation, or when a flush
throws while it cleans up a failed preparation. pidrei releases the claim on
every exit, including a cancelled caller, where every await on the unwinding
chain raises again.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import PromptOptions
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness


_WAIT_S = 5


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def _send_note_while_starting(pi) -> None:
    """A before_agent_start handler that queues a custom message into the starting run."""

    async def on_before_agent_start(_event, _ctx):
        pi.send_message({"customType": "note", "content": "held while starting"}, {"triggerTurn": False})

    pi.on("before_agent_start", on_before_agent_start)


def _fail_custom_message_writes(session) -> None:
    async def append_custom_message_entry(*_args, **_kwargs):
        raise OSError("session write failed")

    session.session_manager.append_custom_message_entry = append_custom_message_entry


def _idle_event(session) -> tonio.Event:
    idle = tonio.Event()

    async def wait() -> None:
        await session.wait_for_idle()
        idle.set()

    tonio.spawn.without_tracking(wait())
    return idle


async def _assert_session_takes_a_new_prompt(harness) -> None:
    harness.append_responses([faux_assistant_message("again")])
    await harness.session.prompt("again")
    assert harness.session.messages[-1].content == [TextContent(text="again")]


def _custom_entries(session) -> list:
    return [entry["customType"] for entry in session.session_manager.get_entries() if entry["type"] == "custom_message"]


@pytest.mark.tonio
async def test_a_flush_failing_after_the_preparation_hands_the_session_back(harnesses):
    async def factory(pi) -> None:
        _send_note_while_starting(pi)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    write = harness.session.session_manager.append_custom_message_entry
    _fail_custom_message_writes(harness.session)

    with pytest.raises(OSError, match="session write failed"):
        await harness.session.prompt("hello")

    assert harness.session.is_idle
    harness.session.session_manager.append_custom_message_entry = write
    await _assert_session_takes_a_new_prompt(harness)


@pytest.mark.tonio
async def test_a_failing_started_callback_hands_the_session_back(harnesses):
    harness = await create_harness()
    harnesses.append(harness)

    def preflight_result(disposition):
        if disposition == "started":
            raise RuntimeError("preflight failed")

    with pytest.raises(RuntimeError, match="preflight failed"):
        await harness.session.prompt("hello", PromptOptions(preflight_result=preflight_result))

    assert harness.session.is_idle
    await _assert_session_takes_a_new_prompt(harness)


@pytest.mark.tonio
async def test_a_flush_failing_after_a_failed_preparation_still_releases_idle_waiters(harnesses):
    waiters: list[tonio.Event] = []
    parked = tonio.Event()

    async def factory(pi) -> None:
        _send_note_while_starting(pi)

        async def wait_for_idle(_event, _ctx):
            waiters.append(_idle_event(harness.session))
            # The waiter is parked on the session before the preparation fails.
            await parked.wait(_WAIT_S)

        pi.on("before_agent_start", wait_for_idle)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    get_idle_wait_event = harness.session._get_idle_wait_event

    def record_parked():
        event = get_idle_wait_event()
        parked.set()
        return event

    harness.session._get_idle_wait_event = record_parked
    _fail_custom_message_writes(harness.session)

    def fail_preparation(*_args, **_kwargs):
        raise RuntimeError("preparation failed")

    harness.session._prepare_prompt_and_tool_loadout = fail_preparation

    with pytest.raises(OSError, match="session write failed") as caught:
        await harness.session.prompt("hello")

    # The preparation's error stays in the chain.
    chain = []
    error = caught.value
    while error is not None:
        chain.append(error)
        error = error.__context__
    assert any(isinstance(error, RuntimeError) and str(error) == "preparation failed" for error in chain)
    assert parked.is_set()
    await waiters[0].wait(_WAIT_S)
    assert waiters[0].is_set()
    assert harness.session.is_idle


@pytest.mark.tonio
async def test_a_caller_cancelled_during_the_preparation_hands_the_session_back(harnesses):
    started = tonio.Event()
    release = tonio.Event()

    async def factory(pi) -> None:
        _send_note_while_starting(pi)

        async def park(_event, _ctx):
            started.set()
            await release.wait(_WAIT_S)

        pi.on("before_agent_start", park)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)

    async with tonio.scope(cancel_on_exc=True) as scope:
        scope.spawn(harness.session.prompt("hello"))
        await started.wait(_WAIT_S)
        assert started.is_set()
        scope.cancel()
    release.set()

    idle = _idle_event(harness.session)
    await idle.wait(_WAIT_S)
    assert idle.is_set()
    # The message held for the run that never started is persisted, as when idle.
    assert _custom_entries(harness.session) == ["note"]
    assert harness.events_of_type("agent_settled") == []
    await _assert_session_takes_a_new_prompt(harness)


@pytest.mark.tonio
async def test_a_caller_cancelled_during_the_run_settles_it_as_aborted(harnesses):
    started = tonio.Event()
    release = tonio.Event()

    async def execute(*_args):
        started.set()
        await release.wait(_WAIT_S)
        return AgentToolResult(content=[TextContent(text="done")], details={})

    tool = ToolDefinition(
        name="park", label="park", description="park", parameters={"type": "object", "properties": {}}, execute=execute
    )
    harness = await create_harness(tools=[tool])
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message(faux_tool_call("park", {}), stop_reason="toolUse")])

    async with tonio.scope(cancel_on_exc=True) as scope:
        scope.spawn(harness.session.prompt("hello"))
        await started.wait(_WAIT_S)
        assert started.is_set()
        scope.cancel()
    # The aborted run winds down once its tool returns.
    release.set()

    idle = _idle_event(harness.session)
    await idle.wait(_WAIT_S)
    assert idle.is_set()
    assert [event.aborted for event in harness.events_of_type("agent_settled")] == [True]
    await _assert_session_takes_a_new_prompt(harness)
