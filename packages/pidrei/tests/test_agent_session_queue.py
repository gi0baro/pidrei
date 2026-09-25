"""Partial mirror of pi's suite/agent-session-queue.test.ts.

Holds the cases ported with 0.87.1 (the rest of pi's characterization suite is
a recorded parity gap — see the classifier's TEST_HOMES). pi's
`createWaitingHarness` becomes `_create_waiting_harness`: a `wait` tool parked
on a `tonio.Event`, the prompt spawned, and tool start observed through a
session listener.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def _create_waiting_harness(*, extension_factories=None):
    tool_release = tonio.Event()
    tool_started = tonio.Event()

    async def execute(*_args):
        await tool_release.wait()
        return AgentToolResult(content=[TextContent(text="released")], details={})

    wait_tool = ToolDefinition(
        name="wait",
        label="Wait",
        description="Wait for release",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    harness = await create_harness(tools=[wait_tool], extension_factories=extension_factories)

    def on_event(event) -> None:
        if event.type == "tool_execution_start" and event.tool_name == "wait":
            tool_started.set()

    harness.session.subscribe(on_event)
    return harness, tool_release, tool_started


# Regression test for #8718.
@pytest.mark.tonio
async def test_runs_direct_steering_and_follow_up_messages_through_input_handlers(harnesses):
    input_events: list[dict] = []

    async def factory(pi) -> None:
        async def on_input(event, _ctx):
            input_events.append(
                {"text": event["text"], "source": event["source"], "streamingBehavior": event["streamingBehavior"]}
            )
            if event["text"].startswith("handle"):
                return {"action": "handled"}
            return {"action": "transform", "text": f"transformed: {event['text']}"}

        pi.on("input", on_input)

    harness, tool_release, tool_started = await _create_waiting_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("wait", {}), stop_reason="toolUse"),
            faux_assistant_message("steered"),
            faux_assistant_message("followed up"),
        ]
    )
    prompt = tonio.spawn(harness.session.prompt("start"))

    await tool_started.wait(5)
    assert tool_started.is_set(), "the wait tool never started"
    input_events.clear()
    try:
        await harness.session.steer("steer me", None, source="rpc")
        await harness.session.steer("handle steer", None, source="rpc")
        await harness.session.follow_up("follow me", None, source="rpc")
        await harness.session.follow_up("handle follow", None, source="rpc")

        assert input_events == [
            {"text": "steer me", "source": "rpc", "streamingBehavior": "steer"},
            {"text": "handle steer", "source": "rpc", "streamingBehavior": "steer"},
            {"text": "follow me", "source": "rpc", "streamingBehavior": "followUp"},
            {"text": "handle follow", "source": "rpc", "streamingBehavior": "followUp"},
        ]
        assert harness.session.get_steering_messages() == ["transformed: steer me"]
        assert harness.session.get_follow_up_messages() == ["transformed: follow me"]
    finally:
        tool_release.set()
    await prompt


# pidrei-only: in pi the final queue check and the end of the run are one
# synchronous step. Here the check is a mailbox job, so input can see the run
# between the check answering "nothing queued" and the run ending. It must not be
# queued into a run that will never drain it; it takes the idle path once the run
# has settled, as it would in pi arriving just after the run.
@pytest.mark.tonio
async def test_input_seeing_the_run_after_its_final_queue_check_is_not_stranded(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("done")])
    session = harness.session
    late: list = []
    run_before_settle_boundary = session._run_before_settle_boundary

    async def boundary_then_late_input() -> bool:
        should_continue = await run_before_settle_boundary()
        if not should_continue and not late:
            remainder = session._dispatch_custom_message({"customType": "late", "content": "late input"}, {})
            late.append(tonio.spawn(remainder()) if remainder is not None else None)
        return should_continue

    session._run_before_settle_boundary = boundary_then_late_input

    await session.prompt("start")
    if late[0] is not None:
        await late[0]

    assert await session.agent.has_queued_messages() is False
    assert [
        entry.get("customType") for entry in session.session_manager.get_entries() if entry["type"] == "custom_message"
    ] == ["late"]
