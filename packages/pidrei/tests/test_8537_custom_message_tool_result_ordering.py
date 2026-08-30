"""Mirror of pi's suite/regressions/8537-custom-message-tool-result-ordering.test.ts."""

import pytest

from pidrei.core.extensions import ToolDefinition
from pidrei.core.messages import convert_to_llm
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def _roles(messages: list) -> list[str]:
    return [message.role for message in messages]


async def _create_notifying_harness(harnesses: list):
    """A harness with a tool that, mid-execution, sends a context-only custom
    message (e.g. a subagent reply notifying the session)."""
    sessions: list = []

    async def execute(*_args):
        await sessions[0].send_custom_message(
            {"customType": "subagent-reply", "content": "subagent replied", "display": True},
            {"triggerTurn": False},
        )
        return AgentToolResult(content=[TextContent(text="tool done")], details={})

    slow_tool = ToolDefinition(
        name="wait",
        label="Wait",
        description="Wait for a background task",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )

    harness = await create_harness(tools=[slow_tool])
    harnesses.append(harness)
    sessions.append(harness.session)
    return harness


@pytest.mark.tonio
async def test_appends_the_message_after_the_turns_tool_results_instead_of_between_call_and_result(harnesses):
    harness = await _create_notifying_harness(harnesses)
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("wait", {}), stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("hi")

    assert _roles(harness.session.messages) == ["user", "assistant", "toolResult", "custom", "assistant"]


@pytest.mark.tonio
async def test_keeps_session_entries_and_message_events_in_the_same_order_as_agent_state(harnesses):
    harness = await _create_notifying_harness(harnesses)
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("wait", {}), stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("hi")

    entry_kinds = []
    for entry in harness.session_manager.get_branch():
        if entry.get("type") == "message":
            entry_kinds.append(entry["message"].role)
        elif entry.get("type") == "custom_message":
            entry_kinds.append("custom")
    assert entry_kinds == ["user", "assistant", "toolResult", "custom", "assistant"]

    # message events must never describe a message the session tree does not contain yet
    message_starts = [event.message.role for event in harness.events if getattr(event, "type", None) == "message_start"]
    assert message_starts == ["user", "assistant", "toolResult", "custom", "assistant"]


@pytest.mark.tonio
async def test_produces_an_llm_history_where_every_tool_result_follows_its_tool_call(harnesses):
    harness = await _create_notifying_harness(harnesses)
    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("wait", {}), stop_reason="toolUse"),
            faux_assistant_message("done"),
            faux_assistant_message("second turn"),
        ]
    )

    await harness.session.prompt("hi")
    await harness.session.prompt("and now?")

    llm_messages = convert_to_llm(harness.session.messages)
    open_tool_call_ids: set[str] = set()
    for message in llm_messages:
        if message.role == "assistant":
            open_tool_call_ids.clear()
            for block in message.content:
                if block.type == "toolCall":
                    open_tool_call_ids.add(block.id)
            continue
        if message.role == "toolResult":
            assert message.tool_call_id in open_tool_call_ids
            open_tool_call_ids.discard(message.tool_call_id)
            continue
        open_tool_call_ids.clear()
