"""Mirror of pi's suite/regressions/12124-websocket-connection-limit-retry.test.ts.

pi's faux response callables take `(context)`; pidrei's take
`(context, options, state, model)`, so the one here absorbs the extra arguments.
"""

import pytest

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_thinking, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness, get_assistant_texts


ERROR_MESSAGE = (
    "Codex error: Responses websocket connection limit reached (60 minutes). Create a new websocket connection "
    "to continue."
)


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


# Regression for #12124: the limit can hit mid-stream after partial thinking, where the provider cannot reconnect.
@pytest.mark.tonio
async def test_retries_the_turn_without_the_failed_partial_response_or_re_running_completed_tools(harnesses):
    tool_runs = 0

    async def execute(*_args):
        nonlocal tool_runs
        tool_runs += 1
        return AgentToolResult(content=[TextContent(text="completed once")], details={})

    record_tool = ToolDefinition(
        name="record",
        label="Record",
        description="Record one operation",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    harness = await create_harness(
        settings={"retry": {"enabled": True, "maxRetries": 2, "baseDelayMs": 0}}, tools=[record_tool]
    )
    harnesses.append(harness)
    retry_messages: list = []

    async def recovered(context, *_rest):
        retry_messages.extend(context.messages)
        return faux_assistant_message("recovered")

    harness.set_responses(
        [
            faux_assistant_message(faux_tool_call("record", {}, id="completed_call"), stop_reason="toolUse"),
            faux_assistant_message(
                [faux_thinking("unfinished thinking")], stop_reason="error", error_message=ERROR_MESSAGE
            ),
            recovered,
        ]
    )

    await harness.session.prompt("test")

    assert harness.faux.state.call_count == 3
    assert tool_runs == 1
    assert not any(message.role == "assistant" and message.stop_reason == "error" for message in retry_messages)
    starts = harness.events_of_type("auto_retry_start")
    assert [(event.attempt, event.error_message) for event in starts] == [(1, ERROR_MESSAGE)]
    ends = harness.events_of_type("auto_retry_end")
    assert [(event.success, event.attempt) for event in ends] == [(True, 1)]
    assert get_assistant_texts(harness)[-1] == "recovered"
