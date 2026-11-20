"""Mirror of pi's suite/regressions/12119-copy-after-abort-or-error.test.ts.

Regression test for https://github.com/earendil-works/pi/issues/12119

/copy uses get_last_assistant_text(), which took the last assistant message and skipped only aborted
messages with no content. After Esc during a tool call the last messages are a tool-call-only message
and an empty one; after a provider error it is an empty error message. Neither has text, so /copy said
"No agent messages to copy yet." while the previous answer was on screen.

pi's blocking tool resolves on its signal's `abort` listener; here it waits on the cancel token's
Event. The tool is a `ToolDefinition`, which the harness registers.
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


@pytest.mark.tonio
async def test_returns_the_previous_answer_after_esc_during_a_tool_call(harnesses):
    tool_started = tonio.Event()

    async def execute(_tool_call_id, _params, cancel=None, _on_update=None, _ctx=None):
        tool_started.set()
        await cancel.event.wait()
        return AgentToolResult(content=[TextContent(text="tool aborted")], details={})

    blocking_tool = ToolDefinition(
        name="block",
        label="Block",
        description="Blocks until aborted",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    harness = await create_harness(tools=[blocking_tool])
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("Here is the answer."),
            faux_assistant_message(faux_tool_call("block", {}), stop_reason="toolUse"),
        ]
    )

    await harness.session.prompt("first question")
    run = tonio.spawn(harness.session.prompt("run the blocking tool"))
    await tool_started.wait(5)
    assert tool_started.is_set(), "the tool never started"
    await harness.session.abort()
    await run

    assert harness.session.get_last_assistant_text() == "Here is the answer."


@pytest.mark.tonio
async def test_returns_the_previous_answer_after_a_provider_error(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message("Here is the answer."),
            faux_assistant_message([], stop_reason="error", error_message="400 invalid_request_error"),
        ]
    )

    await harness.session.prompt("first question")
    await harness.session.prompt("second question")

    assert harness.session.get_last_assistant_text() == "Here is the answer."


@pytest.mark.tonio
async def test_keeps_a_reply_that_ended_normally_without_text_as_the_last_reply(harnesses):
    harness = await create_harness()
    harnesses.append(harness)
    harness.set_responses([faux_assistant_message("Here is the answer."), faux_assistant_message([])])

    await harness.session.prompt("first question")
    await harness.session.prompt("second question")

    assert harness.session.get_last_assistant_text() is None
