"""Mirror of pi's suite/regressions/8409-aborted-tool-auth-setup.test.ts.

pi resolves the blocking tool from the abort signal's event listener; here the
tool awaits its `CancelToken` and returns once it fires. pi points
`agent.streamFunction` at `modelRuntime.streamSimple`; the harness's stream
function already routes there.
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
async def test_ends_the_turn_as_aborted_when_auth_setup_sees_an_already_aborted_signal(harnesses):
    tool_started = tonio.Event()

    async def execute(_tool_call_id, _params, cancel=None, *_rest):
        tool_started.set()
        await cancel.wait(5)
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
    harness.set_responses([faux_assistant_message(faux_tool_call("block", {}), stop_reason="toolUse")])

    prompt = tonio.spawn(harness.session.prompt("run the blocking tool"))
    await tool_started.wait(5)
    assert tool_started.is_set(), "timed out waiting for the blocking tool to start"
    await harness.session.abort()
    await prompt

    final_assistant = next(
        (message for message in reversed(harness.session.messages) if message.role == "assistant"), None
    )
    assert final_assistant is not None, "Expected final assistant message"
    assert final_assistant.stop_reason == "aborted"
