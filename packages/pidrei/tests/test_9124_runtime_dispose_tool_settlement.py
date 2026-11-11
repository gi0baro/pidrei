"""Mirror of pi's suite/regressions/9124-runtime-dispose-tool-settlement.test.ts.

pi's tool rejects 10 ms after its abort signal fires (`setTimeout`); here it raises
once the cancel token fires. The tool is a `ToolDefinition`, which the harness
registers.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session_runtime import AgentSessionRuntime
from pidrei.core.agent_session_services import AgentSessionServices
from pidrei.core.extensions import ToolDefinition
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call

from .harness import create_harness, get_message_text


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def _create_runtime(harnesses, tool: ToolDefinition):
    harness = await create_harness(tools=[tool])
    harnesses.append(harness)
    services = AgentSessionServices(
        cwd=harness.temp_dir,
        agent_dir=harness.temp_dir,
        model_runtime=harness.session.model_runtime,
        settings_manager=harness.settings_manager,
        resource_loader=harness.session.resource_loader,
    )

    async def unused_factory(**_kwargs):
        raise Exception("runtime replacement is not used in this test")

    return harness, AgentSessionRuntime(harness.session, services, unused_factory)


@pytest.mark.tonio
async def test_persists_the_aborted_tool_turn_before_disconnecting_the_session(harnesses):
    tool_started = tonio.Event()

    async def execute(_tool_call_id, _params, cancel=None, _on_update=None, _ctx=None):
        tool_started.set()
        await cancel.event.wait()
        raise Exception("Operation aborted")

    tool = ToolDefinition(
        name="wait",
        label="Wait",
        description="Wait until aborted",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    harness, runtime = await _create_runtime(harnesses, tool)
    harness.set_responses([faux_assistant_message(faux_tool_call("wait", {}), stop_reason="toolUse")])

    prompt = tonio.spawn(runtime.session.prompt("run tool"))
    await tool_started.wait(5)
    assert tool_started.is_set(), "the tool never started"
    try:
        await runtime.dispose()
        # #9124: disposal itself must settle persistence; awaiting the prompt must not be necessary.
        messages = [
            entry["message"]
            for entry in harness.session_manager.get_entries()
            if entry["type"] == "message" and entry["message"].role != "system"
        ]
        assert [message.role for message in messages] == ["user", "assistant", "toolResult", "assistant"]
        assert messages[2].role == "toolResult"
        assert get_message_text(messages[2]) == "Operation aborted"
        assert messages[2].is_error is True
    finally:
        await prompt
