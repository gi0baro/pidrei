"""Mirror of pi's suite/regressions/8724-in-memory-fork-active-tool.test.ts.

pi resolves the blocking tool from the abort signal's event listener; here the
tool awaits its `CancelToken` and returns once it fires.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.agent_session_runtime import AgentSessionRuntime, CreateAgentSessionRuntimeResult
from pidrei.core.agent_session_services import (
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    create_agent_session_from_services,
)
from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness


@pytest.mark.tonio
async def test_does_not_append_the_aborted_turn_to_the_replacement_session():
    tool_started = tonio.Event()

    async def execute(_tool_call_id, _params, cancel=None, *_rest):
        tool_started.set()
        await cancel.wait()
        return AgentToolResult(content=[TextContent(text="tool aborted")], details={})

    blocking_tool = ToolDefinition(
        name="block",
        label="Block",
        description="Wait until aborted",
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )
    harness = await create_harness(tools=[blocking_tool])
    services = AgentSessionServices(
        cwd=harness.temp_dir,
        agent_dir=harness.temp_dir,
        model_runtime=harness.session.model_runtime,
        settings_manager=harness.settings_manager,
        resource_loader=harness.session.resource_loader,
        diagnostics=[],
    )

    async def create_runtime(*, session_manager, session_start_event=None, **_kwargs):
        result = await create_agent_session_from_services(
            CreateAgentSessionFromServicesOptions(
                services=services,
                session_manager=session_manager,
                session_start_event=session_start_event,
                model=harness.get_model(),
                no_tools="all",
            )
        )
        return CreateAgentSessionRuntimeResult(session=result.session, services=services, diagnostics=[])

    runtime = AgentSessionRuntime(harness.session, services, create_runtime)
    try:
        harness.set_responses(
            [
                faux_assistant_message("first response"),
                faux_assistant_message(faux_tool_call("block", {}), stop_reason="toolUse"),
                faux_assistant_message("unused after abort"),
            ]
        )
        await runtime.session.prompt("first prompt")
        first_user_entry_id = runtime.session.get_user_messages_for_forking()[0]["entryId"]
        assert first_user_entry_id is not None

        outgoing_prompt = tonio.spawn(runtime.session.prompt("start blocking tool"))
        await tool_started.wait(5)
        assert tool_started.is_set(), "timed out waiting for the blocking tool to start"
        fork_result = await runtime.fork(first_user_entry_id)
        await outgoing_prompt
        await runtime.session.bind_extensions(ExtensionBindings())

        assert fork_result == {"cancelled": False, "selectedText": "first prompt"}
        assert runtime.session.messages == []
        assert [entry for entry in runtime.session.session_manager.get_entries() if entry["type"] == "message"] == []

        captured_roles: list[str] = []

        async def respond(context, *_rest):
            captured_roles.extend(message.role for message in context.messages)
            return faux_assistant_message("next response")

        harness.set_responses([respond])
        await runtime.session.prompt("next prompt")

        assert captured_roles == ["user"]
    finally:
        if runtime.session is not harness.session:
            await runtime.dispose()
        harness.cleanup()
