"""Mirror of pi's suite/regressions/8935-parallel-preflight-abort.test.ts."""

import pytest

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness, get_message_text


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.mark.tonio
async def test_does_not_start_prepared_tools_after_a_later_preflight_aborts(harnesses):
    executions: list[str] = []
    preflights: list[str] = []
    result_hooks: list[str] = []

    async def execute(_tool_call_id, params, *_rest):
        value = str(params["value"]) if isinstance(params, dict) and "value" in params else ""
        executions.append(value)
        return AgentToolResult(content=[TextContent(text=value)], details={"value": value})

    external_write = ToolDefinition(
        name="external_write",
        label="External write",
        description="Perform an external write",
        parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
        execute=execute,
    )

    def factory(pi) -> None:
        async def on_tool_call(event, ctx):
            value = str(event["input"]["value"]) if "value" in event["input"] else ""
            preflights.append(value)
            if value == "second":
                ctx.abort()

        async def on_tool_result(event, _ctx):
            result_hooks.append(event["toolCallId"])

        pi.on("tool_call", on_tool_call)
        pi.on("tool_result", on_tool_result)

    harness = await create_harness(tools=[external_write], extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("external_write", {"value": "first"}),
                    faux_tool_call("external_write", {"value": "second"}),
                ],
                stop_reason="toolUse",
            ),
        ]
    )

    await harness.session.prompt("run both writes")

    assert preflights == ["first", "second"]
    assert executions == []
    assert result_hooks == []

    starts = harness.events_of_type("tool_execution_start")
    ends = harness.events_of_type("tool_execution_end")
    assert len(starts) == 2
    assert len(ends) == 2
    assert {event.tool_call_id for event in ends} == {event.tool_call_id for event in starts}
    assert all(event.is_error for event in ends)

    tool_results = [message for message in harness.session.messages if message.role == "toolResult"]
    assert [message.tool_call_id for message in tool_results] == [event.tool_call_id for event in starts]
    assert [get_message_text(message) for message in tool_results] == ["Operation aborted", "Operation aborted"]
