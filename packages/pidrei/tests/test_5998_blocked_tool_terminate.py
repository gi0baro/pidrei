"""Mirror of pi's suite/regressions/5998-blocked-tool-terminate.test.ts."""

import pytest

from pidrei.core.extensions import ToolDefinition
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call

from .harness import create_harness, get_assistant_texts


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.mark.tonio
async def test_lets_a_tool_call_handler_terminate_the_run_after_blocking_execution(harnesses):
    async def execute(*_args):
        raise Exception("tool should have been blocked")

    echo_tool = ToolDefinition(
        name="echo",
        label="Echo",
        description="Echo text back",
        parameters={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        execute=execute,
    )

    def factory(pi) -> None:
        async def guard(_event, _ctx):
            return {"block": True, "reason": "Blocked by terminating policy", "terminate": True}

        pi.on("tool_call", guard)

    harness = await create_harness(tools=[echo_tool], extension_factories=[factory])
    harnesses.append(harness)
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("echo", {"text": "hello"})], stop_reason="toolUse"),
            faux_assistant_message("should not run"),
        ]
    )

    await harness.session.prompt("hi")

    assert harness.get_pending_response_count() == 1
    assert "should not run" not in get_assistant_texts(harness)
    assert harness.events_of_type("tool_execution_end")[0].result.terminate is True
    assert any(
        getattr(message, "role", None) == "toolResult" and message.is_error for message in harness.session.messages
    )
