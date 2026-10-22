"""Mirror of pi coding-agent test/suite/regressions/11227-recovery-omits-tool-results.test.ts.

https://github.com/earendil-works/pi/issues/11227: recovery omits tool results with their tool call.
"""

from dataclasses import replace

import pytest

from pidrei.core.compaction import CompactionResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent, ToolResultMessage, Usage, UsageCost, UserMessage
from pidrei_utils import clock

from .harness import create_harness


async def _summary_factory(pi) -> None:
    async def on_before_compact(event, _ctx):
        preparation = event["preparation"]
        return {
            "compaction": CompactionResult(
                summary="summary",
                first_kept_entry_id=preparation.first_kept_entry_id,
                tokens_before=preparation.tokens_before,
                details={},
            )
        }

    pi.on("session_before_compact", on_before_compact)


@pytest.mark.tonio
async def test_pre_prompt_recovery_of_a_resumed_length_stop_omits_the_truncated_call_and_its_result():
    harness = await create_harness(
        models=[{"id": "faux-1", "context_window": 100_000, "max_tokens": 1_000}],
        settings={"compaction": {"enabled": True}},
        extension_factories=[_summary_factory],
    )
    try:
        now = clock.now_ms()
        model = harness.get_model()
        await harness.session_manager.append_message(UserMessage(content="check status", timestamp=now - 1000))
        truncated = replace(
            faux_assistant_message(
                [faux_tool_call("bash", {"command": "echo"}, id="call_truncated")],
                stop_reason="length",
                timestamp=now - 500,
            ),
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(input=100, output=16, cache_read=0, cache_write=0, total_tokens=116, cost=UsageCost()),
        )
        await harness.session_manager.append_message(truncated)
        await harness.session_manager.append_message(
            ToolResultMessage(
                tool_call_id="call_truncated",
                tool_name="bash",
                is_error=True,
                content=[TextContent(text="Not executed: output token limit reached.")],
                timestamp=now - 400,
            )
        )
        harness.session.agent.state.messages = harness.session_manager.build_session_context().messages

        request_messages: list = []

        async def respond(context, *_rest):
            request_messages.extend(context.messages)
            return faux_assistant_message("done")

        harness.set_responses([respond])

        await harness.session.prompt("continue")

        assert "toolResult" not in [message.role for message in request_messages]
        assert not any(message.role == "assistant" and message.stop_reason == "length" for message in request_messages)
    finally:
        harness.cleanup()


# pi's "A context edit replaces the projected copy" branch. The caller holds the original assistant
# message while the context holds the edited copy, so the copy is found by its source entry.
@pytest.mark.tonio
async def test_omits_the_tool_results_of_a_context_edited_attempt_held_as_the_original():
    harness = await create_harness()
    try:
        now = clock.now_ms()
        await harness.session_manager.append_message(UserMessage(content="check status", timestamp=now - 1000))
        truncated = faux_assistant_message(
            [faux_tool_call("bash", {"command": "echo"}, id="call_truncated")],
            stop_reason="length",
            timestamp=now - 500,
        )
        truncated_id = await harness.session_manager.append_message(truncated)
        await harness.session_manager.append_message(
            ToolResultMessage(
                tool_call_id="call_truncated",
                tool_name="bash",
                is_error=True,
                content=[TextContent(text="Not executed: output token limit reached.")],
                timestamp=now - 400,
            )
        )
        await harness.session_manager.append_context_edit(truncated_id, {"content": "edited truncated response"})
        harness.session.agent.state.messages = harness.session_manager.build_session_context().messages
        assert not any(message is truncated for message in harness.session.agent.state.messages)

        await harness.session._omit_recovery_attempt(truncated)

        roles = [message.role for message in harness.session_manager.build_session_context().messages]
        assert roles == ["user"]
    finally:
        harness.cleanup()
