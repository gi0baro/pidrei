"""Mirror of pi's context-estimate.test.ts (the estimateContextTokens cases).

The `buildBaseOptions` assertion from the original suite lands with the
simple-options port (PLAN.md Phase 1).
"""

from pidrei_ai.types import AssistantMessage, Context, TextContent, ToolCall, Usage, UserMessage
from pidrei_ai.utils.estimate import ContextUsageEstimate, estimate_context_tokens, estimate_message_tokens


def create_assistant(timestamp: int, total_tokens: int) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="kept")],
        api="openai-responses",
        provider="openai",
        model="test-model",
        usage=Usage(input=total_tokens, total_tokens=total_tokens),
        stop_reason="stop",
        timestamp=timestamp,
    )


def test_ignores_stale_assistant_usage_after_newer_message_inserted_before_it():
    context = Context(
        system_prompt="system",
        messages=[
            UserMessage(content="summary", timestamp=200),
            create_assistant(100, 9_500),
            UserMessage(content="x" * 4_000, timestamp=300),
        ],
    )

    assert estimate_context_tokens(context) == ContextUsageEstimate(
        tokens=1_005,
        usage_tokens=0,
        trailing_tokens=1_005,
        last_usage_index=None,
    )


def test_uses_assistant_usage_again_after_response_to_inserted_context():
    context = Context(
        messages=[
            UserMessage(content="summary", timestamp=200),
            create_assistant(100, 9_500),
            UserMessage(content="new prompt", timestamp=300),
            create_assistant(400, 2_000),
            UserMessage(content="tail", timestamp=500),
        ],
    )

    assert estimate_context_tokens(context) == ContextUsageEstimate(
        tokens=2_001,
        usage_tokens=2_000,
        trailing_tokens=1,
        last_usage_index=3,
    )


def test_estimate_message_tokens_counts_tool_call_arguments():
    message = AssistantMessage(
        content=[ToolCall(id="t1", name="edit", arguments={"path": "a.py"})],
        api="anthropic-messages",
        provider="anthropic",
        model="m",
        usage=Usage(),
        stop_reason="toolUse",
        timestamp=0,
    )

    # chars = len("edit") + len('{"path":"a.py"}') == 4 + 15 -> ceil(19 / 4) == 5
    assert estimate_message_tokens(message) == 5
