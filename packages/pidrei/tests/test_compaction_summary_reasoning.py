"""Mirror of pi's compaction-summary-reasoning.test.ts.

pi mocks `completeSimple` from pi-ai/compat. pidrei's compaction requires an
explicit `stream_fn` (the compat registry is deliberately unported), so the
mirror injects a recording stream function instead — the same interception
point, one layer lower.
"""

import time
from dataclasses import replace

import pytest

from pidrei.core.compaction import (
    CompactionPreparation,
    CompactionSettings,
    FileOperations,
    compact,
    complete_summarization,
    generate_summary,
    generate_summary_with_usage,
)
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StartEvent,
    TextContent,
    ToolCall,
    Usage,
    UsageCost,
    UserMessage,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


def create_model(reasoning: bool, max_tokens: int = 8192) -> Model:
    return Model(
        id="reasoning-model" if reasoning else "non-reasoning-model",
        name="Reasoning Model" if reasoning else "Non-reasoning Model",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com",
        reasoning=reasoning,
        input=["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=max_tokens,
    )


def mock_summary_response() -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="## Goal\nTest summary")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(input=10, output=10, total_tokens=20, cost=UsageCost()),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def mock_tool_call_response() -> AssistantMessage:
    # Step 2 relaxation (PROPER_MT_DESIGN.md): messages are frozen values now,
    # so the tool-call shape is built by construction instead of mutation.
    return replace(
        mock_summary_response(),
        content=[ToolCall(id="tool-call-1", name="read", arguments={"path": "README.md"})],
        stop_reason="toolUse",
    )


def recording_stream_fn(response: AssistantMessage | None = None) -> tuple:
    """A stream function that records the options of every summarization call."""
    calls: list = []

    async def stream_fn(model, context, options=None):
        calls.append(options)
        stream = AssistantMessageEventStream()
        message = response if response is not None else mock_summary_response()
        stream.push(StartEvent(partial=mock_summary_response()))
        stream.push(DoneEvent(reason="toolUse" if message.stop_reason == "toolUse" else "stop", message=message))
        return stream

    return stream_fn, calls


def messages() -> list:
    return [UserMessage(content=[TextContent(text="Summarize this.")], timestamp=int(time.time() * 1000))]


@pytest.mark.tonio
async def test_uses_the_provided_thinking_level_for_reasoning_capable_models():
    stream_fn, calls = recording_stream_fn()

    result = await generate_summary_with_usage(
        messages(), create_model(True), 2000, "test-key", None, None, None, None, "medium", stream_fn
    )

    assert result.text == "## Goal\nTest summary"
    assert result.usage == mock_summary_response().usage

    assert len(calls) == 1
    assert calls[0].reasoning == "medium"
    assert calls[0].api_key == "test-key"


@pytest.mark.tonio
async def test_preserves_the_string_result_from_generate_summary():
    stream_fn, _calls = recording_stream_fn()

    summary = await generate_summary(messages(), create_model(False), 2000, "test-key", stream_fn=stream_fn)

    assert summary == "## Goal\nTest summary"


@pytest.mark.tonio
async def test_uses_fresh_routing_sessions_without_prompt_caching():
    stream_fn, calls = recording_stream_fn()

    await generate_summary(messages(), create_model(False), 2000, "test-key", stream_fn=stream_fn)
    await generate_summary(messages(), create_model(False), 2000, "test-key", stream_fn=stream_fn)

    assert len(calls) == 2
    assert all(options.cache_retention == "none" for options in calls)
    assert all(options.tool_choice == "none" for options in calls)
    assert calls[0].session_id != calls[1].session_id


@pytest.mark.tonio
async def test_honors_a_caller_supplied_routing_session_without_prompt_caching():
    stream_fn, calls = recording_stream_fn()

    await complete_summarization(
        create_model(False),
        Context(system_prompt="Summarize", messages=[]),
        SimpleStreamOptions(session_id="current-routing-session", cache_retention="long", tool_choice="auto"),
        stream_fn,
    )

    assert calls[0].session_id == "current-routing-session"
    assert calls[0].cache_retention == "none"
    assert calls[0].tool_choice == "none"


@pytest.mark.tonio
async def test_preserves_the_standalone_split_turn_summary_prompt():
    contexts: list = []

    async def stream_fn(model, context, options=None):
        contexts.append(context)
        stream = AssistantMessageEventStream()
        stream.push(StartEvent(partial=mock_summary_response()))
        stream.push(DoneEvent(reason="stop", message=mock_summary_response()))
        return stream

    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=[],
        turn_prefix_messages=messages(),
        is_split_turn=True,
        tokens_before=100,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )

    await compact(preparation, create_model(False), "test-key", stream_fn=stream_fn)

    # pi stringifies the request messages; the port reads the text blocks directly.
    prompt = "".join(block.text for message in contexts[0].messages for block in message.content)
    assert "This is the PREFIX of a turn that was too large to keep" in prompt
    assert "<conversation>" in prompt


@pytest.mark.tonio
async def test_rejects_tool_calls_from_conversation_summaries():
    stream_fn, _calls = recording_stream_fn(mock_tool_call_response())

    with pytest.raises(Exception, match="Summarization attempted to call a tool"):
        await generate_summary_with_usage(
            messages(), create_model(False), 2000, "test-key", None, None, None, None, None, stream_fn
        )


@pytest.mark.tonio
async def test_rejects_tool_calls_from_split_turn_summaries():
    stream_fn, _calls = recording_stream_fn(mock_tool_call_response())
    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=[],
        turn_prefix_messages=messages(),
        is_split_turn=True,
        tokens_before=100,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )

    with pytest.raises(Exception, match="Turn prefix summarization attempted to call a tool"):
        await compact(preparation, create_model(False), "test-key", stream_fn=stream_fn)


@pytest.mark.tonio
async def test_does_not_set_reasoning_when_thinking_is_off():
    stream_fn, calls = recording_stream_fn()

    await generate_summary(messages(), create_model(True), 2000, "test-key", None, None, None, None, "off", stream_fn)

    assert len(calls) == 1
    assert calls[0].api_key == "test-key"
    assert calls[0].reasoning is None


@pytest.mark.tonio
async def test_does_not_set_reasoning_for_non_reasoning_models():
    stream_fn, calls = recording_stream_fn()

    await generate_summary(
        messages(), create_model(False), 2000, "test-key", None, None, None, None, "medium", stream_fn
    )

    assert len(calls) == 1
    assert calls[0].api_key == "test-key"
    assert calls[0].reasoning is None


# pi's two remaining refusal-fallback cases assert that `refusalFallbacks` is absent
# from the summarization options. `SimpleStreamOptions` no longer declares that field
# at all (`ed867e90` moved fallbacks onto the model's compat metadata), so here the
# assertion could never fail; the cases are dropped instead of mirrored vacuously. The
# compat-driven behaviour they defer to is covered by
# packages/ai/tests/test_anthropic_fallback_usage.py.


@pytest.mark.tonio
async def test_clamps_compaction_summary_max_tokens_to_the_model_output_cap():
    stream_fn, calls = recording_stream_fn()
    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=messages(),
        turn_prefix_messages=messages(),
        is_split_turn=True,
        tokens_before=600000,
        file_ops=FileOperations(),
        settings=CompactionSettings(enabled=True, reserve_tokens=500000, keep_recent_tokens=20000),
    )

    result = await compact(preparation, create_model(False, 128000), "test-key", stream_fn=stream_fn)

    assert result.usage == Usage(input=20, output=20, total_tokens=40, cost=UsageCost())
    assert [options.max_tokens for options in calls] == [128000, 128000]
