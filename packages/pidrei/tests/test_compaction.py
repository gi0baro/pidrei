"""Mirror of pi coding-agent test/compaction.test.ts — the token-calculation half.

The cut-point, buildSessionContext, prepareCompaction and large-fixture blocks
are covered by the agent-session compaction suites (`test_agent_session_compaction*.py`,
the numbered regressions); this file pins the pure token helpers that used to
be shared with the agent harness compaction port (removed 2026-09-04,
UPSTREAM_EXPERIMENTAL_RULING.md) and the summarization serializer.
"""

import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pidrei.core.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    CutPointResult,
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    get_last_assistant_usage,
    prepare_compaction,
    serialize_conversation,
    should_compact,
)
from pidrei.core.messages import BashExecutionMessage, BranchSummaryMessage, CompactionSummaryMessage, CustomMessage
from pidrei_ai.types import (
    AssistantMessage,
    ImageContent,
    SystemMessage,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)


# ============================================================================
# Test fixtures
# ============================================================================


def _now_ms() -> int:
    return int(time.time() * 1000)


def create_mock_usage(input: int, output: int, cache_read: int = 0, cache_write: int = 0) -> Usage:
    return Usage(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input + output + cache_read + cache_write,
        cost=UsageCost(),
    )


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=_now_ms())


def create_assistant_message(text: str, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=usage if usage is not None else create_mock_usage(100, 50),
        stop_reason="stop",
        timestamp=_now_ms(),
    )


_next_id = 0
_last_id: str | None = None


@pytest.fixture(autouse=True)
def _reset_entry_counter():
    """pi resets its id counter and parent chain before each test."""
    global _next_id, _last_id
    _next_id = 0
    _last_id = None


def create_message_entry(message) -> dict:
    """Entries chain through `parentId` like pi's helper, so projection-based
    compaction walks the whole list from its last entry."""
    global _next_id, _last_id
    _next_id += 1
    entry_id = f"entry-{_next_id}"
    entry = {"type": "message", "id": entry_id, "parentId": _last_id, "timestamp": _now_ms(), "message": message}
    _last_id = entry_id
    return entry


# ============================================================================
# Token calculation
# ============================================================================


def test_should_calculate_total_context_tokens_from_usage():
    assert calculate_context_tokens(create_mock_usage(1000, 500, 200, 100)) == 1800


def test_should_handle_zero_values():
    assert calculate_context_tokens(create_mock_usage(0, 0, 0, 0)) == 0


# ============================================================================
# getLastAssistantUsage
# ============================================================================


def test_should_find_the_last_non_aborted_assistant_message_usage():
    entries = [
        create_message_entry(create_user_message("Hello")),
        create_message_entry(create_assistant_message("Hi", create_mock_usage(100, 50))),
        create_message_entry(create_user_message("How are you?")),
        create_message_entry(create_assistant_message("Good", create_mock_usage(200, 100))),
    ]
    usage = get_last_assistant_usage(entries)
    assert usage is not None
    assert usage.input == 200


def test_should_skip_aborted_messages():
    aborted = replace(create_assistant_message("Aborted", create_mock_usage(300, 150)), stop_reason="aborted")
    entries = [
        create_message_entry(create_user_message("Hello")),
        create_message_entry(create_assistant_message("Hi", create_mock_usage(100, 50))),
        create_message_entry(create_user_message("How are you?")),
        create_message_entry(aborted),
    ]
    usage = get_last_assistant_usage(entries)
    assert usage is not None
    assert usage.input == 100


def test_should_skip_all_zero_assistant_usage():
    entries = [
        create_message_entry(create_user_message("Hello")),
        create_message_entry(create_assistant_message("Hi", create_mock_usage(100, 50))),
        create_message_entry(create_user_message("continue")),
        create_message_entry(create_assistant_message("Partial", create_mock_usage(0, 0))),
    ]
    usage = get_last_assistant_usage(entries)
    assert usage is not None
    assert usage.input == 100


def test_should_return_none_if_no_assistant_messages():
    assert get_last_assistant_usage([create_message_entry(create_user_message("Hello"))]) is None


# ============================================================================
# estimateContextTokens
# ============================================================================


def test_uses_the_last_non_zero_assistant_usage_as_the_context_anchor():
    messages = [
        create_user_message("Hello"),
        create_assistant_message("Hi", create_mock_usage(100, 50)),
        create_user_message("continue"),
        create_assistant_message("Partial thinking", create_mock_usage(0, 0)),
    ]
    estimate = estimate_context_tokens(messages)
    assert estimate.usage_tokens == 150
    assert estimate.last_usage_index == 1
    assert estimate.trailing_tokens > 0
    assert estimate.tokens == 150 + estimate.trailing_tokens


def test_estimates_without_any_usage_anchor():
    estimate = estimate_context_tokens([create_user_message("no usage")])
    assert estimate.last_usage_index is None
    assert estimate.usage_tokens == 0
    assert estimate.tokens == estimate.trailing_tokens > 0


def test_estimates_tokens_across_supported_message_roles():
    assistant = replace(
        create_assistant_message("assistant"),
        content=[
            ThinkingContent(thinking="thinking"),
            ToolCall(id="call-1", name="read", arguments={"path": "file.ts"}),
        ],
    )
    tool_result_with_image = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[TextContent(text="tool text"), ImageContent(mime_type="image/png", data="abc")],
        is_error=False,
        timestamp=_now_ms(),
    )
    bash_execution = BashExecutionMessage(
        command="npm run check", output="ok", exit_code=0, cancelled=False, truncated=False, timestamp=_now_ms()
    )

    assert estimate_tokens(create_user_message("plain user")) > 0
    assert estimate_tokens(assistant) > 0
    assert (
        estimate_tokens(CustomMessage(custom_type="note", content="custom text", display=True, timestamp=_now_ms())) > 0
    )
    assert estimate_tokens(tool_result_with_image) > 1000  # the image counts as 4800 chars
    assert estimate_tokens(bash_execution) > 0
    assert estimate_tokens(BranchSummaryMessage(summary="branch", from_id="x", timestamp=_now_ms())) > 0
    assert estimate_tokens(CompactionSummaryMessage(summary="compact", tokens_before=123, timestamp=_now_ms())) > 0
    assert estimate_tokens(SimpleNamespace(role="unknown", timestamp=_now_ms())) == 0


# ============================================================================
# shouldCompact
# ============================================================================


def test_should_return_true_when_context_exceeds_threshold():
    settings = CompactionSettings(enabled=True, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, settings) is True
    assert should_compact(89000, 100000, settings) is False


def test_should_return_false_when_disabled():
    settings = CompactionSettings(enabled=False, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, settings) is False


# ============================================================================
# findCutPoint
# ============================================================================


# Regression test for #9740.
def test_should_fall_back_to_the_latest_valid_cut_point_before_oversized_trailing_tool_results():
    old_user = create_message_entry(create_user_message("old history"))
    old_assistant = create_message_entry(create_assistant_message("old answer"))
    current_user = create_message_entry(create_user_message("read the large file"))
    tool_call = create_message_entry(
        replace(
            create_assistant_message(""),
            content=[ToolCall(id="call-1", name="read", arguments={"path": "big.txt"})],
            stop_reason="toolUse",
        )
    )
    tool_result = create_message_entry(
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="read",
            content=[TextContent(text="x" * 8000)],
            is_error=False,
            timestamp=_now_ms(),
        )
    )
    entries = [old_user, old_assistant, current_user, tool_call, tool_result]

    result = find_cut_point(entries, 0, len(entries), 1000)
    assert result == CutPointResult(first_kept_entry_index=3, turn_start_index=2, is_split_turn=True)

    preparation = prepare_compaction(entries, replace(DEFAULT_COMPACTION_SETTINGS, keep_recent_tokens=1000))
    assert preparation is not None
    assert preparation.first_kept_entry_id == tool_call["id"]
    assert preparation.messages_to_summarize == [old_user["message"], old_assistant["message"]]
    assert preparation.turn_prefix_messages == [current_user["message"]]


# ============================================================================
# prepareCompaction
# ============================================================================


def test_does_not_treat_system_messages_as_conversation_history():
    system = create_message_entry(
        SystemMessage(content="", sections={"preamble": "current prompt"}, timestamp=_now_ms())
    )
    user = create_message_entry(create_user_message("one long turn"))
    assistant = create_message_entry(create_assistant_message("assistant suffix"))
    preparation = prepare_compaction(
        [system, user, assistant], replace(DEFAULT_COMPACTION_SETTINGS, keep_recent_tokens=1)
    )

    assert preparation is not None
    assert preparation.first_kept_entry_id == assistant["id"]
    assert preparation.is_split_turn is True
    assert preparation.messages_to_summarize == []
    assert preparation.turn_prefix_messages == [user["message"]]


# ============================================================================
# serializeConversation
# ============================================================================


def test_serializes_conversation_with_truncated_tool_results():
    messages = [
        ToolResultMessage(
            tool_call_id="tc1",
            tool_name="read",
            content=[TextContent(text="x" * 5000)],
            is_error=False,
            timestamp=_now_ms(),
        )
    ]
    result = serialize_conversation(messages)
    assert "[Tool result]:" in result
    assert "[... 3000 more characters truncated]" in result
