"""Mirror of pi agent/test/harness/compaction.test.ts."""

import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from pidrei_agent.harness.compaction.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionPreparation,
    CompactionSettings,
    calculate_context_tokens,
    compact,
    estimate_context_tokens,
    estimate_tokens,
    find_cut_point,
    find_turn_start_index,
    generate_summary,
    generate_summary_with_usage,
    get_last_assistant_usage,
    prepare_compaction,
    should_compact,
)
from pidrei_agent.harness.compaction.utils import FileOperations, serialize_conversation
from pidrei_agent.harness.session.session import build_session_context
from pidrei_agent.harness.types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomMessageEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionModelRef,
    ThinkingLevelChangeEntry,
    get_or_throw,
)
from pidrei_ai.providers.faux import FauxModelDefinition, faux_assistant_message, faux_provider
from pidrei_ai.registry import create_models
from pidrei_ai.types import (
    AssistantMessage,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)


_next_id = 0


def create_id() -> str:
    global _next_id
    entry_id = f"entry-{_next_id}"
    _next_id += 1
    return entry_id


@pytest.fixture(autouse=True)
def _reset_ids():
    global _next_id
    _next_id = 0


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


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
    return UserMessage(content=[TextContent(text=text)], timestamp=int(time.time() * 1000))


def create_assistant_message(text: str, usage: Usage | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=usage if usage is not None else create_mock_usage(100, 50),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def create_message_entry(message, parent_id: str | None = None) -> MessageEntry:
    return MessageEntry(id=create_id(), parent_id=parent_id, timestamp=_iso_now(), message=message)


def create_compaction_entry(
    summary: str,
    first_kept_entry_id: str,
    parent_id: str | None = None,
    retained_tail=None,
) -> CompactionEntry:
    return CompactionEntry(
        id=create_id(),
        parent_id=parent_id,
        timestamp=_iso_now(),
        summary=summary,
        first_kept_entry_id=first_kept_entry_id,
        tokens_before=1234,
        retained_tail=retained_tail,
    )


def create_thinking_level_entry(level: str, parent_id: str | None = None) -> ThinkingLevelChangeEntry:
    return ThinkingLevelChangeEntry(id=create_id(), parent_id=parent_id, timestamp=_iso_now(), thinking_level=level)


def create_model_change_entry(provider: str, model_id: str, parent_id: str | None = None) -> ModelChangeEntry:
    return ModelChangeEntry(
        id=create_id(), parent_id=parent_id, timestamp=_iso_now(), provider=provider, model_id=model_id
    )


# Shared collection; each faux provider gets a unique id so coexisting fakes route correctly.
models = create_models()
_faux_count = 0


def create_faux_model(reasoning: bool, max_tokens: int = 8192):
    global _faux_count
    _faux_count += 1
    faux = faux_provider(
        provider=f"faux-{_faux_count}",
        models=[
            FauxModelDefinition(
                id="reasoning-model" if reasoning else "non-reasoning-model",
                reasoning=reasoning,
                context_window=200000,
                max_tokens=max_tokens,
            )
        ],
    )
    models.set_provider(faux.provider)
    return faux, faux.get_model()


class _StubModels:
    """pi: Object.create(models) with a stubbed completeSimple."""

    def __init__(self, responses: list[AssistantMessage]):
        self._responses = list(responses)

    async def complete_simple(self, _model, _context, _options=None) -> AssistantMessage:
        if not self._responses:
            raise Exception("No faux completeSimple response queued")
        return self._responses.pop(0)


def empty_file_ops() -> FileOperations:
    return FileOperations()


@pytest.mark.tonio
async def test_calculates_total_context_tokens_from_usage():
    assert calculate_context_tokens(create_mock_usage(1000, 500, 200, 100)) == 1800
    assert calculate_context_tokens(create_mock_usage(0, 0, 0, 0)) == 0


@pytest.mark.tonio
async def test_checks_compaction_threshold():
    settings = CompactionSettings(enabled=True, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, settings) is True
    assert should_compact(89000, 100000, settings) is False
    disabled = CompactionSettings(enabled=False, reserve_tokens=10000, keep_recent_tokens=20000)
    assert should_compact(95000, 100000, disabled) is False


@pytest.mark.tonio
async def test_finds_a_cut_point_based_on_token_differences():
    entries = []
    parent_id = None
    for i in range(10):
        user = create_message_entry(create_user_message(f"User {i}"), parent_id)
        entries.append(user)
        assistant = create_message_entry(
            create_assistant_message(f"Assistant {i}", create_mock_usage(0, 100, (i + 1) * 1000, 0)), user.id
        )
        entries.append(assistant)
        parent_id = assistant.id

    result = find_cut_point(entries, 0, len(entries), 2500)
    assert entries[result.first_kept_entry_index].type == "message"


@pytest.mark.tonio
async def test_covers_cut_point_and_turn_start_edge_cases():
    thinking = create_thinking_level_entry("high")
    model_change = create_model_change_entry("openai", "gpt-4", thinking.id)
    result = find_cut_point([thinking, model_change], 0, 2, 1)
    assert (result.first_kept_entry_index, result.turn_start_index, result.is_split_turn) == (0, -1, False)

    branch_summary = BranchSummaryEntry(
        id=create_id(), parent_id=model_change.id, timestamp=_iso_now(), from_id="branch", summary="branch summary"
    )
    custom_message = CustomMessageEntry(
        id=create_id(),
        parent_id=branch_summary.id,
        timestamp=_iso_now(),
        custom_type="note",
        content="custom content",
        display=True,
    )
    assert find_turn_start_index([thinking, branch_summary], 1, 0) == 1
    assert find_turn_start_index([thinking, custom_message], 1, 0) == 1
    assert find_turn_start_index([thinking, model_change], 1, 0) == -1

    result = find_cut_point([thinking, branch_summary, custom_message], 0, 3, 1)
    assert result.first_kept_entry_index == 0

    tool_result = create_message_entry(
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="read",
            content=[TextContent(text="tool output")],
            is_error=False,
            timestamp=int(time.time() * 1000),
        )
    )
    result = find_cut_point([tool_result], 0, 1, 1)
    assert (result.first_kept_entry_index, result.turn_start_index, result.is_split_turn) == (0, -1, False)

    user = create_message_entry(create_user_message("user"))
    compaction = create_compaction_entry("summary", user.id, user.id)
    assistant = create_message_entry(create_assistant_message("assistant"), compaction.id)
    assert find_cut_point([user, compaction, assistant], 0, 3, 1).first_kept_entry_index == 2


@pytest.mark.tonio
async def test_estimates_tokens_and_context_usage_across_supported_message_roles():
    from dataclasses import replace as dc_replace

    from pidrei_agent.harness.messages import (
        BashExecutionMessage,
        BranchSummaryMessage,
        CompactionSummaryMessage,
        CustomMessage,
    )

    usage = create_mock_usage(10, 5, 3, 2)
    assistant = create_assistant_message("assistant", usage)
    assistant_with_thinking_and_tool = dc_replace(
        assistant,
        content=[
            ThinkingContent(thinking="thinking"),
            ToolCall(id="call-1", name="read", arguments={"path": "file.ts"}),
        ],
    )
    custom_string = CustomMessage(
        custom_type="note", content="custom text", display=True, timestamp=int(time.time() * 1000)
    )
    tool_result_with_image = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[TextContent(text="tool text"), ImageContent(mime_type="image/png", data="abc")],
        is_error=False,
        timestamp=int(time.time() * 1000),
    )
    bash_execution = BashExecutionMessage(
        command="npm run check",
        output="ok",
        exit_code=0,
        cancelled=False,
        truncated=False,
        timestamp=int(time.time() * 1000),
    )
    branch_summary_message = BranchSummaryMessage(summary="branch", from_id="x", timestamp=int(time.time() * 1000))
    compaction_summary_message = CompactionSummaryMessage(
        summary="compact", tokens_before=123, timestamp=int(time.time() * 1000)
    )

    assert estimate_tokens(UserMessage(content="plain user", timestamp=int(time.time() * 1000))) > 0
    assert estimate_tokens(assistant_with_thinking_and_tool) > 0
    assert estimate_tokens(custom_string) > 0
    assert estimate_tokens(tool_result_with_image) > 1000
    assert estimate_tokens(bash_execution) > 0
    assert estimate_tokens(branch_summary_message) > 0
    assert estimate_tokens(compaction_summary_message) > 0
    assert estimate_tokens(SimpleNamespace(role="unknown", timestamp=int(time.time() * 1000))) == 0
    assert (
        get_last_assistant_usage([create_message_entry(create_user_message("user")), create_message_entry(assistant)])
        is usage
    )
    assert (
        get_last_assistant_usage(
            [
                create_message_entry(dc_replace(assistant, stop_reason="aborted")),
                create_message_entry(dc_replace(assistant, stop_reason="error")),
            ]
        )
        is None
    )
    assert (
        get_last_assistant_usage(
            [
                create_message_entry(create_user_message("user")),
                create_message_entry(assistant),
                create_message_entry(create_assistant_message("partial", create_mock_usage(0, 0))),
            ]
        )
        is usage
    )
    assert estimate_context_tokens([create_user_message("no usage")]).last_usage_index is None
    tail_estimate = estimate_context_tokens([assistant, create_user_message("tail")])
    assert (tail_estimate.usage_tokens, tail_estimate.last_usage_index) == (20, 0)
    estimate = estimate_context_tokens(
        [
            create_user_message("Hello"),
            assistant,
            create_user_message("continue"),
            create_assistant_message("Partial thinking", create_mock_usage(0, 0)),
        ]
    )
    assert estimate.usage_tokens == 20
    assert estimate.last_usage_index == 1
    assert estimate.trailing_tokens > 0
    assert estimate.tokens == 20 + estimate.trailing_tokens


@pytest.mark.tonio
async def test_builds_session_context_with_a_compaction_entry():
    u1 = create_message_entry(create_user_message("1"))
    a1 = create_message_entry(create_assistant_message("a"), u1.id)
    u2 = create_message_entry(create_user_message("2"), a1.id)
    a2 = create_message_entry(create_assistant_message("b"), u2.id)
    compaction = create_compaction_entry(
        "Summary of 1,a,2,b", u2.id, a2.id, [create_user_message("2"), create_assistant_message("b")]
    )
    u3 = create_message_entry(create_user_message("3"), compaction.id)
    a3 = create_message_entry(create_assistant_message("c"), u3.id)
    loaded = build_session_context([u1, a1, u2, a2, compaction, u3, a3])
    assert len(loaded.messages) == 5
    assert [getattr(m, "role", None) for m in loaded.messages] == [
        "compactionSummary",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.tonio
async def test_falls_back_to_first_kept_entry_id_when_a_compaction_has_no_retained_tail():
    u1 = create_message_entry(create_user_message("1"))
    a1 = create_message_entry(create_assistant_message("a"), u1.id)
    u2 = create_message_entry(create_user_message("2"), a1.id)
    a2 = create_message_entry(create_assistant_message("b"), u2.id)
    compaction = create_compaction_entry("Summary of 1,a,2,b", u2.id, a2.id)
    u3 = create_message_entry(create_user_message("3"), compaction.id)
    loaded = build_session_context([u1, a1, u2, a2, compaction, u3])
    assert [getattr(m, "role", None) for m in loaded.messages] == ["compactionSummary", "user", "assistant", "user"]


@pytest.mark.tonio
async def test_tracks_model_and_thinking_level_changes_in_built_context():
    user = create_message_entry(create_user_message("1"))
    model_change = create_model_change_entry("openai", "gpt-4", user.id)
    assistant = create_message_entry(create_assistant_message("a"), model_change.id)
    thinking_change = create_thinking_level_entry("high", assistant.id)
    loaded = build_session_context([user, model_change, assistant, thinking_change])
    assert loaded.model == SessionModelRef(provider="anthropic", model_id="claude-sonnet-4-5")
    assert loaded.thinking_level == "high"


@pytest.mark.tonio
async def test_prepares_compaction_using_the_latest_compaction_summary_as_previous_summary():
    u1 = create_message_entry(create_user_message("user msg 1"))
    a1 = create_message_entry(create_assistant_message("assistant msg 1"), u1.id)
    u2 = create_message_entry(create_user_message("user msg 2"), a1.id)
    a2 = create_message_entry(create_assistant_message("assistant msg 2", create_mock_usage(5000, 1000)), u2.id)
    compaction1 = create_compaction_entry("First summary", u2.id, a2.id)
    u3 = create_message_entry(create_user_message("user msg 3"), compaction1.id)
    a3 = create_message_entry(create_assistant_message("assistant msg 3", create_mock_usage(8000, 2000)), u3.id)
    path_entries = [u1, a1, u2, a2, compaction1, u3, a3]
    preparation = get_or_throw(prepare_compaction(path_entries, DEFAULT_COMPACTION_SETTINGS))
    assert preparation is not None
    assert preparation.previous_summary == "First summary"
    assert preparation.first_kept_entry_id
    assert len(preparation.retained_tail) > 0
    assert preparation.tokens_before == estimate_context_tokens(build_session_context(path_entries).messages).tokens


@pytest.mark.tonio
async def test_prepares_split_turn_compaction_with_prior_file_operation_details():
    from dataclasses import replace as dc_replace

    u1 = create_message_entry(create_user_message("user msg 1"))
    assistant_message = dc_replace(
        create_assistant_message("assistant msg 1"),
        content=[ToolCall(id="tool-1", name="write", arguments={"path": "written.ts"})],
    )
    a1 = create_message_entry(assistant_message, u1.id)
    compaction1 = create_compaction_entry("First summary", u1.id, a1.id)
    compaction1.details = {"readFiles": ["old-read.ts"], "modifiedFiles": ["old-edit.ts"]}
    u2 = create_message_entry(create_user_message("large turn"), compaction1.id)
    a2 = create_message_entry(create_assistant_message("large assistant message"), u2.id)
    preparation = get_or_throw(
        prepare_compaction(
            [u1, a1, compaction1, u2, a2],
            CompactionSettings(enabled=True, reserve_tokens=100, keep_recent_tokens=1),
        )
    )

    assert preparation is not None
    assert preparation.previous_summary == "First summary"
    assert preparation.is_split_turn is True
    assert [getattr(m, "role", None) for m in preparation.turn_prefix_messages] == ["user"]
    assert "old-read.ts" in preparation.file_ops.read
    assert "old-edit.ts" in preparation.file_ops.edited
    assert "written.ts" in preparation.file_ops.written


@pytest.mark.tonio
async def test_prepares_custom_and_branch_summary_entries_for_summarization():
    branch_summary = BranchSummaryEntry(
        id=create_id(), parent_id=None, timestamp=_iso_now(), from_id="branch", summary="branch summary"
    )
    custom_message = CustomMessageEntry(
        id=create_id(),
        parent_id=branch_summary.id,
        timestamp=_iso_now(),
        custom_type="note",
        content="custom content",
        display=True,
    )
    user = create_message_entry(create_user_message("keep"), custom_message.id)
    assistant = create_message_entry(create_assistant_message("assistant"), user.id)
    preparation = get_or_throw(
        prepare_compaction(
            [branch_summary, custom_message, user, assistant],
            CompactionSettings(enabled=True, reserve_tokens=100, keep_recent_tokens=1),
        )
    )

    assert preparation is not None
    assert [getattr(m, "role", None) for m in preparation.messages_to_summarize] == ["branchSummary", "custom"]


@pytest.mark.tonio
async def test_does_not_prepare_compaction_when_there_is_nothing_valid_to_compact():
    compaction = create_compaction_entry("already compacted", "entry-keep")
    assert get_or_throw(prepare_compaction([compaction], DEFAULT_COMPACTION_SETTINGS)) is None
    assert get_or_throw(prepare_compaction([], DEFAULT_COMPACTION_SETTINGS)) is None


@pytest.mark.tonio
async def test_serializes_conversation_with_truncated_tool_results():
    long_content = "x" * 5000
    messages = [
        ToolResultMessage(
            tool_call_id="tc1",
            tool_name="read",
            content=[TextContent(text=long_content)],
            is_error=False,
            timestamp=int(time.time() * 1000),
        )
    ]
    result = serialize_conversation(messages)
    assert "[Tool result]:" in result
    assert "[... 3000 more characters truncated]" in result


@pytest.mark.tonio
async def test_passes_reasoning_through_generate_summary_only_for_reasoning_models_with_thinking_enabled():
    messages = [create_user_message("Summarize this.")]
    seen_options = []

    async def responder(_context, options, _state, _model):
        seen_options.append(options)
        return faux_assistant_message("## Goal\nTest summary")

    faux_reasoning, reasoning_model = create_faux_model(True)
    faux_reasoning.set_responses([responder])
    get_or_throw(await generate_summary(messages, models, reasoning_model, 2000, None, None, None, "medium"))
    assert seen_options[0].reasoning == "medium"

    faux_off, off_model = create_faux_model(True)
    faux_off.set_responses([responder])
    get_or_throw(await generate_summary(messages, models, off_model, 2000, None, None, None, "off"))
    assert seen_options[1].reasoning is None

    faux_non_reasoning, non_reasoning_model = create_faux_model(False)
    faux_non_reasoning.set_responses([responder])
    get_or_throw(await generate_summary(messages, models, non_reasoning_model, 2000, None, None, None, "medium"))
    assert seen_options[2].reasoning is None


@pytest.mark.tonio
async def test_includes_previous_summaries_and_custom_instructions_in_generate_summary_prompts():
    messages = [create_user_message("Summarize this.")]
    prompt_text = ""

    async def responder(context, _options, _state, _model):
        nonlocal prompt_text
        message = context.messages[0] if context.messages else None
        content = message.content if message is not None and getattr(message, "role", None) == "user" else []
        prompt_text = content[0].text if isinstance(content, list) and content and content[0].type == "text" else ""
        return faux_assistant_message("## Goal\nTest summary")

    faux, model = create_faux_model(False)
    faux.set_responses([responder])

    summary_text, summary_usage = get_or_throw(
        await generate_summary_with_usage(messages, models, model, 2000, None, "focus", "old summary")
    )

    assert "Test summary" in summary_text
    assert summary_usage.input > 0
    assert summary_usage.output > 0
    assert summary_usage.total_tokens == (
        summary_usage.input + summary_usage.output + summary_usage.cache_read + summary_usage.cache_write
    )
    assert "<previous-summary>\nold summary\n</previous-summary>" in prompt_text
    assert "Additional focus: focus" in prompt_text


@pytest.mark.tonio
async def test_preserves_the_string_result_from_generate_summary():
    messages = [create_user_message("Summarize this.")]
    faux, model = create_faux_model(False)
    faux.set_responses([faux_assistant_message("## Goal\nTest summary")])

    assert get_or_throw(await generate_summary(messages, models, model, 2000)) == "## Goal\nTest summary"


@pytest.mark.tonio
async def test_returns_error_results_for_failed_or_aborted_summary_generations():
    messages = [create_user_message("Summarize this.")]
    error_faux, error_model = create_faux_model(False)
    error_faux.set_responses([faux_assistant_message("", stop_reason="error", error_message="boom")])
    error_result = await generate_summary(messages, models, error_model, 2000)
    assert error_result.ok is False
    assert error_result.error.code == "summarization_failed"
    assert error_result.error.message == "Summarization failed: boom"

    aborted_faux, aborted_model = create_faux_model(False)
    aborted_faux.set_responses([faux_assistant_message("", stop_reason="aborted", error_message="stopped")])
    aborted_result = await generate_summary(messages, models, aborted_model, 2000)
    assert aborted_result.ok is False
    assert aborted_result.error.code == "aborted"
    assert aborted_result.error.message == "stopped"


@pytest.mark.tonio
async def test_clamps_compaction_summary_max_tokens_to_the_model_output_cap():
    messages = [create_user_message("Summarize this.")]
    seen_options = []

    async def responder(_context, options, _state, _model):
        seen_options.append(options)
        return faux_assistant_message("## Goal\nTest summary")

    faux, model = create_faux_model(False, 128000)
    faux.set_responses([responder, responder])
    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=messages,
        turn_prefix_messages=messages,
        retained_tail=messages,
        is_split_turn=True,
        tokens_before=600000,
        file_ops=empty_file_ops(),
        settings=CompactionSettings(enabled=True, reserve_tokens=500000, keep_recent_tokens=20000),
    )

    get_or_throw(await compact(preparation, models, model))

    assert [options.max_tokens for options in seen_options] == [128000, 128000]
    assert [options.cache_retention for options in seen_options] == ["none", "none"]
    session_ids = [options.session_id for options in seen_options]
    assert session_ids[0] != session_ids[1]


@pytest.mark.tonio
async def test_returns_compaction_error_results_without_throwing():
    from dataclasses import replace as dc_replace

    messages = [create_user_message("Summarize this.")]
    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=messages,
        turn_prefix_messages=[],
        retained_tail=messages,
        is_split_turn=False,
        tokens_before=100,
        file_ops=empty_file_ops(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )
    history_faux, history_model = create_faux_model(False)
    history_faux.set_responses([faux_assistant_message("", stop_reason="error", error_message="history failed")])
    result = await compact(preparation, models, history_model)
    assert result.ok is False
    assert result.error.code == "summarization_failed"
    assert result.error.message == "Summarization failed: history failed"

    _, invalid_model = create_faux_model(False)
    invalid_result = await compact(
        dc_replace(preparation, messages_to_summarize=[], first_kept_entry_id=""), models, invalid_model
    )
    assert invalid_result.ok is False
    assert invalid_result.error.code == "invalid_session"


@pytest.mark.tonio
async def test_combines_usage_for_split_turn_compaction_summaries():
    from dataclasses import replace as dc_replace

    messages = [create_user_message("Summarize this.")]
    _, model = create_faux_model(False)
    history_usage = create_mock_usage(1, 2, 3, 4)
    turn_prefix_usage = create_mock_usage(5, 6, 7, 8)
    usage_models = _StubModels(
        [
            dc_replace(faux_assistant_message("history summary"), usage=history_usage),
            dc_replace(faux_assistant_message("turn prefix summary"), usage=turn_prefix_usage),
        ]
    )
    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=messages,
        turn_prefix_messages=messages,
        is_split_turn=True,
        tokens_before=100,
        retained_tail=messages,
        file_ops=empty_file_ops(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )

    result = get_or_throw(await compact(preparation, usage_models, model))

    assert result.usage == create_mock_usage(6, 8, 10, 12)


@pytest.mark.tonio
async def test_passes_reasoning_through_turn_prefix_summaries_when_enabled():
    messages = [create_user_message("Summarize this.")]
    seen_options = []

    async def responder(_context, options, _state, _model):
        seen_options.append(options)
        return faux_assistant_message("## Original Request\nTest summary")

    faux, model = create_faux_model(True)
    faux.set_responses([responder])
    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=[],
        turn_prefix_messages=messages,
        retained_tail=messages,
        is_split_turn=True,
        tokens_before=100,
        file_ops=empty_file_ops(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )

    get_or_throw(await compact(preparation, models, model, None, None, "high"))

    assert seen_options[0].reasoning == "high"


@pytest.mark.tonio
async def test_returns_turn_prefix_compaction_errors_without_throwing():
    messages = [create_user_message("Summarize this.")]
    preparation = CompactionPreparation(
        first_kept_entry_id="entry-keep",
        messages_to_summarize=[],
        turn_prefix_messages=messages,
        retained_tail=messages,
        is_split_turn=True,
        tokens_before=100,
        file_ops=empty_file_ops(),
        settings=CompactionSettings(enabled=True, reserve_tokens=2000, keep_recent_tokens=20),
    )
    faux, model = create_faux_model(False)
    faux.set_responses([faux_assistant_message("", stop_reason="error", error_message="prefix failed")])

    result = await compact(preparation, models, model)
    assert result.ok is False
    assert result.error.code == "summarization_failed"
    assert result.error.message == "Turn prefix summarization failed: prefix failed"

    aborted_faux, aborted_model = create_faux_model(False)
    aborted_faux.set_responses([faux_assistant_message("", stop_reason="aborted", error_message="prefix stopped")])
    aborted = await compact(preparation, models, aborted_model)
    assert aborted.ok is False
    assert aborted.error.code == "aborted"
    assert aborted.error.message == "prefix stopped"


@pytest.mark.tonio
async def test_returns_a_compaction_result_with_file_details():
    from dataclasses import replace as dc_replace

    u1 = create_message_entry(create_user_message("read a file"))
    assistant_message = dc_replace(
        create_assistant_message("calling tool", create_mock_usage(1000, 200)),
        content=[ToolCall(id="tool-1", name="read", arguments={"path": "src/index.ts"})],
    )
    a1 = create_message_entry(assistant_message, u1.id)
    u2 = create_message_entry(create_user_message("continue"), a1.id)
    a2 = create_message_entry(create_assistant_message("done", create_mock_usage(4000, 500)), u2.id)
    preparation = get_or_throw(prepare_compaction([u1, a1, u2, a2], DEFAULT_COMPACTION_SETTINGS))
    assert preparation is not None
    faux, model = create_faux_model(False)
    faux.set_responses([faux_assistant_message("## Goal\nTest summary")])
    result = get_or_throw(await compact(preparation, models, model))
    assert len(result.summary) > 0
    assert result.first_kept_entry_id
    assert result.usage is not None and result.usage.total_tokens > 0
    assert result.retained_tail is not None and len(result.retained_tail) > 0
    assert result.details is not None
