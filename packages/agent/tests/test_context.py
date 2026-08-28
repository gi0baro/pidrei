"""v4 session context projection (mirror of pi agent/test/harness/session/context.test.ts)."""

from dataclasses import replace

from pidrei_agent.harness.session.context import (
    SessionContextBuildOptions,
    SessionModelRef,
    build_session_context,
)
from pidrei_agent.harness.session.types import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    Entry,
    MessageEntry,
    ModelChangeEntry,
    ThinkingLevelEntry,
)
from pidrei_ai.types import AssistantMessage, TextContent, Usage, UserMessage


def user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


def assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(),
        stop_reason="stop",
        timestamp=1,
    )


def _stamp(entry: Entry, seq: int) -> Entry:
    entry.seq = seq
    entry.timestamp = seq
    return entry


def test_starts_at_the_latest_compaction_and_materializes_its_retained_tail():
    entries: list[Entry] = [
        _stamp(MessageEntry(id="old", parent_id=None, message=user_message("old")), 1),
        _stamp(
            CompactionEntry(
                id="compact",
                parent_id="old",
                summary="summary",
                retained_tail=[user_message("retained"), assistant_message("answer")],
                tokens_before=100,
            ),
            2,
        ),
        _stamp(ModelChangeEntry(id="model", parent_id="compact", provider="openai", model_id="gpt-5"), 3),
        _stamp(ThinkingLevelEntry(id="thinking", parent_id="model", thinking_level="high"), 4),
        _stamp(MessageEntry(id="tail", parent_id="thinking", message=user_message("tail")), 5),
    ]

    context = build_session_context(entries)
    assert [message.role for message in context.messages] == ["compactionSummary", "user", "assistant", "user"]
    assert context.model == SessionModelRef(provider="openai", model_id="gpt-5")
    assert context.thinking_level == "high"


def test_applies_caller_transforms_after_the_compaction_boundary():
    entries: list[Entry] = [
        _stamp(MessageEntry(id="old", parent_id=None, message=user_message("old")), 1),
        _stamp(
            CompactionEntry(id="compact", parent_id="old", summary="summary", retained_tail=[], tokens_before=100),
            2,
        ),
        _stamp(BranchSummaryEntry(id="branch", parent_id="compact", from_id="abandoned", summary="branch summary"), 3),
        _stamp(MessageEntry(id="tail", parent_id="branch", message=user_message("tail")), 4),
    ]

    context = build_session_context(
        entries,
        SessionContextBuildOptions(
            entry_transforms=[
                lambda context_entries: [candidate for candidate in context_entries if candidate.type != "compaction"]
            ]
        ),
    )
    assert [message.role for message in context.messages] == ["branchSummary", "user"]


def test_projects_custom_entries_and_omits_deferred_assistant_handles():
    # The `deferred` response-handle field on AssistantMessage arrives with the
    # 0.84 ai wave; the projection only reads stop_reason.
    # Step 2 relaxation (PROPER_MT_DESIGN.md): messages are frozen values now,
    # so the deferred shape is built by construction instead of mutation.
    deferred = replace(assistant_message(""), content=[], stop_reason="deferred")
    entries: list[Entry] = [
        _stamp(MessageEntry(id="user", parent_id=None, message=user_message("hello")), 1),
        _stamp(MessageEntry(id="deferred", parent_id="user", message=deferred), 2),
        _stamp(CustomEntry(id="custom", parent_id="deferred", custom_type="note", data="project me"), 3),
    ]

    context = build_session_context(
        entries,
        SessionContextBuildOptions(
            entry_projectors={"note": lambda custom, index, all_entries: [user_message(f"note: {custom.data}")]}
        ),
    )
    assert [message.role for message in context.messages] == ["user", "user"]
    assert context.messages[1].content == [TextContent(text="note: project me")]
