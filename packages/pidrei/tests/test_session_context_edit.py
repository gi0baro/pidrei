"""Mirror of pi coding-agent test/session-context-edit.test.ts.

pi checks `JSON.stringify(...)` of prepared messages for absent text; the
dataclass `repr` carries the same content here.
"""

import dataclasses
import time

import pytest

from pidrei.core.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    estimate_projected_context_tokens,
    prepare_compaction,
)
from pidrei.core.session_manager import SessionManager
from pidrei_ai.types import (
    AssistantMessage,
    SystemMessage,
    TextContent,
    Tool,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)


def _now() -> int:
    return int(time.time() * 1000)


def assistant(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="faux",
        provider="faux",
        model="faux",
        usage=Usage(input=10, output=1, cache_read=0, cache_write=0, total_tokens=11, cost=UsageCost()),
        stop_reason="stop",
        timestamp=_now(),
    )


def user(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=_now())


def text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(part.text for part in content if part.type == "text")


def _summary_or_text(message) -> str:
    return message.summary if hasattr(message, "summary") else text(message)


def _keep_one_token():
    return dataclasses.replace(DEFAULT_COMPACTION_SETTINGS, keep_recent_tokens=1)


@pytest.mark.tonio
async def test_omits_a_target_only_from_model_projection():
    session = SessionManager.in_memory()
    await session.append_message(user("request"))
    assistant_id = await session.append_message(assistant("partial"))
    result = ToolResultMessage(
        tool_call_id="call-1",
        tool_name="read",
        content=[TextContent(text="raw output")],
        details={"path": "large.txt"},
        is_error=True,
        timestamp=_now(),
    )
    result_id = await session.append_message(result)
    await session.append_context_edit(assistant_id, None)
    await session.append_context_edit(result_id, None)

    assert len([entry for entry in session.get_branch() if entry["type"] == "message"]) == 3
    assert [message.role for message in session.build_session_projection().messages] == ["user"]
    assert session.get_entry(result_id)["message"] is result


@pytest.mark.tonio
async def test_replaces_only_content_and_lets_the_latest_edit_win():
    session = SessionManager.in_memory()
    target_id = await session.append_message(assistant("original"))
    await session.append_context_edit(target_id, {"content": [TextContent(text="first")]})
    await session.append_context_edit(target_id, None)
    await session.append_context_edit(target_id, {"content": [TextContent(text="restored")]})

    projected = session.build_session_projection().messages[0]
    assert projected.role == "assistant"
    assert text(projected) == "restored"
    assert projected.usage.total_tokens == 11
    assert text(session.get_entry(target_id)["message"]) == "original"


@pytest.mark.tonio
async def test_normalizes_string_replacements_for_array_only_assistant_and_tool_result_roles():
    session = SessionManager.in_memory()
    assistant_id = await session.append_message(assistant("original"))
    result_id = await session.append_message(
        ToolResultMessage(
            tool_call_id="call-1",
            tool_name="read",
            content=[TextContent(text="original result")],
            is_error=False,
            timestamp=_now(),
        )
    )
    assistant_edit_id = await session.append_context_edit(assistant_id, {"content": "assistant replacement"})
    result_edit_id = await session.append_context_edit(result_id, {"content": "result replacement"})

    assert session.get_entry(assistant_edit_id)["replacement"] == {
        "content": [TextContent(text="assistant replacement")]
    }
    assert session.get_entry(result_edit_id)["replacement"] == {"content": [TextContent(text="result replacement")]}
    projected = session.build_session_projection().messages
    assert (projected[0].role, projected[0].content) == ("assistant", [TextContent(text="assistant replacement")])
    assert (projected[1].role, projected[1].content) == ("toolResult", [TextContent(text="result replacement")])


@pytest.mark.tonio
async def test_normalizes_imported_string_replacements_while_projecting_array_only_roles():
    session = SessionManager.in_memory()
    assistant_id = await session.append_message(assistant("original"))
    edit_id = await session.append_context_edit(assistant_id, None)
    edit = session.get_entry(edit_id)
    assert edit is not None and edit["type"] == "context_edit", "expected context edit"
    edit["replacement"] = {"content": "imported replacement"}

    projected = session.build_session_projection().messages[0]
    assert (projected.role, projected.content) == ("assistant", [TextContent(text="imported replacement")])


@pytest.mark.tonio
async def test_keeps_edits_branch_relative():
    session = SessionManager.in_memory()
    target_id = await session.append_message(user("original"))
    await session.append_context_edit(target_id, {"content": "edited"})
    assert text(session.build_session_projection().messages[0]) == "edited"

    session.branch(target_id)
    assert text(session.build_session_projection().messages[0]) == "original"


@pytest.mark.tonio
async def test_uses_a_self_referencing_compaction_to_retain_no_preceding_entries():
    session = SessionManager.in_memory()
    await session.append_message(user("discarded"))
    compaction_id = await session.append_compaction("exact handoff", None, 100)
    await session.append_message(user("after"))

    compaction = session.get_entry(compaction_id)
    assert (compaction["type"], compaction["firstKeptEntryId"]) == ("compaction", compaction_id)
    messages = session.build_session_projection().messages
    assert [message.role for message in messages] == ["compactionSummary", "user"]
    assert [_summary_or_text(message) for message in messages] == ["exact handoff", "after"]


@pytest.mark.tonio
async def test_applies_post_compaction_edits_to_retained_pre_compaction_entries():
    session = SessionManager.in_memory()
    await session.append_message(user("summarized"))
    retained_id = await session.append_message(user("original retained"))
    await session.append_compaction("summary", retained_id, 100)
    await session.append_context_edit(retained_id, {"content": "edited retained"})

    assert [_summary_or_text(message) for message in session.build_session_projection().messages] == [
        "summary",
        "edited retained",
    ]


@pytest.mark.tonio
async def test_uses_only_the_newest_summary_when_a_repeated_compaction_retains_entries_before_the_older_one():
    session = SessionManager.in_memory()
    await session.append_message(user("summarized first"))
    retained_id = await session.append_message(user("retained"))
    await session.append_compaction("first summary", retained_id, 100)
    await session.append_message(assistant("after first compaction"))
    await session.append_compaction("second summary", retained_id, 80)
    await session.append_message(user("new tail " * 100))

    summaries = [
        message.summary for message in session.build_session_projection().messages if hasattr(message, "summary")
    ]
    assert summaries == ["second summary"]
    preparation = prepare_compaction(session.get_branch(), _keep_one_token())
    assert preparation is not None
    assert preparation.previous_summary == "second summary"


@pytest.mark.tonio
async def test_supports_repeated_retain_none_compactions():
    session = SessionManager.in_memory()
    await session.append_message(user("discarded"))
    await session.append_compaction("first handoff", None, 100)
    await session.append_message(user("also discarded"))
    second_id = await session.append_compaction("second handoff", None, 50)

    assert session.get_entry(second_id)["firstKeptEntryId"] == second_id
    assert [getattr(message, "summary", "") for message in session.build_session_projection().messages] == [
        "second handoff"
    ]


@pytest.mark.tonio
async def test_does_not_trust_pre_edit_assistant_usage_for_projected_context_estimates():
    session = SessionManager.in_memory()
    large_user_id = await session.append_message(user("discarded input " * 2_000))
    response = assistant("small answer")
    response = dataclasses.replace(
        response, usage=dataclasses.replace(response.usage, input=10_000, total_tokens=10_001)
    )
    assistant_id = await session.append_message(response)
    await session.append_context_edit(large_user_id, None)

    edited_estimate = estimate_projected_context_tokens(session.build_session_projection(), session.get_branch())
    assert edited_estimate.usage_tokens == 0
    assert edited_estimate.tokens < 100

    await session.append_context_edit(assistant_id, None)
    assert estimate_projected_context_tokens(session.build_session_projection(), session.get_branch()).tokens == 0


@pytest.mark.tonio
async def test_uses_assistant_usage_captured_after_the_latest_context_edit():
    session = SessionManager.in_memory()
    user_id = await session.append_message(user("original"))
    await session.append_context_edit(user_id, {"content": "edited"})
    response = assistant("answer")
    response = dataclasses.replace(
        response, usage=dataclasses.replace(response.usage, input=4_000, output=100, total_tokens=4_100)
    )
    await session.append_message(response)
    await session.append_message(user("next"))

    estimate = estimate_projected_context_tokens(session.build_session_projection(), session.get_branch())
    assert estimate.usage_tokens == 4_100
    assert estimate.trailing_tokens == 1
    assert estimate.tokens == 4_101


@pytest.mark.tonio
async def test_does_not_reuse_post_edit_assistant_usage_after_a_later_compaction():
    session = SessionManager.in_memory()
    user_id = await session.append_message(user("small input"))
    await session.append_context_edit(user_id, {"content": "edited input"})
    response = assistant("answer")
    response = dataclasses.replace(
        response, usage=dataclasses.replace(response.usage, input=50_000, output=1, total_tokens=50_001)
    )
    await session.append_message(response)
    await session.append_compaction("small summary", user_id, 50_001)

    estimate = estimate_projected_context_tokens(session.build_session_projection(), session.get_branch())
    assert estimate.usage_tokens == 0
    assert estimate.tokens < 100


@pytest.mark.tonio
async def test_includes_effective_system_and_tool_context_in_edited_estimates():
    session = SessionManager.in_memory()
    await session.append_message(
        SystemMessage(
            content="system prompt " * 3_000,
            tools_added=[
                Tool(
                    name="example",
                    description="tool declaration " * 100,
                    parameters={"type": "object", "properties": {}},
                )
            ],
            timestamp=_now(),
        )
    )
    user_id = await session.append_message(user("ask"))
    await session.append_message(assistant("done"))
    await session.append_context_edit(user_id, {"content": "ask"})

    assert estimate_projected_context_tokens(session.build_session_projection(), session.get_branch()).tokens > 10_000


@pytest.mark.tonio
async def test_does_not_advance_past_a_boundary_replacement_of_the_candidate_input():
    session = SessionManager.in_memory()
    await session.append_message(user("old request"))
    await session.append_message(assistant("old answer"))
    replaced_user_id = await session.append_message(user("original input"))
    assistant_id = await session.append_message(assistant("answered original input"))
    await session.append_context_edit(replaced_user_id, {"content": "NEW-INSTRUCTION " * 100})
    await session.append_context_edit(assistant_id, None)
    await session.append_custom_entry("bookkeeping", {"source": "test"})

    preparation = prepare_compaction(session.get_branch(), _keep_one_token())
    assert preparation is not None
    assert preparation.first_kept_entry_id == replaced_user_id
    assert "NEW-INSTRUCTION" not in repr(preparation.messages_to_summarize)
    assert "NEW-INSTRUCTION" not in repr(preparation.turn_prefix_messages)


@pytest.mark.tonio
async def test_does_not_let_metadata_move_the_cut_past_unsent_boundary_input():
    session = SessionManager.in_memory()
    await session.append_message(user("old request"))
    await session.append_message(assistant("old answer"))
    instruction_id = await session.append_custom_message_entry("next-work", "UNSENT-INSTRUCTION " * 100, False)
    await session.append_custom_entry("bookkeeping", {"source": "test"})

    preparation = prepare_compaction(session.get_branch(), _keep_one_token())
    assert preparation is not None
    assert preparation.first_kept_entry_id == instruction_id
    assert "UNSENT-INSTRUCTION" not in repr(preparation.messages_to_summarize)
    assert "UNSENT-INSTRUCTION" not in repr(preparation.turn_prefix_messages)


@pytest.mark.tonio
async def test_does_not_treat_an_omitted_custom_message_as_a_recovery_attempt():
    session = SessionManager.in_memory()
    await session.append_message(user("unanswered input " * 100))
    custom_id = await session.append_custom_message_entry("temporary", "temporary context", False)
    await session.append_context_edit(custom_id, None)

    assert prepare_compaction(session.get_branch(), _keep_one_token()) is None


@pytest.mark.tonio
async def test_advances_past_input_for_an_omitted_assistant_recovery_suffix_with_metadata():
    session = SessionManager.in_memory()
    user_id = await session.append_message(user("recovery input " * 100))
    attempt_id = await session.append_message(assistant("failed attempt"))
    await session.append_context_edit(attempt_id, None)
    await session.append_custom_entry("bookkeeping", {"source": "test"})

    preparation = prepare_compaction(session.get_branch(), _keep_one_token())
    assert preparation is not None
    assert preparation.first_kept_entry_id == attempt_id
    assert len(preparation.turn_prefix_messages) == 1
    prefix = preparation.turn_prefix_messages[0]
    assert prefix.role == "user" and "recovery input" in prefix.content
    assert not any(
        message.role == "user" and "recovery input" in message.content for message in preparation.messages_to_summarize
    )
    assert user_id != attempt_id


@pytest.mark.tonio
async def test_prepares_compaction_from_edited_model_content():
    session = SessionManager.in_memory()
    omitted_id = await session.append_message(user("OMIT-ME " * 100))
    await session.append_message(assistant("old answer " * 100))
    await session.append_context_edit(omitted_id, None)
    await session.append_message(user("keep"))
    await session.append_message(assistant("suffix"))

    preparation = prepare_compaction(session.get_branch(), _keep_one_token())
    assert preparation is not None
    assert "OMIT-ME" not in repr(preparation.messages_to_summarize)
    assert "OMIT-ME" not in repr(preparation.turn_prefix_messages)
