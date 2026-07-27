"""Mirrors pi coding-agent test/session-manager/{build-context,tree-traversal,
save-entry,migration,custom-session-id,file-operations,labels}.test.ts."""

import json
import os
import re
import time

import pytest
import tonio.colored as tonio

from pidrei.core.session_manager import (
    SessionContextModel,
    SessionManager,
    build_context_entries,
    build_session_context,
    find_most_recent_session,
    load_entries_from_file,
    migrate_session_entries,
)
from pidrei_ai.types import UserMessage

from .coding_session_helpers import assistant_msg, make_usage, tool_result_msg, user_msg


UUID_V7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")

HEADER_SCAN_LIMIT_BYTES = 1024 * 1024


def msg(entry_id, parent_id, role, text):
    base = {"type": "message", "id": entry_id, "parentId": parent_id, "timestamp": "2025-01-01T00:00:00Z"}
    if role == "user":
        return {**base, "message": UserMessage(content=text, timestamp=1)}
    return {
        **base,
        "message": assistant_msg(text, model="claude-test", timestamp=1),
    }


def compaction(entry_id, parent_id, summary, first_kept_entry_id):
    return {
        "type": "compaction",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2025-01-01T00:00:00Z",
        "summary": summary,
        "firstKeptEntryId": first_kept_entry_id,
        "tokensBefore": 1000,
    }


def branch_summary(entry_id, parent_id, summary, from_id):
    return {
        "type": "branch_summary",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2025-01-01T00:00:00Z",
        "summary": summary,
        "fromId": from_id,
    }


def custom(entry_id, parent_id, custom_type, data=None):
    return {
        "type": "custom",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2025-01-01T00:00:00Z",
        "customType": custom_type,
        "data": data,
    }


def thinking_level(entry_id, parent_id, level):
    return {
        "type": "thinking_level_change",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2025-01-01T00:00:00Z",
        "thinkingLevel": level,
    }


def model_change(entry_id, parent_id, provider, model_id):
    return {
        "type": "model_change",
        "id": entry_id,
        "parentId": parent_id,
        "timestamp": "2025-01-01T00:00:00Z",
        "provider": provider,
        "modelId": model_id,
    }


class TestBuildSessionContextTrivial:
    def test_empty_entries_returns_empty_context(self):
        ctx = build_session_context([])
        assert ctx.messages == []
        assert ctx.thinking_level == "off"
        assert ctx.model is None

    def test_single_user_message(self):
        ctx = build_session_context([msg("1", None, "user", "hello")])
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "user"

    def test_simple_conversation(self):
        entries = [
            msg("1", None, "user", "hello"),
            msg("2", "1", "assistant", "hi there"),
            msg("3", "2", "user", "how are you"),
            msg("4", "3", "assistant", "great"),
        ]
        ctx = build_session_context(entries)
        assert len(ctx.messages) == 4
        assert [m.role for m in ctx.messages] == ["user", "assistant", "user", "assistant"]

    def test_tracks_thinking_level_changes(self):
        entries = [
            msg("1", None, "user", "hello"),
            thinking_level("2", "1", "high"),
            msg("3", "2", "assistant", "thinking hard"),
        ]
        ctx = build_session_context(entries)
        assert ctx.thinking_level == "high"
        assert len(ctx.messages) == 2

    def test_tracks_model_from_assistant_message(self):
        entries = [msg("1", None, "user", "hello"), msg("2", "1", "assistant", "hi")]
        ctx = build_session_context(entries)
        assert ctx.model == SessionContextModel(provider="anthropic", model_id="claude-test")

    def test_tracks_model_from_model_change_entry(self):
        entries = [
            msg("1", None, "user", "hello"),
            model_change("2", "1", "openai", "gpt-4"),
            msg("3", "2", "assistant", "hi"),
        ]
        ctx = build_session_context(entries)
        # Assistant message overwrites model change
        assert ctx.model == SessionContextModel(provider="anthropic", model_id="claude-test")


class TestBuildSessionContextCompaction:
    def test_includes_summary_before_kept_messages(self):
        entries = [
            msg("1", None, "user", "first"),
            msg("2", "1", "assistant", "response1"),
            msg("3", "2", "user", "second"),
            msg("4", "3", "assistant", "response2"),
            compaction("5", "4", "Summary of first two turns", "3"),
            msg("6", "5", "user", "third"),
            msg("7", "6", "assistant", "response3"),
        ]
        ctx = build_session_context(entries)

        # Summary + kept (3,4) + after (6,7) = 5 messages
        assert len(ctx.messages) == 5
        assert "Summary of first two turns" in ctx.messages[0].summary
        assert ctx.messages[1].content == "second"
        assert ctx.messages[2].content[0].text == "response2"
        assert ctx.messages[3].content == "third"
        assert ctx.messages[4].content[0].text == "response3"

    def test_handles_compaction_keeping_from_first_message(self):
        entries = [
            msg("1", None, "user", "first"),
            msg("2", "1", "assistant", "response"),
            compaction("3", "2", "Empty summary", "1"),
            msg("4", "3", "user", "second"),
        ]
        ctx = build_session_context(entries)

        # Summary + all messages (1,2,4)
        assert len(ctx.messages) == 4
        assert "Empty summary" in ctx.messages[0].summary

    def test_multiple_compactions_uses_latest(self):
        entries = [
            msg("1", None, "user", "a"),
            msg("2", "1", "assistant", "b"),
            compaction("3", "2", "First summary", "1"),
            msg("4", "3", "user", "c"),
            msg("5", "4", "assistant", "d"),
            compaction("6", "5", "Second summary", "4"),
            msg("7", "6", "user", "e"),
        ]
        ctx = build_session_context(entries)

        # Should use second summary, keep from 4
        assert len(ctx.messages) == 4
        assert "Second summary" in ctx.messages[0].summary

    def test_build_context_entries_compaction_aware_including_custom_entries(self):
        entries = [
            msg("1", None, "user", "first"),
            custom("2", "1", "old-state", {"hidden": True}),
            msg("3", "2", "assistant", "response1"),
            custom("4", "3", "kept-card", {"title": "Kept"}),
            msg("5", "4", "user", "second"),
            compaction("6", "5", "Summary", "4"),
            custom("7", "6", "after-card", {"title": "After"}),
            msg("8", "7", "assistant", "response2"),
        ]

        assert [entry["id"] for entry in build_context_entries(entries)] == ["6", "4", "5", "7", "8"]
        ctx = build_session_context(entries)
        assert [message.role for message in ctx.messages] == ["compactionSummary", "user", "assistant"]

    def test_keeps_settings_from_full_path_after_compaction(self):
        entries = [
            msg("1", None, "user", "first"),
            thinking_level("2", "1", "high"),
            msg("3", "2", "assistant", "response1"),
            msg("4", "3", "user", "second"),
            compaction("5", "4", "Summary", "4"),
        ]

        ctx = build_session_context(entries)
        assert ctx.thinking_level == "high"
        assert [message.role for message in ctx.messages] == ["compactionSummary", "user"]


class TestBuildSessionContextBranches:
    def test_follows_path_to_specified_leaf(self):
        entries = [
            msg("1", None, "user", "start"),
            msg("2", "1", "assistant", "response"),
            msg("3", "2", "user", "branch A"),
            msg("4", "2", "user", "branch B"),
        ]

        ctx_a = build_session_context(entries, "3")
        assert len(ctx_a.messages) == 3
        assert ctx_a.messages[2].content == "branch A"

        ctx_b = build_session_context(entries, "4")
        assert len(ctx_b.messages) == 3
        assert ctx_b.messages[2].content == "branch B"

    def test_includes_branch_summary_in_path(self):
        entries = [
            msg("1", None, "user", "start"),
            msg("2", "1", "assistant", "response"),
            msg("3", "2", "user", "abandoned path"),
            branch_summary("4", "2", "Summary of abandoned work", "3"),
            msg("5", "4", "user", "new direction"),
        ]
        ctx = build_session_context(entries, "5")

        assert len(ctx.messages) == 4
        assert "Summary of abandoned work" in ctx.messages[2].summary
        assert ctx.messages[3].content == "new direction"

    def test_complex_tree_with_multiple_branches_and_compaction(self):
        entries = [
            msg("1", None, "user", "start"),
            msg("2", "1", "assistant", "r1"),
            msg("3", "2", "user", "q2"),
            msg("4", "3", "assistant", "r2"),
            compaction("5", "4", "Compacted history", "3"),
            msg("6", "5", "user", "q3"),
            msg("7", "6", "assistant", "r3"),
            # Abandoned branch from 3
            msg("8", "3", "user", "wrong path"),
            msg("9", "8", "assistant", "wrong response"),
            # Branch summary resuming from 3
            branch_summary("10", "3", "Tried wrong approach", "9"),
            msg("11", "10", "user", "better approach"),
        ]

        # Main path to 7: summary + kept(3,4) + after(6,7)
        ctx_main = build_session_context(entries, "7")
        assert len(ctx_main.messages) == 5
        assert "Compacted history" in ctx_main.messages[0].summary
        assert ctx_main.messages[1].content == "q2"
        assert ctx_main.messages[2].content[0].text == "r2"
        assert ctx_main.messages[3].content == "q3"
        assert ctx_main.messages[4].content[0].text == "r3"

        # Branch path to 11: 1,2,3 + branch_summary + 11
        ctx_branch = build_session_context(entries, "11")
        assert len(ctx_branch.messages) == 5
        assert ctx_branch.messages[0].content == "start"
        assert ctx_branch.messages[1].content[0].text == "r1"
        assert ctx_branch.messages[2].content == "q2"
        assert "Tried wrong approach" in ctx_branch.messages[3].summary
        assert ctx_branch.messages[4].content == "better approach"


class TestBuildSessionContextEdgeCases:
    def test_uses_last_entry_when_leaf_id_not_found(self):
        entries = [msg("1", None, "user", "hello"), msg("2", "1", "assistant", "hi")]
        ctx = build_session_context(entries, "nonexistent")
        assert len(ctx.messages) == 2

    def test_handles_orphaned_entries_gracefully(self):
        entries = [
            msg("1", None, "user", "hello"),
            msg("2", "missing", "assistant", "orphan"),  # parent doesn't exist
        ]
        ctx = build_session_context(entries, "2")
        # Only the orphan since the parent chain is broken
        assert len(ctx.messages) == 1


class TestAppendOperations:
    @pytest.mark.tonio
    async def test_append_message_creates_entry_with_correct_parent_chain(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("first"))
        id2 = await session.append_message(assistant_msg("second"))
        id3 = await session.append_message(user_msg("third"))

        entries = session.get_entries()
        assert len(entries) == 3

        assert entries[0]["id"] == id1
        assert entries[0]["parentId"] is None
        assert entries[0]["type"] == "message"

        assert entries[1]["id"] == id2
        assert entries[1]["parentId"] == id1

        assert entries[2]["id"] == id3
        assert entries[2]["parentId"] == id2

    @pytest.mark.tonio
    async def test_append_thinking_level_change_integrates_into_tree(self):
        session = SessionManager.in_memory()

        msg_id = await session.append_message(user_msg("hello"))
        thinking_id = await session.append_thinking_level_change("high")
        await session.append_message(assistant_msg("response"))

        entries = session.get_entries()
        assert len(entries) == 3

        thinking_entry = next(e for e in entries if e["type"] == "thinking_level_change")
        assert thinking_entry["id"] == thinking_id
        assert thinking_entry["parentId"] == msg_id

        assert entries[2]["parentId"] == thinking_id

    @pytest.mark.tonio
    async def test_append_model_change_integrates_into_tree(self):
        session = SessionManager.in_memory()

        msg_id = await session.append_message(user_msg("hello"))
        model_id = await session.append_model_change("openai", "gpt-4")
        await session.append_message(assistant_msg("response"))

        entries = session.get_entries()
        model_entry = next(e for e in entries if e["type"] == "model_change")
        assert model_entry["id"] == model_id
        assert model_entry["parentId"] == msg_id
        assert model_entry["provider"] == "openai"
        assert model_entry["modelId"] == "gpt-4"

        assert entries[2]["parentId"] == model_id

    @pytest.mark.tonio
    async def test_append_compaction_integrates_into_tree(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))
        usage = make_usage()
        compaction_id = await session.append_compaction("summary", id1, 1000, None, False, usage)
        await session.append_message(user_msg("3"))

        entries = session.get_entries()
        compaction_entry = next(e for e in entries if e["type"] == "compaction")
        assert compaction_entry["id"] == compaction_id
        assert compaction_entry["parentId"] == id2
        assert compaction_entry["summary"] == "summary"
        assert compaction_entry["firstKeptEntryId"] == id1
        assert compaction_entry["tokensBefore"] == 1000
        assert compaction_entry["usage"] == usage

        assert entries[3]["parentId"] == compaction_id

    @pytest.mark.tonio
    async def test_append_custom_entry_integrates_into_tree(self):
        session = SessionManager.in_memory()

        msg_id = await session.append_message(user_msg("hello"))
        custom_id = await session.append_custom_entry("my_data", {"key": "value"})
        await session.append_message(assistant_msg("response"))

        entries = session.get_entries()
        custom_entry = next(e for e in entries if e["type"] == "custom")
        assert custom_entry["id"] == custom_id
        assert custom_entry["parentId"] == msg_id
        assert custom_entry["customType"] == "my_data"
        assert custom_entry["data"] == {"key": "value"}

        assert entries[2]["parentId"] == custom_id

    @pytest.mark.tonio
    async def test_leaf_pointer_advances_after_each_append(self):
        session = SessionManager.in_memory()

        assert session.get_leaf_id() is None

        id1 = await session.append_message(user_msg("1"))
        assert session.get_leaf_id() == id1

        id2 = await session.append_message(assistant_msg("2"))
        assert session.get_leaf_id() == id2

        id3 = await session.append_thinking_level_change("high")
        assert session.get_leaf_id() == id3


class TestGetPath:
    def test_returns_empty_for_empty_session(self):
        session = SessionManager.in_memory()
        assert session.get_branch() == []

    @pytest.mark.tonio
    async def test_returns_single_entry_path(self):
        session = SessionManager.in_memory()
        entry_id = await session.append_message(user_msg("hello"))

        path = session.get_branch()
        assert len(path) == 1
        assert path[0]["id"] == entry_id

    @pytest.mark.tonio
    async def test_returns_full_path_from_root_to_leaf(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))
        id3 = await session.append_thinking_level_change("high")
        id4 = await session.append_message(user_msg("3"))

        path = session.get_branch()
        assert [e["id"] for e in path] == [id1, id2, id3, id4]

    @pytest.mark.tonio
    async def test_returns_path_from_specified_entry_to_root(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))
        await session.append_message(user_msg("3"))
        await session.append_message(assistant_msg("4"))

        path = session.get_branch(id2)
        assert [e["id"] for e in path] == [id1, id2]


class TestGetTree:
    def test_returns_empty_for_empty_session(self):
        session = SessionManager.in_memory()
        assert session.get_tree() == []

    @pytest.mark.tonio
    async def test_returns_single_root_for_linear_session(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))
        id3 = await session.append_message(user_msg("3"))

        tree = session.get_tree()
        assert len(tree) == 1

        root = tree[0]
        assert root.entry["id"] == id1
        assert len(root.children) == 1
        assert root.children[0].entry["id"] == id2
        assert len(root.children[0].children) == 1
        assert root.children[0].children[0].entry["id"] == id3
        assert root.children[0].children[0].children == []

    @pytest.mark.tonio
    async def test_returns_tree_with_branches_after_branch(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))
        id3 = await session.append_message(user_msg("3"))

        session.branch(id2)
        id4 = await session.append_message(user_msg("4-branch"))

        tree = session.get_tree()
        assert len(tree) == 1

        root = tree[0]
        assert root.entry["id"] == id1
        assert len(root.children) == 1

        node2 = root.children[0]
        assert node2.entry["id"] == id2
        assert len(node2.children) == 2  # id3 and id4 are siblings

        child_ids = sorted(c.entry["id"] for c in node2.children)
        assert child_ids == sorted([id3, id4])

    @pytest.mark.tonio
    async def test_handles_multiple_branches_at_same_point(self):
        session = SessionManager.in_memory()

        await session.append_message(user_msg("root"))
        id2 = await session.append_message(assistant_msg("response"))

        session.branch(id2)
        id_a = await session.append_message(user_msg("branch-A"))

        session.branch(id2)
        id_b = await session.append_message(user_msg("branch-B"))

        session.branch(id2)
        id_c = await session.append_message(user_msg("branch-C"))

        tree = session.get_tree()
        node2 = tree[0].children[0]
        assert node2.entry["id"] == id2
        assert len(node2.children) == 3

        branch_ids = sorted(c.entry["id"] for c in node2.children)
        assert branch_ids == sorted([id_a, id_b, id_c])

    @pytest.mark.tonio
    async def test_handles_deep_branching(self):
        session = SessionManager.in_memory()

        await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))
        id3 = await session.append_message(user_msg("3"))
        await session.append_message(assistant_msg("4"))

        session.branch(id2)
        id5 = await session.append_message(user_msg("5"))
        await session.append_message(assistant_msg("6"))

        session.branch(id5)
        await session.append_message(user_msg("7"))

        tree = session.get_tree()

        node2 = tree[0].children[0]
        assert len(node2.children) == 2  # id3 and id5

        node5 = next(c for c in node2.children if c.entry["id"] == id5)
        assert len(node5.children) == 2  # id6 and id7

        node3 = next(c for c in node2.children if c.entry["id"] == id3)
        assert len(node3.children) == 1  # id4


class TestBranch:
    @pytest.mark.tonio
    async def test_moves_leaf_pointer_to_specified_entry(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        await session.append_message(assistant_msg("2"))
        id3 = await session.append_message(user_msg("3"))

        assert session.get_leaf_id() == id3

        session.branch(id1)
        assert session.get_leaf_id() == id1

    @pytest.mark.tonio
    async def test_throws_for_non_existent_entry(self):
        session = SessionManager.in_memory()
        await session.append_message(user_msg("hello"))

        with pytest.raises(Exception, match="Entry nonexistent not found"):
            session.branch("nonexistent")

    @pytest.mark.tonio
    async def test_new_appends_become_children_of_branch_point(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        await session.append_message(assistant_msg("2"))

        session.branch(id1)
        id3 = await session.append_message(user_msg("branched"))

        branched_entry = next(e for e in session.get_entries() if e["id"] == id3)
        assert branched_entry["parentId"] == id1  # sibling of id2


class TestBranchWithSummary:
    @pytest.mark.tonio
    async def test_inserts_branch_summary_and_advances_leaf(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        await session.append_message(assistant_msg("2"))
        await session.append_message(user_msg("3"))

        usage = make_usage()
        summary_id = await session.branch_with_summary(id1, "Summary of abandoned work", None, False, usage)

        assert session.get_leaf_id() == summary_id

        summary_entry = next(e for e in session.get_entries() if e["type"] == "branch_summary")
        assert summary_entry["parentId"] == id1
        assert summary_entry["summary"] == "Summary of abandoned work"
        assert summary_entry["usage"] == usage

    @pytest.mark.tonio
    async def test_throws_for_non_existent_entry(self):
        session = SessionManager.in_memory()
        await session.append_message(user_msg("hello"))

        with pytest.raises(Exception, match="Entry nonexistent not found"):
            await session.branch_with_summary("nonexistent", "summary")


class TestLeafAndEntryAccess:
    def test_get_leaf_entry_returns_none_for_empty_session(self):
        session = SessionManager.in_memory()
        assert session.get_leaf_entry() is None

    @pytest.mark.tonio
    async def test_get_leaf_entry_returns_current_leaf(self):
        session = SessionManager.in_memory()

        await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))

        leaf = session.get_leaf_entry()
        assert leaf is not None
        assert leaf["id"] == id2

    def test_get_entry_returns_none_for_non_existent(self):
        session = SessionManager.in_memory()
        assert session.get_entry("nonexistent") is None

    @pytest.mark.tonio
    async def test_get_entry_returns_entry_by_id(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("first"))
        id2 = await session.append_message(assistant_msg("second"))

        entry1 = session.get_entry(id1)
        assert entry1 is not None
        assert entry1["type"] == "message"
        assert entry1["message"].content == "first"

        entry2 = session.get_entry(id2)
        assert entry2 is not None
        assert entry2["message"].content[0].text == "second"


class TestBuildSessionContextWithBranchesManager:
    @pytest.mark.tonio
    async def test_returns_messages_from_current_branch_only(self):
        session = SessionManager.in_memory()

        await session.append_message(user_msg("msg1"))
        id2 = await session.append_message(assistant_msg("msg2"))
        await session.append_message(user_msg("msg3"))

        session.branch(id2)
        await session.append_message(assistant_msg("msg4-branch"))

        ctx = session.build_session_context()
        assert len(ctx.messages) == 3  # msg1, msg2, msg4-branch (not msg3)

        assert ctx.messages[0].content == "msg1"
        assert ctx.messages[1].content[0].text == "msg2"
        assert ctx.messages[2].content[0].text == "msg4-branch"


class TestCreateBranchedSession:
    @pytest.mark.tonio
    async def test_throws_for_non_existent_entry(self):
        session = SessionManager.in_memory()
        await session.append_message(user_msg("hello"))

        with pytest.raises(Exception, match="Entry nonexistent not found"):
            await session.create_branched_session("nonexistent")

    @pytest.mark.tonio
    async def test_creates_new_session_with_path_to_leaf_in_memory(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))
        id3 = await session.append_message(user_msg("3"))
        await session.append_message(assistant_msg("4"))

        session.branch(id3)
        await session.append_message(user_msg("5"))

        result = await session.create_branched_session(id2)
        assert result is None  # in-memory returns None

        entries = session.get_entries()
        assert [e["id"] for e in entries] == [id1, id2]

    @pytest.mark.tonio
    async def test_extracts_correct_path_from_branched_tree(self):
        session = SessionManager.in_memory()

        id1 = await session.append_message(user_msg("1"))
        id2 = await session.append_message(assistant_msg("2"))
        await session.append_message(user_msg("3"))

        session.branch(id2)
        id4 = await session.append_message(user_msg("4"))
        id5 = await session.append_message(assistant_msg("5"))

        await session.create_branched_session(id5)

        entries = session.get_entries()
        assert [e["id"] for e in entries] == [id1, id2, id4, id5]

    @pytest.mark.tonio
    async def test_does_not_duplicate_entries_when_forking_from_first_user_message(self, tmp_dir):
        temp_dir = str(tmp_dir)
        session = await SessionManager.create(temp_dir, temp_dir)
        id1 = await session.append_message(user_msg("first question"))
        await session.append_message(assistant_msg("first answer"))
        await session.append_message(user_msg("second question"))
        await session.append_message(assistant_msg("second answer"))

        # Fork from the very first user message (no assistant in the branched path)
        new_file = await session.create_branched_session(id1)
        assert new_file is not None

        # The branched path has no assistant, so the file should not exist yet
        # (deferred to persist on first assistant, matching new_session() contract)
        assert not os.path.exists(new_file)

        # Simulate extension adding entry before assistant
        await session.append_custom_entry("preset-state", {"name": "plan"})

        # Now the assistant responds
        await session.append_message(assistant_msg("new answer"))

        # File should now exist with exactly one header and no duplicate IDs
        assert os.path.exists(new_file)
        with open(new_file, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle.read().strip().split("\n") if line]

        assert len([r for r in records if r["type"] == "session"]) == 1

        entry_ids = [r["id"] for r in records if r["type"] != "session" and isinstance(r.get("id"), str)]
        assert len(set(entry_ids)) == len(entry_ids)

    @pytest.mark.tonio
    async def test_preserves_tool_and_summary_usage_across_file_backed_reload(self, tmp_dir):
        temp_dir = str(tmp_dir)
        session = await SessionManager.create(temp_dir, temp_dir)
        root_id = await session.append_message(user_msg("question"))
        await session.append_message(assistant_msg("answer"))
        usage = make_usage()
        await session.append_message(tool_result_msg("result", usage=usage))
        await session.append_compaction("summary", root_id, 100, None, False, usage)
        await session.branch_with_summary(root_id, "branch summary", None, False, usage)

        file = session.get_session_file()
        assert file is not None
        reopened = await SessionManager.open(file, temp_dir)
        entries = reopened.get_entries()
        compaction_entry = next(e for e in entries if e["type"] == "compaction")
        assert compaction_entry["usage"] == usage
        branch_entry = next(e for e in entries if e["type"] == "branch_summary")
        assert branch_entry["usage"] == usage
        tool_entry = next(
            e for e in entries if e["type"] == "message" and getattr(e["message"], "role", None) == "toolResult"
        )
        assert tool_entry["message"].usage == usage

    @pytest.mark.tonio
    async def test_writes_file_immediately_when_forking_from_point_with_assistant(self, tmp_dir):
        temp_dir = str(tmp_dir)
        session = await SessionManager.create(temp_dir, temp_dir)
        await session.append_message(user_msg("first question"))
        id2 = await session.append_message(assistant_msg("first answer"))
        await session.append_message(user_msg("second question"))
        await session.append_message(assistant_msg("second answer"))

        new_file = await session.create_branched_session(id2)
        assert new_file is not None

        # Path includes an assistant, so file should be written immediately
        assert os.path.exists(new_file)
        with open(new_file, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle.read().strip().split("\n") if line]
        assert len([r for r in records if r["type"] == "session"]) == 1


class TestSaveCustomEntry:
    @pytest.mark.tonio
    async def test_saves_custom_entries_and_includes_in_tree_traversal(self):
        session = SessionManager.in_memory()

        msg_id = await session.append_message(UserMessage(content="hello", timestamp=1))
        custom_id = await session.append_custom_entry("my_data", {"foo": "bar"})
        msg2_id = await session.append_message(assistant_msg("hi", timestamp=2))

        entries = session.get_entries()
        assert len(entries) == 3

        custom_entry = next(e for e in entries if e["type"] == "custom")
        assert custom_entry["customType"] == "my_data"
        assert custom_entry["data"] == {"foo": "bar"}
        assert custom_entry["id"] == custom_id
        assert custom_entry["parentId"] == msg_id

        path = session.get_branch()
        assert [e["id"] for e in path] == [msg_id, custom_id, msg2_id]

        # build_session_context should work (custom entries skipped in messages)
        ctx = session.build_session_context()
        assert len(ctx.messages) == 2  # only message entries


class TestMigration:
    def test_adds_id_parent_id_to_v1_entries(self):
        entries = [
            {"type": "session", "id": "sess-1", "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp"},
            {
                "type": "message",
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {"role": "user", "content": "hi", "timestamp": 1},
            },
            {
                "type": "message",
                "timestamp": "2025-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "api": "test",
                    "provider": "test",
                    "model": "test",
                    "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                    "stopReason": "stop",
                    "timestamp": 2,
                },
            },
        ]

        migrate_session_entries(entries)

        # Header should have version set (v3 is current after hookMessage->custom migration)
        assert entries[0]["version"] == 3

        msg1, msg2 = entries[1], entries[2]
        assert isinstance(msg1["id"], str) and len(msg1["id"]) == 8
        assert msg1["parentId"] is None
        assert isinstance(msg2["id"], str) and len(msg2["id"]) == 8
        assert msg2["parentId"] == msg1["id"]

    def test_is_idempotent(self):
        entries = [
            {"type": "session", "id": "sess-1", "version": 2, "timestamp": "2025-01-01T00:00:00Z", "cwd": "/tmp"},
            {
                "type": "message",
                "id": "abc12345",
                "parentId": None,
                "timestamp": "2025-01-01T00:00:01Z",
                "message": {"role": "user", "content": "hi", "timestamp": 1},
            },
            {
                "type": "message",
                "id": "def67890",
                "parentId": "abc12345",
                "timestamp": "2025-01-01T00:00:02Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "api": "test",
                    "provider": "test",
                    "model": "test",
                    "usage": {"input": 1, "output": 1, "cacheRead": 0, "cacheWrite": 0},
                    "stopReason": "stop",
                    "timestamp": 2,
                },
            },
        ]

        migrate_session_entries(entries)

        assert entries[1]["id"] == "abc12345"
        assert entries[2]["id"] == "def67890"
        assert entries[2]["parentId"] == "abc12345"


class TestCustomSessionId:
    def test_uses_provided_id(self):
        session = SessionManager.in_memory()
        session.new_session({"id": "my-custom-id"})
        assert session.get_session_id() == "my-custom-id"

    def test_uses_provided_id_when_creating_in_memory_session(self):
        session = SessionManager.in_memory(os.getcwd(), {"id": "memory-session-id"})
        assert session.get_session_id() == "memory-session-id"
        assert session.get_header()["id"] == "memory-session-id"
        assert session.get_session_file() is None

    def test_allows_alphanumeric_ids_with_interior_punctuation(self):
        session = SessionManager.in_memory()
        session.new_session({"id": "abc-123_def.456"})
        assert session.get_session_id() == "abc-123_def.456"

    def test_rejects_invalid_custom_session_ids(self):
        invalid_ids = ["", "-abc", "abc-", "_abc", "abc_", ".abc", "abc.", "abc/def", "abc\\def", "abc def"]

        for invalid_id in invalid_ids:
            session = SessionManager.in_memory()
            with pytest.raises(Exception, match="Session id must be non-empty, contain only alphanumeric characters"):
                session.new_session({"id": invalid_id})

    def test_generates_uuidv7_when_no_id_provided(self):
        session = SessionManager.in_memory()
        session.new_session()
        assert UUID_V7_RE.match(session.get_session_id())

    def test_generates_uuidv7_when_options_without_id(self):
        session = SessionManager.in_memory()
        session.new_session({"parentSession": "parent.jsonl"})
        assert UUID_V7_RE.match(session.get_session_id())

    def test_includes_custom_id_in_header(self):
        session = SessionManager.in_memory()
        session.new_session({"id": "header-test-id"})

        header = session.get_header()
        assert header is not None
        assert header["id"] == "header-test-id"

    def test_generates_uuidv7_when_constructed_without_explicit_id(self):
        session = SessionManager.in_memory()
        assert UUID_V7_RE.match(session.get_session_id())
        assert session.get_header()["id"] == session.get_session_id()

    @pytest.mark.tonio
    async def test_uses_provided_id_when_creating_persisted_session(self, tmp_dir):
        temp_dir = str(tmp_dir)
        session = await SessionManager.create(temp_dir, temp_dir, {"id": "created-session-id"})

        assert session.get_session_id() == "created-session-id"
        assert session.get_header()["id"] == "created-session-id"
        session_file = session.get_session_file()
        assert "created-session-id" in session_file
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z_created-session-id\.jsonl$",
            os.path.basename(session_file),
        )
        assert not os.path.exists(session_file)

    @pytest.mark.tonio
    async def test_generates_uuidv7_when_creating_branched_session(self):
        session = SessionManager.in_memory()
        first_id = await session.append_message(user_msg("hello"))

        await session.create_branched_session(first_id)

        assert UUID_V7_RE.match(session.get_session_id())
        assert session.get_header()["id"] == session.get_session_id()

    @pytest.mark.tonio
    async def test_generates_uuidv7_when_forking_from_session_file(self, tmp_dir):
        temp_dir = str(tmp_dir)
        source_path = os.path.join(temp_dir, "source.jsonl")
        lines = [
            json.dumps(
                {
                    "type": "session",
                    "version": 3,
                    "id": "legacy-session-id",
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "cwd": temp_dir,
                }
            ),
            json.dumps(
                {
                    "type": "message",
                    "id": "entry-1",
                    "parentId": None,
                    "timestamp": "2025-01-01T00:00:00.000Z",
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hello"}],
                        "api": "openai-responses",
                        "provider": "openai",
                        "model": "gpt-5.4",
                        "usage": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                            "totalTokens": 0,
                            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
                        },
                        "stopReason": "stop",
                        "timestamp": 1,
                    },
                }
            ),
        ]
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")

        forked = await SessionManager.fork_from(source_path, temp_dir, temp_dir)
        header = forked.get_header()
        assert header is not None
        assert UUID_V7_RE.match(header["id"])
        assert header["parentSession"] == source_path

    @pytest.mark.tonio
    async def test_uses_provided_id_when_forking_from_session_file(self, tmp_dir):
        temp_dir = str(tmp_dir)
        source_path = os.path.join(temp_dir, "source.jsonl")
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "session",
                        "version": 3,
                        "id": "source-session-id",
                        "timestamp": "2025-01-01T00:00:00.000Z",
                        "cwd": temp_dir,
                    }
                )
                + "\n"
            )

        forked = await SessionManager.fork_from(source_path, temp_dir, temp_dir, {"id": "forked-session-id"})
        header = forked.get_header()
        assert header["id"] == "forked-session-id"
        assert header["parentSession"] == source_path
        session_file = forked.get_session_file()
        assert "forked-session-id" in session_file
        assert re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z_forked-session-id\.jsonl$",
            os.path.basename(session_file),
        )


def _write_session_header(file: str, cwd: str, session_id: str, prefix: str = "") -> None:
    with open(file, "w", encoding="utf-8") as handle:
        handle.write(
            prefix
            + json.dumps(
                {"type": "session", "version": 3, "id": session_id, "timestamp": "2025-01-01T00:00:00Z", "cwd": cwd}
            )
            + "\n"
        )


class TestLoadEntriesFromFile:
    def test_returns_empty_for_non_existent_file(self, tmp_path):
        assert load_entries_from_file(os.path.join(str(tmp_path), "nonexistent.jsonl")) == []

    def test_returns_empty_for_empty_file(self, tmp_path):
        file = os.path.join(str(tmp_path), "empty.jsonl")
        open(file, "w").close()
        assert load_entries_from_file(file) == []

    def test_returns_empty_for_file_without_valid_header(self, tmp_path):
        file = os.path.join(str(tmp_path), "no-header.jsonl")
        with open(file, "w") as handle:
            handle.write('{"type":"message","id":"1"}\n')
        assert load_entries_from_file(file) == []

    def test_returns_empty_for_malformed_json(self, tmp_path):
        file = os.path.join(str(tmp_path), "malformed.jsonl")
        with open(file, "w") as handle:
            handle.write("not json\n")
        assert load_entries_from_file(file) == []

    def test_loads_valid_session_file(self, tmp_path):
        file = os.path.join(str(tmp_path), "valid.jsonl")
        with open(file, "w") as handle:
            handle.write(
                '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
                '{"type":"message","id":"1","parentId":null,"timestamp":"2025-01-01T00:00:01Z",'
                '"message":{"role":"user","content":"hi","timestamp":1}}\n'
            )
        entries = load_entries_from_file(file)
        assert len(entries) == 2
        assert entries[0]["type"] == "session"
        assert entries[1]["type"] == "message"

    def test_skips_malformed_lines_but_keeps_valid_ones(self, tmp_path):
        file = os.path.join(str(tmp_path), "mixed.jsonl")
        with open(file, "w") as handle:
            handle.write(
                '{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n'
                "not valid json\n"
                '{"type":"message","id":"1","parentId":null,"timestamp":"2025-01-01T00:00:01Z",'
                '"message":{"role":"user","content":"hi","timestamp":1}}\n'
            )
        assert len(load_entries_from_file(file)) == 2

    @pytest.mark.parametrize(
        ("prefix", "session_id"),
        [
            ("\n  \n", "leading-blank"),
            ("not json\n{broken json\n", "leading-malformed"),
            ("", "a" * 8192),
        ],
        ids=["leading-blank-lines", "leading-malformed-lines", "multi-buffer-header"],
    )
    @pytest.mark.tonio
    async def test_reads_cwd_from_session_headers(self, tmp_dir, prefix, session_id):
        temp_dir = str(tmp_dir)
        file = os.path.join(temp_dir, "header.jsonl")
        stored_cwd = os.path.join(temp_dir, "stored-project")
        _write_session_header(file, stored_cwd, session_id, prefix)

        session_manager = await SessionManager.open(file, temp_dir)
        assert session_manager.get_session_id() == session_id
        assert session_manager.get_cwd() == stored_cwd

    @pytest.mark.tonio
    async def test_opens_compatible_sessions_beyond_discovery_scan_limit(self, tmp_dir):
        temp_dir = str(tmp_dir)
        stored_cwd = os.path.join(temp_dir, "stored-project")
        override_cwd = os.path.join(temp_dir, "override-project")
        cases = [
            ("large-header", "a" * (HEADER_SCAN_LIMIT_BYTES + 1), ""),
            ("large-prefix", "large-prefix", "x" * (HEADER_SCAN_LIMIT_BYTES + 1) + "\n"),
        ]

        for name, session_id, prefix in cases:
            file = os.path.join(temp_dir, f"{name}.jsonl")
            _write_session_header(file, stored_cwd, session_id, prefix)
            for cwd_override in (None, override_cwd):
                session_manager = await SessionManager.open(file, temp_dir, cwd_override)
                assert session_manager.get_session_id() == session_id
                assert session_manager.get_cwd() == (cwd_override if cwd_override is not None else stored_cwd)

    @pytest.mark.tonio
    async def test_opens_session_files_with_many_chunks(self, tmp_dir):
        # pi's mirror writes past Node's max string length to prove chunked
        # reading; Python has no string cap, so a multi-buffer sparse file
        # exercises the same chunked line-splitting path at reduced cost.
        temp_dir = str(tmp_dir)
        file = os.path.join(temp_dir, "large.jsonl")
        with open(file, "w") as handle:
            handle.write('{"type":"session","version":3,"id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')

        stride = 16 * 1024 * 1024
        with open(file, "r+b") as handle:
            for offset in range(stride, 4 * stride + 1, stride):
                handle.seek(offset)
                handle.write(b"\n")

        with open(file, "a") as handle:
            handle.write(
                '{"type":"message","id":"1","parentId":null,"timestamp":"2025-01-01T00:00:01Z",'
                '"message":{"role":"user","content":"hi","timestamp":1}}\n'
            )

        session_manager = await SessionManager.open(file, temp_dir)
        assert session_manager.get_session_id() == "abc"
        assert len(session_manager.get_entries()) == 1
        assert session_manager.build_session_context().messages == [UserMessage(content="hi", timestamp=1)]


class TestFindMostRecentSession:
    def test_returns_none_for_empty_directory(self, tmp_path):
        assert find_most_recent_session(str(tmp_path)) is None

    def test_returns_none_for_non_existent_directory(self, tmp_path):
        assert find_most_recent_session(os.path.join(str(tmp_path), "nonexistent")) is None

    def test_ignores_non_jsonl_files(self, tmp_path):
        (tmp_path / "file.txt").write_text("hello")
        (tmp_path / "file.json").write_text("{}")
        assert find_most_recent_session(str(tmp_path)) is None

    def test_ignores_jsonl_without_valid_header(self, tmp_path):
        (tmp_path / "invalid.jsonl").write_text('{"type":"message"}\n')
        assert find_most_recent_session(str(tmp_path)) is None

    def test_returns_single_valid_session_file(self, tmp_path):
        file = tmp_path / "session.jsonl"
        file.write_text('{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')
        assert find_most_recent_session(str(tmp_path)) == str(file)

    def test_returns_most_recently_modified_session(self, tmp_path):
        file1 = tmp_path / "older.jsonl"
        file2 = tmp_path / "newer.jsonl"

        file1.write_text('{"type":"session","id":"old","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')
        time.sleep(0.01)
        file2.write_text('{"type":"session","id":"new","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')

        assert find_most_recent_session(str(tmp_path)) == str(file2)

    def test_skips_invalid_files_and_returns_valid_one(self, tmp_path):
        invalid = tmp_path / "invalid.jsonl"
        valid = tmp_path / "valid.jsonl"

        invalid.write_text('{"type":"not-session"}\n')
        time.sleep(0.01)
        valid.write_text('{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')

        assert find_most_recent_session(str(tmp_path)) == str(valid)

    def test_skips_oversized_corrupt_files_and_returns_valid_session(self, tmp_path):
        (tmp_path / "oversized.jsonl").write_text("x" * (HEADER_SCAN_LIMIT_BYTES + 1))
        valid = tmp_path / "valid.jsonl"
        valid.write_text('{"type":"session","id":"abc","timestamp":"2025-01-01T00:00:00Z","cwd":"/tmp"}\n')

        assert find_most_recent_session(str(tmp_path)) == str(valid)

    def test_filters_most_recent_session_by_cwd(self, tmp_path):
        project_a = os.path.join(str(tmp_path), "project-a")
        project_b = os.path.join(str(tmp_path), "project-b")
        file_a = tmp_path / "a.jsonl"
        file_b = tmp_path / "b.jsonl"

        file_a.write_text(
            json.dumps({"type": "session", "id": "a", "timestamp": "2025-01-01T00:00:00Z", "cwd": project_a}) + "\n"
        )
        time.sleep(0.01)
        file_b.write_text(
            json.dumps({"type": "session", "id": "b", "timestamp": "2025-01-01T00:00:00Z", "cwd": project_b}) + "\n"
        )

        assert find_most_recent_session(str(tmp_path), project_a) == str(file_a)
        assert find_most_recent_session(str(tmp_path), project_b) == str(file_b)


async def _create_persisted_session(cwd: str, session_dir: str, label: str) -> str:
    session = await SessionManager.create(cwd, session_dir)
    await session.append_message(user_msg(label))
    await session.append_message(assistant_msg(f"reply to {label}"))
    session_file = session.get_session_file()
    assert session_file is not None
    return session_file


@pytest.mark.tonio
async def test_scopes_current_folder_apis_by_cwd_while_listing_all_flat_sessions(tmp_dir):
    tmp_dir = str(tmp_dir)
    project_a = os.path.join(tmp_dir, "project-a")
    project_b = os.path.join(tmp_dir, "project-b")
    os.makedirs(project_a)
    os.makedirs(project_b)

    session_a = await _create_persisted_session(project_a, tmp_dir, "from A")
    await tonio.time.sleep(0.01)
    session_b = await _create_persisted_session(project_b, tmp_dir, "from B")

    current_a = await SessionManager.list(project_a, tmp_dir)
    assert [session.path for session in current_a] == [session_a]

    all_sessions = await SessionManager.list_all(tmp_dir)
    assert {session.path for session in all_sessions} == {session_a, session_b}

    continued_a = await SessionManager.continue_recent(project_a, tmp_dir)
    assert continued_a.get_session_file() == session_a


class TestSetSessionFileCorrupted:
    @pytest.mark.tonio
    async def test_truncates_and_rewrites_empty_file_with_valid_header(self, tmp_dir):
        empty_file = os.path.join(str(tmp_dir), "empty.jsonl")
        open(empty_file, "w").close()

        sm = await SessionManager.open(empty_file, str(tmp_dir))

        assert sm.get_session_id()
        assert sm.get_header() is not None
        assert sm.get_header()["type"] == "session"

        with open(empty_file, encoding="utf-8") as handle:
            lines = [line for line in handle.read().strip().split("\n") if line]
        assert len(lines) == 1
        header = json.loads(lines[0])
        assert header["type"] == "session"
        assert header["id"] == sm.get_session_id()

    @pytest.mark.tonio
    async def test_throws_and_preserves_non_empty_file_without_valid_header(self, tmp_dir):
        no_header_file = os.path.join(str(tmp_dir), "no-header.jsonl")
        original_content = (
            '{"type":"message","id":"abc","parentId":"orphaned","timestamp":"2025-01-01T00:00:00Z",'
            '"message":{"role":"assistant","content":"test"}}\n'
        )
        with open(no_header_file, "w") as handle:
            handle.write(original_content)

        with pytest.raises(Exception, match="Session file is not a valid pidrei session"):
            await SessionManager.open(no_header_file, str(tmp_dir))
        with open(no_header_file, encoding="utf-8") as handle:
            assert handle.read() == original_content

    @pytest.mark.tonio
    async def test_throws_and_preserves_non_session_jsonl_files(self, tmp_dir):
        non_session_file = os.path.join(str(tmp_dir), "not-a-session.log")
        original_content = '{"type":"event","data":"not a session"}\n'
        with open(non_session_file, "w") as handle:
            handle.write(original_content)

        with pytest.raises(Exception, match="Session file is not a valid pidrei session"):
            await SessionManager.open(non_session_file, str(tmp_dir))
        with open(non_session_file, encoding="utf-8") as handle:
            assert handle.read() == original_content

    @pytest.mark.tonio
    async def test_preserves_explicit_session_file_path_when_recovering(self, tmp_dir):
        explicit_path = os.path.join(str(tmp_dir), "my-session.jsonl")
        open(explicit_path, "w").close()

        sm = await SessionManager.open(explicit_path, str(tmp_dir))
        assert sm.get_session_file() == explicit_path

    @pytest.mark.tonio
    async def test_subsequent_loads_of_initialized_empty_file_work(self, tmp_dir):
        empty_file = os.path.join(str(tmp_dir), "empty.jsonl")
        open(empty_file, "w").close()

        sm1 = await SessionManager.open(empty_file, str(tmp_dir))
        session_id = sm1.get_session_id()

        sm2 = await SessionManager.open(empty_file, str(tmp_dir))
        assert sm2.get_session_id() == session_id
        assert sm2.get_header()["type"] == "session"


class TestLabels:
    @pytest.mark.tonio
    async def test_sets_and_gets_labels(self):
        session = SessionManager.in_memory()

        msg_id = await session.append_message(UserMessage(content="hello", timestamp=1))

        assert session.get_label(msg_id) is None

        label_id = await session.append_label_change(msg_id, "checkpoint")
        assert session.get_label(msg_id) == "checkpoint"

        label_entry = next(e for e in session.get_entries() if e["type"] == "label")
        assert label_entry["id"] == label_id
        assert label_entry["targetId"] == msg_id
        assert label_entry["label"] == "checkpoint"

    @pytest.mark.tonio
    async def test_clears_labels_with_none(self):
        session = SessionManager.in_memory()

        msg_id = await session.append_message(UserMessage(content="hello", timestamp=1))

        await session.append_label_change(msg_id, "checkpoint")
        assert session.get_label(msg_id) == "checkpoint"

        await session.append_label_change(msg_id, None)
        assert session.get_label(msg_id) is None

    @pytest.mark.tonio
    async def test_last_label_wins(self):
        session = SessionManager.in_memory()

        msg_id = await session.append_message(UserMessage(content="hello", timestamp=1))

        await session.append_label_change(msg_id, "first")
        await session.append_label_change(msg_id, "second")
        last_label_id = await session.append_label_change(msg_id, "third")

        assert session.get_label(msg_id) == "third"

        last_label_entry = next(e for e in session.get_entries() if e["id"] == last_label_id)
        tree = session.get_tree()
        msg_node = next(n for n in tree if n.entry["id"] == msg_id)
        assert msg_node.label_timestamp == last_label_entry["timestamp"]

    @pytest.mark.tonio
    async def test_labels_are_included_in_tree_nodes(self):
        session = SessionManager.in_memory()

        msg1_id = await session.append_message(UserMessage(content="hello", timestamp=1))
        msg2_id = await session.append_message(assistant_msg("hi", timestamp=2))

        msg1_label_id = await session.append_label_change(msg1_id, "start")
        msg2_label_id = await session.append_label_change(msg2_id, "response")

        entries = session.get_entries()
        msg1_label_entry = next(e for e in entries if e["id"] == msg1_label_id)
        msg2_label_entry = next(e for e in entries if e["id"] == msg2_label_id)
        tree = session.get_tree()

        msg1_node = next(n for n in tree if n.entry["id"] == msg1_id)
        assert msg1_node.label == "start"
        assert msg1_node.label_timestamp == msg1_label_entry["timestamp"]

        msg2_node = next(n for n in msg1_node.children if n.entry["id"] == msg2_id)
        assert msg2_node.label == "response"
        assert msg2_node.label_timestamp == msg2_label_entry["timestamp"]

    @pytest.mark.tonio
    async def test_labels_preserved_in_create_branched_session(self):
        session = SessionManager.in_memory()

        msg1_id = await session.append_message(UserMessage(content="hello", timestamp=1))
        msg2_id = await session.append_message(assistant_msg("hi", timestamp=2))

        msg1_label_id = await session.append_label_change(msg1_id, "important")
        msg2_label_id = await session.append_label_change(msg2_id, "also-important")
        original_entries = session.get_entries()
        msg1_label_entry = next(e for e in original_entries if e["id"] == msg1_label_id)
        msg2_label_entry = next(e for e in original_entries if e["id"] == msg2_label_id)

        await session.create_branched_session(msg2_id)

        assert session.get_label(msg1_id) == "important"
        assert session.get_label(msg2_id) == "also-important"

        label_entries = [e for e in session.get_entries() if e["type"] == "label"]
        assert len(label_entries) == 2

        tree = session.get_tree()
        msg1_node = next(n for n in tree if n.entry["id"] == msg1_id)
        msg2_node = next(n for n in msg1_node.children if n.entry["id"] == msg2_id)
        assert msg1_node.label_timestamp == msg1_label_entry["timestamp"]
        assert msg2_node.label_timestamp == msg2_label_entry["timestamp"]

    @pytest.mark.tonio
    async def test_rewires_children_of_removed_labels_when_forking(self):
        session = SessionManager.in_memory()

        msg1_id = await session.append_message(UserMessage(content="hello", timestamp=1))
        await session.append_label_change(msg1_id, "checkpoint")
        model_change_id = await session.append_model_change("anthropic", "claude-test")
        msg2_id = await session.append_message(UserMessage(content="followup", timestamp=2))

        await session.create_branched_session(msg2_id)

        assert session.get_entry(model_change_id)["parentId"] == msg1_id

    @pytest.mark.tonio
    async def test_labels_not_on_path_are_not_preserved(self):
        session = SessionManager.in_memory()

        msg1_id = await session.append_message(UserMessage(content="hello", timestamp=1))
        msg2_id = await session.append_message(assistant_msg("hi", timestamp=2))
        msg3_id = await session.append_message(UserMessage(content="followup", timestamp=3))

        await session.append_label_change(msg1_id, "first")
        await session.append_label_change(msg2_id, "second")
        await session.append_label_change(msg3_id, "third")

        await session.create_branched_session(msg2_id)

        assert session.get_label(msg1_id) == "first"
        assert session.get_label(msg2_id) == "second"
        assert session.get_label(msg3_id) is None

    @pytest.mark.tonio
    async def test_labels_not_included_in_build_session_context(self):
        session = SessionManager.in_memory()

        msg_id = await session.append_message(UserMessage(content="hello", timestamp=1))
        await session.append_label_change(msg_id, "checkpoint")

        ctx = session.build_session_context()
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == "user"

    @pytest.mark.tonio
    async def test_throws_when_labeling_non_existent_entry(self):
        session = SessionManager.in_memory()

        with pytest.raises(Exception, match="Entry non-existent not found"):
            await session.append_label_change("non-existent", "label")
