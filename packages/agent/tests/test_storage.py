"""Mirror of pi agent/test/harness/storage.test.ts."""

import json
import os

import pytest

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.session.jsonl_storage import JsonlSessionStorage, load_jsonl_session_metadata
from pidrei_agent.harness.session.memory_storage import InMemorySessionStorage
from pidrei_agent.harness.types import (
    BranchSummaryEntry,
    CompactionEntry,
    JsonlSessionMetadata,
    LabelEntry,
    MessageEntry,
    SessionMetadata,
    SessionStats,
    ok,
)
from pidrei_ai.types import AssistantMessage, TextContent, Usage, UsageCost
from tests.session_helpers import create_assistant_message, create_temp_dir, create_user_message


def make_usage(input, output, cache_read, cache_write, total_tokens, cost_scale) -> Usage:
    return Usage(
        input=input,
        output=output,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=total_tokens,
        cost=UsageCost(
            input=input * cost_scale,
            output=output * cost_scale,
            cache_read=cache_read * cost_scale,
            cache_write=cache_write * cost_scale,
            total=(input + output + cache_read + cache_write) * cost_scale / 100,
        ),
    )


def message_entry(entry_id: str, message, parent_id=None, timestamp="2026-01-01T00:00:00.000Z") -> MessageEntry:
    return MessageEntry(id=entry_id, parent_id=parent_id, timestamp=timestamp, message=message)


def stats_assistant_message() -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text="reply")],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(
            input=10,
            output=20,
            cache_read=30,
            cache_write=40,
            total_tokens=100,
            cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
        ),
        stop_reason="stop",
        timestamp=0,
    )


def stats_compaction_entry() -> CompactionEntry:
    return CompactionEntry(
        id="compaction",
        parent_id="assistant",
        timestamp="2026-01-01T00:00:01.000Z",
        summary="summary",
        first_kept_entry_id="assistant",
        tokens_before=1234,
        usage=Usage(
            input=1,
            output=2,
            cache_read=3,
            cache_write=4,
            total_tokens=10,
            cost=UsageCost(input=0.01, output=0.02, cache_read=0.03, cache_write=0.04, total=0.1),
        ),
    )


def stats_branch_summary_entry() -> BranchSummaryEntry:
    return BranchSummaryEntry(
        id="branch-summary",
        parent_id="compaction",
        timestamp="2026-01-01T00:00:02.000Z",
        from_id="assistant",
        summary="branch",
        usage=Usage(
            input=5,
            output=6,
            cache_read=7,
            cache_write=8,
            total_tokens=26,
            cost=UsageCost(input=0.05, output=0.06, cache_read=0.07, cache_write=0.08, total=0.26),
        ),
    )


EXPECTED_STATS = SessionStats(message_count=1, cached_tokens=40, uncached_tokens=68, total_tokens=136, cost_total=1.36)


# --- InMemorySessionStorage ----------------------------------------------------


@pytest.mark.tonio
async def test_memory_returns_configured_session_metadata():
    metadata = SessionMetadata(id="session-1", created_at="2026-01-01T00:00:00.000Z")
    storage = InMemorySessionStorage(metadata=metadata)
    assert await storage.get_metadata() == metadata


@pytest.mark.tonio
async def test_memory_copies_initial_entries_and_persists_leaf_changes():
    entry = message_entry("entry-1", create_user_message("one"))
    initial_entries = [entry]
    storage = InMemorySessionStorage(entries=initial_entries)
    initial_entries.append(message_entry("entry-2", create_user_message("two")))
    assert [stored.id for stored in await storage.get_entries()] == ["entry-1"]
    assert await storage.get_leaf_id() == "entry-1"
    await storage.set_leaf_id(None)
    assert await storage.get_leaf_id() is None
    last = (await storage.get_entries())[-1]
    assert (last.type, last.target_id) == ("leaf", None)


@pytest.mark.tonio
async def test_memory_rejects_invalid_leaf_ids():
    storage = InMemorySessionStorage()
    with pytest.raises(Exception, match="Entry missing not found"):
        await storage.set_leaf_id("missing")


@pytest.mark.tonio
async def test_memory_finds_entries_by_type():
    storage = InMemorySessionStorage(entries=[message_entry("entry-1", create_user_message("one"))])
    assert [found.id for found in await storage.find_entries("message")] == ["entry-1"]
    assert await storage.find_entries("session_info") == []


@pytest.mark.tonio
async def test_memory_maintains_label_lookup():
    storage = InMemorySessionStorage(entries=[message_entry("entry-1", create_user_message("one"))])
    assert await storage.get_label("entry-1") is None
    await storage.append_entry(
        LabelEntry(
            id="label-1",
            parent_id="entry-1",
            timestamp="2026-01-01T00:00:01.000Z",
            target_id="entry-1",
            label="checkpoint",
        )
    )
    assert await storage.get_label("entry-1") == "checkpoint"
    await storage.append_entry(
        LabelEntry(
            id="label-2", parent_id="label-1", timestamp="2026-01-01T00:00:02.000Z", target_id="entry-1", label=None
        )
    )
    assert await storage.get_label("entry-1") is None


@pytest.mark.tonio
async def test_memory_includes_summary_entry_usage_in_session_stats():
    storage = InMemorySessionStorage(
        entries=[
            message_entry("assistant", stats_assistant_message()),
            stats_compaction_entry(),
            stats_branch_summary_entry(),
        ]
    )
    stats = await storage.get_session_stats()
    assert stats.message_count == EXPECTED_STATS.message_count
    assert stats.cached_tokens == EXPECTED_STATS.cached_tokens
    assert stats.uncached_tokens == EXPECTED_STATS.uncached_tokens
    assert stats.total_tokens == EXPECTED_STATS.total_tokens
    assert stats.cost_total == pytest.approx(EXPECTED_STATS.cost_total)


@pytest.mark.tonio
async def test_memory_walks_paths_to_root_or_retained_tail_compaction():
    root = message_entry("root", create_user_message("root"))
    child = message_entry("child", create_assistant_message("child"), parent_id="root")
    compaction = CompactionEntry(
        id="compaction",
        parent_id="child",
        timestamp="2026-01-01T00:00:01.000Z",
        summary="summary",
        first_kept_entry_id="child",
        tokens_before=1234,
        retained_tail=[create_assistant_message("child")],
    )
    after_compaction = message_entry("after-compaction", create_user_message("after"), parent_id="compaction")
    storage = InMemorySessionStorage(entries=[root, child, compaction, after_compaction])
    assert [entry.id for entry in await storage.get_path_to_root_or_compaction("child")] == ["root", "child"]
    assert [entry.id for entry in await storage.get_path_to_root_or_compaction("after-compaction")] == [
        "compaction",
        "after-compaction",
    ]
    assert await storage.get_path_to_root_or_compaction(None) == []


# --- JsonlSessionStorage --------------------------------------------------------


@pytest.mark.tonio
async def test_jsonl_throws_for_missing_files_when_opening():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    with pytest.raises(Exception) as excinfo:
        await JsonlSessionStorage.open(env, file_path)
    assert excinfo.value.code == "not_found"


@pytest.mark.tonio
async def test_jsonl_writes_the_header_on_create():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    storage = await JsonlSessionStorage.create(env, file_path, cwd=directory, session_id="session-1")
    assert os.path.exists(file_path)
    with open(file_path, encoding="utf-8") as file:
        assert len(file.read().strip().split("\n")) == 1
    assert await storage.get_leaf_id() is None
    assert await storage.get_entries() == []
    await storage.append_entry(message_entry("user-1", create_user_message("one")))
    with open(file_path, encoding="utf-8") as file:
        lines = file.read().strip().split("\n")
    assert json.loads(lines[0])["type"] == "session"
    assert json.loads(lines[1])["id"] == "user-1"
    assert len(lines) == 2


@pytest.mark.tonio
async def test_jsonl_throws_for_malformed_session_headers():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write("not json\n")
    with pytest.raises(Exception, match="first line is not a valid session header"):
        await JsonlSessionStorage.open(env, file_path)


@pytest.mark.tonio
async def test_jsonl_throws_for_malformed_entry_lines():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    header = {
        "type": "session",
        "version": 3,
        "id": "session-1",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "cwd": directory,
    }
    entry = {
        "type": "message",
        "id": "entry-1",
        "parentId": None,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "message": {"role": "user", "content": [{"type": "text", "text": "one"}], "timestamp": 0},
    }
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"{json.dumps(header)}\nnot json\n{json.dumps(entry)}\n")
    with pytest.raises(Exception) as excinfo:
        await JsonlSessionStorage.open(env, file_path)
    assert excinfo.value.code == "invalid_entry"


@pytest.mark.tonio
async def test_jsonl_creates_and_reads_session_metadata_from_the_header():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    storage = await JsonlSessionStorage.create(
        env, file_path, cwd=directory, session_id="session-1", parent_session_path="/tmp/parent.jsonl"
    )
    metadata = await storage.get_metadata()
    assert metadata.id == "session-1"
    assert metadata.cwd == directory
    assert metadata.path == file_path
    assert metadata.parent_session_path == "/tmp/parent.jsonl"
    await storage.append_entry(message_entry("user-1", create_user_message("one")))
    assert await load_jsonl_session_metadata(env, file_path) == metadata


@pytest.mark.tonio
async def test_jsonl_round_trips_custom_header_metadata():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    storage = await JsonlSessionStorage.create(
        env, file_path, cwd=directory, session_id="session-1", metadata={"profile": "reviewer"}
    )
    assert (await storage.get_metadata()).metadata == {"profile": "reviewer"}
    loaded = await JsonlSessionStorage.open(env, file_path)
    assert (await loaded.get_metadata()).metadata == {"profile": "reviewer"}
    assert (await load_jsonl_session_metadata(env, file_path)).metadata == {"profile": "reviewer"}


@pytest.mark.tonio
async def test_jsonl_omits_header_metadata_when_not_provided():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    await JsonlSessionStorage.create(env, file_path, cwd=directory, session_id="session-1")
    with open(file_path, encoding="utf-8") as file:
        assert "metadata" not in json.loads(file.read().strip())
    assert (await load_jsonl_session_metadata(env, file_path)).metadata is None


@pytest.mark.tonio
async def test_jsonl_throws_for_non_object_header_metadata():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    header = {
        "type": "session",
        "version": 3,
        "id": "session-1",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "cwd": directory,
        "metadata": "profile",
    }
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"{json.dumps(header)}\n")
    with pytest.raises(Exception, match="session header metadata must be an object"):
        await JsonlSessionStorage.open(env, file_path)


@pytest.mark.tonio
async def test_jsonl_loads_existing_entries_and_reconstructs_leaf():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    storage = await JsonlSessionStorage.create(env, file_path, cwd=directory, session_id="session-1")
    root = message_entry("root", create_user_message("root"))
    child = message_entry("child", create_assistant_message("child"), parent_id="root")
    await storage.append_entry(root)
    await storage.append_entry(child)
    loaded = await JsonlSessionStorage.open(env, file_path)
    assert await loaded.get_leaf_id() == "child"
    assert [entry.id for entry in await loaded.get_entries()] == ["root", "child"]
    await loaded.set_leaf_id("root")
    reloaded = await JsonlSessionStorage.open(env, file_path)
    assert await reloaded.get_leaf_id() == "root"
    last = (await reloaded.get_entries())[-1]
    assert (last.type, last.target_id) == ("leaf", "root")
    assert [entry.id for entry in await loaded.get_path_to_root_or_compaction("child")] == ["root", "child"]


@pytest.mark.tonio
async def test_jsonl_finds_entries_by_type():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    storage = await JsonlSessionStorage.create(env, file_path, cwd=directory, session_id="session-1")
    await storage.append_entry(message_entry("entry-1", create_user_message("one")))
    assert [found.id for found in await storage.find_entries("message")] == ["entry-1"]
    assert await storage.find_entries("session_info") == []


@pytest.mark.tonio
async def test_jsonl_maintains_label_lookup():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    storage = await JsonlSessionStorage.create(env, file_path, cwd=directory, session_id="session-1")
    await storage.append_entry(message_entry("entry-1", create_user_message("one")))
    assert await storage.get_label("entry-1") is None
    await storage.append_entry(
        LabelEntry(
            id="label-1",
            parent_id="entry-1",
            timestamp="2026-01-01T00:00:01.000Z",
            target_id="entry-1",
            label="checkpoint",
        )
    )
    assert await storage.get_label("entry-1") == "checkpoint"
    await storage.append_entry(
        LabelEntry(
            id="label-2", parent_id="label-1", timestamp="2026-01-01T00:00:02.000Z", target_id="entry-1", label=None
        )
    )
    assert await storage.get_label("entry-1") is None
    loaded = await JsonlSessionStorage.open(env, file_path)
    assert await loaded.get_label("entry-1") is None


@pytest.mark.tonio
async def test_jsonl_includes_summary_entry_usage_in_session_stats():
    directory = create_temp_dir()
    env = LocalExecutionEnv(cwd=directory)
    file_path = os.path.join(directory, "session.jsonl")
    storage = await JsonlSessionStorage.create(env, file_path, cwd=directory, session_id="session-1")
    await storage.append_entry(message_entry("assistant", stats_assistant_message()))
    await storage.append_entry(stats_compaction_entry())
    await storage.append_entry(stats_branch_summary_entry())
    stats = await storage.get_session_stats()
    assert stats.message_count == EXPECTED_STATS.message_count
    assert stats.cached_tokens == EXPECTED_STATS.cached_tokens
    assert stats.uncached_tokens == EXPECTED_STATS.uncached_tokens
    assert stats.total_tokens == EXPECTED_STATS.total_tokens
    assert stats.cost_total == pytest.approx(EXPECTED_STATS.cost_total)


@pytest.mark.tonio
async def test_jsonl_reads_session_metadata_through_the_line_reading_filesystem_operation():
    directory = create_temp_dir()
    file_path = os.path.join(directory, "session.jsonl")
    header = {
        "type": "session",
        "version": 3,
        "id": "session-1",
        "timestamp": "2026-01-01T00:00:00.000Z",
        "cwd": directory,
    }

    class LinesOnlyFs:
        async def read_text_lines(self, _path, max_lines=None, cancel=None):
            return ok([json.dumps(header)])

        async def read_text_file(self, _path, cancel=None):
            raise Exception("read_text_file should not be called for metadata")

        async def write_file(self, _path, _content, cancel=None):
            return ok(None)

        async def append_file(self, _path, _content, cancel=None):
            return ok(None)

    metadata = await load_jsonl_session_metadata(LinesOnlyFs(), file_path)
    assert metadata == JsonlSessionMetadata(
        id="session-1",
        created_at="2026-01-01T00:00:00.000Z",
        cwd=directory,
        path=file_path,
        parent_session_path=None,
    )
