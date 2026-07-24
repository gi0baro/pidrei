"""Mirror of pi agent/test/harness/session.test.ts (both storage backends)."""

import json
import os

import pytest

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.session.jsonl_storage import JsonlSessionStorage
from pidrei_agent.harness.session.memory_storage import InMemorySessionStorage
from pidrei_agent.harness.session.session import Session, SessionContextBuildOptions
from pidrei_agent.harness.types import SessionModelRef
from pidrei_ai.types import Usage, UsageCost
from tests.session_helpers import create_assistant_message, create_temp_dir, create_user_message


def _get_text_data(data) -> str:
    if not isinstance(data, dict) or "text" not in data:
        return ""
    value = data["text"]
    return value if isinstance(value, str) else ""


@pytest.fixture(params=["memory", "jsonl"])
def make_storage(request):
    state: dict = {"backend": request.param}

    async def factory():
        if request.param == "memory":
            return InMemorySessionStorage()
        directory = create_temp_dir()
        state["dir"] = directory
        state["file_path"] = os.path.join(directory, "session.jsonl")
        env = LocalExecutionEnv(cwd=directory)
        return await JsonlSessionStorage.create(env, state["file_path"], cwd=directory, session_id="session-1")

    factory.state = state
    return factory


@pytest.mark.tonio
async def test_appends_messages_and_builds_context_in_order(make_storage):
    session = Session(await make_storage())
    await session.append_message(create_user_message("one"))
    await session.append_message(create_assistant_message("two"))
    context = await session.build_context()
    assert [getattr(m, "role", None) for m in context.messages] == ["user", "assistant"]


@pytest.mark.tonio
async def test_tracks_model_and_thinking_level_changes(make_storage):
    session = Session(await make_storage())
    await session.append_message(create_user_message("one"))
    await session.append_model_change("openai", "gpt-4.1")
    await session.append_thinking_level_change("high")
    context = await session.build_context()
    assert context.thinking_level == "high"
    assert context.model == SessionModelRef(provider="openai", model_id="gpt-4.1")


@pytest.mark.tonio
async def test_supports_branching_by_moving_the_leaf_and_appending_a_new_branch(make_storage):
    session = Session(await make_storage())
    user1 = await session.append_message(create_user_message("one"))
    assistant1 = await session.append_message(create_assistant_message("two"))
    await session.append_message(create_user_message("three"))
    await session.move_to(user1)
    await session.append_message(create_assistant_message("branched"))
    branch = await session.get_branch()
    assert user1 in [entry.id for entry in branch]
    assert assistant1 not in [entry.id for entry in branch]
    context = await session.build_context()
    assert [getattr(m, "role", None) for m in context.messages] == ["user", "assistant"]


@pytest.mark.tonio
async def test_supports_moving_the_leaf_to_root(make_storage):
    session = Session(await make_storage())
    await session.append_message(create_user_message("one"))
    await session.move_to(None)
    assert await session.get_leaf_id() is None
    assert (await session.build_context()).messages == []


@pytest.mark.tonio
async def test_reconstructs_compaction_summaries_in_context(make_storage):
    session = Session(await make_storage())
    await session.append_message(create_user_message("one"))
    await session.append_message(create_assistant_message("two"))
    user2 = await session.append_message(create_user_message("three"))
    await session.append_message(create_assistant_message("four"))
    await session.append_compaction(
        "summary",
        user2,
        1234,
        None,
        None,
        None,
        [create_user_message("three"), create_assistant_message("four")],
    )
    await session.append_message(create_user_message("five"))
    context = await session.build_context()
    assert getattr(context.messages[0], "role", None) == "compactionSummary"
    assert len(context.messages) == 4
    assert [getattr(m, "role", None) for m in context.messages] == [
        "compactionSummary",
        "user",
        "assistant",
        "user",
    ]


@pytest.mark.tonio
async def test_supports_moving_with_branch_summary_entries_in_context(make_storage):
    session = Session(await make_storage())
    user1 = await session.append_message(create_user_message("one"))
    summary_id = await session.move_to(user1, {"summary": "summary text"})
    assert summary_id
    summary_entry = await session.get_entry(summary_id)
    assert summary_entry is not None
    assert (summary_entry.type, summary_entry.parent_id, summary_entry.from_id) == ("branch_summary", user1, user1)
    context = await session.build_context()
    assert getattr(context.messages[1], "role", None) == "branchSummary"


@pytest.mark.tonio
async def test_persists_compaction_usage(make_storage):
    session = Session(await make_storage())
    first_kept_entry_id = await session.append_message(create_user_message("one"))
    usage = Usage(
        input=1,
        output=2,
        cache_read=3,
        cache_write=4,
        total_tokens=10,
        cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )

    compaction_id = await session.append_compaction("summary", first_kept_entry_id, 1234, None, False, usage)

    compaction_entry = await session.get_entry(compaction_id)
    assert compaction_entry is not None and compaction_entry.type == "compaction"
    assert compaction_entry.usage == usage


@pytest.mark.tonio
async def test_persists_branch_summary_usage(make_storage):
    session = Session(await make_storage())
    user1 = await session.append_message(create_user_message("one"))
    usage = Usage(
        input=1,
        output=2,
        cache_read=3,
        cache_write=4,
        total_tokens=10,
        cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )

    summary_id = await session.move_to(user1, {"summary": "summary text", "usage": usage})

    summary_entry = await session.get_entry(summary_id)
    assert summary_entry is not None and summary_entry.type == "branch_summary"
    assert summary_entry.usage == usage


@pytest.mark.tonio
async def test_supports_custom_message_entries_in_context(make_storage):
    session = Session(await make_storage())
    await session.append_message(create_user_message("one"))
    await session.append_custom_message_entry("custom", "hello", True, {"ok": True})
    context = await session.build_context()
    assert getattr(context.messages[1], "role", None) == "custom"


@pytest.mark.tonio
async def test_keeps_custom_entries_in_context_entries_but_omits_them_from_messages_by_default(make_storage):
    session = Session(await make_storage())
    await session.append_message(create_user_message("one"))
    await session.append_custom_entry("chat_message", {"text": "hello"})
    context_entries = await session.build_context_entries()
    context = await session.build_context()
    assert [entry.type for entry in context_entries] == ["message", "custom"]
    assert len(context.messages) == 1


@pytest.mark.tonio
async def test_projects_custom_entries_with_configured_custom_entry_projectors(make_storage):
    session = Session(
        await make_storage(),
        SessionContextBuildOptions(
            entry_projectors={
                "chat_message": lambda entry, _index, _entries: [
                    create_user_message(f"chat: {_get_text_data(entry.data)}")
                ]
            }
        ),
    )
    await session.append_message(create_user_message("one"))
    await session.append_custom_entry("chat_message", {"text": "hello"})
    context = await session.build_context()
    assert [getattr(m, "role", None) for m in context.messages] == ["user", "user"]
    assert context.messages[1].content[0].text == "chat: hello"


@pytest.mark.tonio
async def test_applies_context_entry_transforms_after_default_compaction_selection(make_storage):
    observed_first_entry_type = None

    def drop_compaction(entries):
        nonlocal observed_first_entry_type
        observed_first_entry_type = entries[0].type if entries else None
        return [entry for entry in entries if entry.type != "compaction"]

    session = Session(await make_storage(), SessionContextBuildOptions(entry_transforms=[drop_compaction]))
    await session.append_message(create_user_message("one"))
    kept = await session.append_message(create_user_message("two"))
    await session.append_compaction("summary", kept, 1234)
    await session.append_message(create_user_message("three"))
    context = await session.build_context()
    assert observed_first_entry_type == "compaction"
    assert [getattr(m, "role", None) for m in context.messages] == ["user", "user"]


@pytest.mark.tonio
async def test_normalizes_session_names(make_storage):
    session = Session(await make_storage())
    await session.append_session_name(" hello\nworld\r\nagain ")
    assert await session.get_session_name() == "hello world again"


@pytest.mark.tonio
async def test_supports_labels_and_session_info_entries_without_affecting_context(make_storage):
    session = Session(await make_storage())
    user1 = await session.append_message(create_user_message("one"))
    await session.append_label(user1, "checkpoint")
    await session.append_session_name("name")
    entries = await session.get_entries()
    assert any(entry.type == "label" for entry in entries)
    assert any(entry.type == "session_info" for entry in entries)
    assert await session.get_label(user1) == "checkpoint"
    assert await session.get_session_name() == "name"
    assert len((await session.build_context()).messages) == 1


@pytest.mark.tonio
async def test_rejects_labels_for_missing_entries(make_storage):
    session = Session(await make_storage())
    with pytest.raises(Exception, match="Entry missing not found"):
        await session.append_label("missing", "checkpoint")


@pytest.mark.tonio
async def test_persists_leaf_changes_and_appended_entries_via_storage(make_storage):
    storage = await make_storage()
    session = Session(storage)
    user1 = await session.append_message(create_user_message("one"))
    await session.append_message(create_assistant_message("two"))
    await session.append_label(user1, "checkpoint")
    await session.append_session_name("name")
    await session.move_to(user1)
    await session.append_message(create_assistant_message("branched"))
    session2 = Session(storage)
    context = await session2.build_context()
    assert [getattr(m, "role", None) for m in context.messages] == ["user", "assistant"]
    assert await session2.get_label(user1) == "checkpoint"
    assert await session2.get_session_name() == "name"

    # pi's JSONL variant inspects the persisted file after this test.
    if make_storage.state["backend"] == "jsonl":
        with open(make_storage.state["file_path"], encoding="utf-8") as file:
            lines = file.read().strip().split("\n")
        assert len(lines) > 1
        header = json.loads(lines[0])
        assert header["type"] == "session"
        assert header["version"] == 3
        entries = [json.loads(line) for line in lines[1:]]
        assert any(entry["type"] == "leaf" for entry in entries)
        for entry in entries:
            assert entry["type"] != "entry"
            assert isinstance(entry["id"], str)
