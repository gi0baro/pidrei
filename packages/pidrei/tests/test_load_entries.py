"""Mirror of pi coding-agent test/session-manager/load-entries.test.ts.

pi's `FileEntry[]` are plain JSON objects whether they came from a file or
from `getEntries()`; pidrei's `get_entries()` hands back decoded entries
(message dataclasses), and `in_memory(..., entries)` accepts either shape.
"""

import re

import pytest

from pidrei.core.session_manager import SessionManager
from pidrei_ai.types import TextContent, UserMessage


UUID_V7_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


def user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


async def stored_entries(build) -> list[dict]:
    source = SessionManager.in_memory("/project")
    await build(source)
    return source.get_entries()


@pytest.mark.tonio
async def test_adopts_entries_verbatim():
    async def build(source: SessionManager) -> None:
        await source.append_message(user_message("hello"))
        await source.append_model_change("anthropic", "claude-opus-4-5")
        await source.append_message(user_message("again"))

    entries = await stored_entries(build)

    session = SessionManager.in_memory("/project", None, entries)

    assert session.get_entries() == entries


@pytest.mark.tonio
async def test_keeps_the_loaded_leaf_so_appends_continue_the_conversation():
    async def build(source: SessionManager) -> None:
        await source.append_message(user_message("hello"))
        await source.append_message(user_message("again"))

    entries = await stored_entries(build)
    last_id = entries[-1]["id"]

    session = SessionManager.in_memory("/project", None, entries)
    appended_id = await session.append_message(user_message("continued"))

    assert session.get_leaf_id() == appended_id
    assert session.get_entry(appended_id)["parentId"] == last_id


@pytest.mark.tonio
async def test_never_mints_an_id_that_collides_with_a_loaded_entry():
    async def build(source: SessionManager) -> None:
        for index in range(50):
            await source.append_message(user_message(f"message {index}"))

    entries = await stored_entries(build)

    session = SessionManager.in_memory("/project", None, entries)
    appended_id = await session.append_message(user_message("continued"))

    assert not any(entry["id"] == appended_id for entry in entries)


@pytest.mark.tonio
async def test_rebuilds_the_branch_structure_rather_than_a_flat_chain():
    async def build(source: SessionManager) -> None:
        first_id = await source.append_message(user_message("hello"))
        await source.append_message(user_message("abandoned"))
        source.branch(first_id)
        await source.append_message(user_message("kept"))

    entries = await stored_entries(build)

    session = SessionManager.in_memory("/project", None, entries)
    roots = session.get_tree()

    assert len(roots) == 1
    assert len(roots[0].children) == 2


@pytest.mark.tonio
async def test_rebuilds_labels():
    labelled = {"id": ""}

    async def build(source: SessionManager) -> None:
        labelled["id"] = await source.append_message(user_message("hello"))
        await source.append_label_change(labelled["id"], "checkpoint")

    entries = await stored_entries(build)

    session = SessionManager.in_memory("/project", None, entries)

    assert session.get_label(labelled["id"]) == "checkpoint"


@pytest.mark.tonio
async def test_resolves_a_compaction_against_the_entry_it_was_written_against():
    kept = {"id": ""}

    async def build(source: SessionManager) -> None:
        await source.append_message(user_message("dropped"))
        kept["id"] = await source.append_message(user_message("kept"))
        await source.append_compaction("summary so far", kept["id"], 1000)

    entries = await stored_entries(build)

    session = SessionManager.in_memory("/project", None, entries)
    context = session.build_context_entries()

    assert any(entry["id"] == kept["id"] for entry in context)


@pytest.mark.tonio
async def test_creates_a_header_from_the_options_when_the_entries_carry_none():
    async def build(source: SessionManager) -> None:
        await source.append_message(user_message("hello"))

    entries = await stored_entries(build)

    session = SessionManager.in_memory("/project", {"id": "restored-session"}, entries)

    assert session.get_session_id() == "restored-session"
    assert session.get_header()["id"] == "restored-session"
    assert session.get_header()["cwd"] == "/project"


@pytest.mark.tonio
async def test_generates_a_session_id_when_the_options_carry_none():
    async def build(source: SessionManager) -> None:
        await source.append_message(user_message("hello"))

    entries = await stored_entries(build)

    session = SessionManager.in_memory("/project", None, entries)

    assert UUID_V7_RE.match(session.get_session_id())
    assert session.get_header()["id"] == session.get_session_id()


@pytest.mark.tonio
async def test_stays_off_the_filesystem():
    async def build(source: SessionManager) -> None:
        await source.append_message(user_message("hello"))

    entries = await stored_entries(build)

    session = SessionManager.in_memory("/project", None, entries)
    await session.append_message(user_message("continued"))

    assert session.get_session_file() is None
    assert session.is_persisted() is False


def test_starts_an_empty_session_when_the_entries_are_empty():
    session = SessionManager.in_memory("/project", {"id": "empty-session"}, [])

    assert session.get_session_id() == "empty-session"
    assert session.get_entries() == []
    assert session.get_leaf_id() is None


@pytest.mark.tonio
async def test_takes_the_session_identity_from_a_header_among_the_entries():
    async def build(source: SessionManager) -> None:
        await source.append_message(user_message("hello"))

    body = await stored_entries(build)
    entries = [
        {
            "type": "session",
            "version": 3,
            "id": "stored-session",
            "timestamp": "2026-01-01T00:00:00Z",
            "cwd": "/stored",
        },
        *body,
    ]

    session = SessionManager.in_memory("/project", {"id": "ignored"}, entries)

    assert session.get_session_id() == "stored-session"
    assert session.get_header()["cwd"] == "/stored"


def test_migrates_entries_restored_with_an_older_header():
    entries = [
        {"type": "session", "version": 2, "id": "v2-session", "timestamp": "2026-01-01T00:00:00Z", "cwd": "/project"},
        {
            "type": "message",
            "id": "abc12345",
            "parentId": None,
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"role": "hookMessage", "content": "from a hook", "timestamp": 1},
        },
    ]

    session = SessionManager.in_memory("/project", None, entries)
    restored = session.get_entries()[0]

    assert session.get_header()["version"] == 3
    assert restored["message"].role == "custom"
    assert restored["id"] == "abc12345"


def test_adopts_headerless_entries_as_current_version_without_migrating_them():
    entries = [
        {
            "type": "message",
            "id": "abc12345",
            "parentId": None,
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"role": "hookMessage", "content": "from a hook", "timestamp": 1},
        }
    ]

    session = SessionManager.in_memory("/project", None, entries)
    restored = session.get_entries()[0]

    # An unknown role has no dataclass, so the message stays the wire dict it came in as.
    assert restored["message"]["role"] == "hookMessage"
