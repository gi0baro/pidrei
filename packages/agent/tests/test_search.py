"""Mirror of pi agent/test/harness/session/search.test.ts.

pi's `WorkspaceMetadata` extends SessionMetadata with a `cwd` the assertions
never read; plain `SessionMetadata` stands in for the memory cases. The abort
case uses a pre-cancelled `CancelToken` (pi: aborted `AbortController`).
"""

import os

import pytest

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.session.jsonl import (
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionRepo,
    JsonlSessionRepoOptions,
)
from pidrei_agent.harness.session.jsonl.repo import list_jsonl_session_metadata, load_jsonl_session_storage
from pidrei_agent.harness.session.memory import InMemorySessionStorage
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.session.types import SessionMetadata
from pidrei_agent.search import SessionSearchOptions, create_scanning_session_search
from pidrei_ai.types import TextContent, UserMessage
from pidrei_ai.utils.cancel import AbortError, CancelToken
from tests.session_helpers import create_temp_dir


def message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


def create_memory_session(id: str, created_at: int) -> Session:
    return Session(InMemorySessionStorage(SessionMetadata(id=id, created_at=created_at)))


async def collect(iterable) -> list:
    items = []
    async for item in iterable:
        items.append(item)
    return items


@pytest.mark.tonio
async def test_scans_an_arbitrary_in_memory_projected_source():
    root = create_memory_session("root", 1)
    await root.append_message(message("fix auth flow"))
    other = create_memory_session("other", 2)
    await other.append_message(message("auth in another workspace"))
    search = create_scanning_session_search([root, other])

    assert not hasattr(search, "apply")
    hits = await collect(search.search("auth"))
    assert [hit.session_id for hit in hits] == ["root", "other"]
    assert await collect(search.search("missing")) == []


@pytest.mark.tonio
async def test_includes_labels_in_memory_scanning_projections():
    session = create_memory_session("session", 1)
    entry_id = await session.append_message(message("plain body"))
    await session.set_label(entry_id, "important label")
    search = create_scanning_session_search([session])

    hits = await collect(search.search("important"))
    assert [(hit.session_id, hit.entry_id) for hit in hits] == [("session", entry_id)]


@pytest.mark.tonio
async def test_honors_entry_type_filters_and_abort_signals_in_scanning_search():
    session = create_memory_session("session", 1)
    message_entry_id = await session.append_message(message("auth message"))
    await session.append_custom_entry("note", {"text": "auth custom"})
    search = create_scanning_session_search([session])

    hits = await collect(search.search("auth", SessionSearchOptions(entry_types=["message"])))
    assert [(hit.session_id, hit.entry_id) for hit in hits] == [("session", message_entry_id)]

    cancel = CancelToken()
    cancel.cancel()
    with pytest.raises(AbortError):
        await collect(search.search("auth", SessionSearchOptions(cancel=cancel)))


@pytest.mark.tonio
async def test_scans_jsonl_sessions_from_disk_through_the_jsonl_scanning_source():
    root = create_temp_dir()
    options = JsonlSessionRepoOptions(fs=LocalExecutionEnv(cwd=root), sessions_root=root)
    repository = JsonlSessionRepo(options)
    cwd = os.path.join(root, "workspace")
    other_cwd = os.path.join(root, "other")
    session = await repository.create(JsonlSessionCreateOptions(id="jsonl", cwd=cwd))
    entry_id = await session.append_message(message("jsonl backed auth entry"))
    await session.set_label(entry_id, "disk label")
    other = await repository.create(JsonlSessionCreateOptions(id="other", cwd=other_cwd))
    other_entry_id = await other.append_message(message("jsonl backed auth entry in another cwd"))

    async def jsonl_readables(query: JsonlSessionListOptions | None = None):
        for metadata in await list_jsonl_session_metadata(options, query):
            yield await load_jsonl_session_storage(options, metadata)

    search = create_scanning_session_search(jsonl_readables)

    auth_hits = await collect(search.search("auth"))
    assert len(auth_hits) == 2
    assert {(hit.session_id, hit.entry_id) for hit in auth_hits} == {
        ("jsonl", entry_id),
        ("other", other_entry_id),
    }
    disk_hits = await collect(search.search("disk"))
    assert [(hit.session_id, hit.entry_id) for hit in disk_hits] == [("jsonl", entry_id)]
