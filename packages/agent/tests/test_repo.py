"""Mirror of pi agent/test/harness/repo.test.ts."""

import os

import pytest

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.session.jsonl_repo import JsonlSessionRepo
from pidrei_agent.harness.session.memory_repo import InMemorySessionRepo
from tests.session_helpers import create_assistant_message, create_temp_dir, create_user_message


@pytest.mark.tonio
async def test_memory_repo_opens_deletes_and_forks_by_metadata():
    repo = InMemorySessionRepo()
    session = await repo.create(id="session-1")
    metadata = await session.get_metadata()
    user1 = await session.append_message(create_user_message("one"))
    assistant1 = await session.append_message(create_assistant_message("two"))
    user2 = await session.append_message(create_user_message("three"))
    assert await repo.open(metadata) is session
    assert [info.id for info in await repo.list()] == ["session-1"]
    fork = await repo.fork(metadata, entry_id=user2, id="session-2")
    assert [entry.id for entry in await fork.get_entries()] == [user1, assistant1]
    full_fork = await repo.fork(metadata, id="session-3")
    assert [entry.id for entry in await full_fork.get_entries()] == [user1, assistant1, user2]
    await repo.delete(metadata)
    with pytest.raises(Exception, match="Session not found: session-1"):
        await repo.open(metadata)


@pytest.mark.tonio
async def test_jsonl_repo_stores_sessions_below_encoded_cwd_directories_and_lists_by_cwd():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    cwd = "/tmp/my-project"
    other_cwd = "/tmp/other-project"
    repo = JsonlSessionRepo(fs=env, sessions_root=root)
    session = await repo.create(cwd=cwd, id="019de8c2-de29-73e9-ae0c-e134db34c447")
    other_session = await repo.create(cwd=other_cwd, id="other-session")
    metadata = await session.get_metadata()
    other_metadata = await other_session.get_metadata()
    assert "--tmp-my-project--" in metadata.path
    assert "--tmp-other-project--" in other_metadata.path
    assert os.path.exists(metadata.path)
    assert [listed.id for listed in await repo.list(cwd=cwd)] == [metadata.id]
    assert sorted(listed.id for listed in await repo.list()) == sorted([metadata.id, other_metadata.id])


@pytest.mark.tonio
async def test_jsonl_repo_opens_deletes_and_forks_by_metadata():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    repo = JsonlSessionRepo(fs=env, sessions_root=root)
    source = await repo.create(cwd="/tmp/source", id="source-session")
    source_metadata = await source.get_metadata()
    user1 = await source.append_message(create_user_message("one"))
    assistant1 = await source.append_message(create_assistant_message("two"))
    user2 = await source.append_message(create_user_message("three"))
    assert await (await repo.open(source_metadata)).get_metadata() == source_metadata
    fork = await repo.fork(source_metadata, cwd="/tmp/target", id="fork-session", entry_id=user2)
    fork_metadata = await fork.get_metadata()
    assert fork_metadata.cwd == "/tmp/target"
    assert fork_metadata.parent_session_path == source_metadata.path
    assert [entry.id for entry in await fork.get_entries()] == [user1, assistant1]
    full_fork = await repo.fork(source_metadata, cwd="/tmp/target", id="full-fork-session")
    assert [entry.id for entry in await full_fork.get_entries()] == [user1, assistant1, user2]
    await repo.delete(source_metadata)
    assert not os.path.exists(source_metadata.path)
    with pytest.raises(Exception, match="Session not found"):
        await repo.open(source_metadata)


@pytest.mark.tonio
async def test_jsonl_repo_persists_header_metadata_through_create_list_and_fork():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    repo = JsonlSessionRepo(fs=env, sessions_root=root)
    source = await repo.create(cwd="/tmp/source", id="source-session", metadata={"profile": "reviewer"})
    source_metadata = await source.get_metadata()
    assert source_metadata.metadata == {"profile": "reviewer"}
    assert [listed.metadata for listed in await repo.list(cwd="/tmp/source")] == [{"profile": "reviewer"}]
    fork = await repo.fork(source_metadata, cwd="/tmp/target", id="fork-session")
    assert (await fork.get_metadata()).metadata == {"profile": "reviewer"}
    overridden = await repo.fork(
        source_metadata, cwd="/tmp/target", id="overridden-session", metadata={"profile": "writer"}
    )
    assert (await overridden.get_metadata()).metadata == {"profile": "writer"}
