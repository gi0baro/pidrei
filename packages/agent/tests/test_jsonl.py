"""JSONL v4 repository conformance and persistence (mirror of pi
agent/test/harness/session/jsonl.test.ts)."""

import json
import os
import re
from datetime import UTC, datetime
from typing import Any

import pytest

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.session.jsonl import (
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepo,
    JsonlSessionRepoOptions,
)
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.session.testing.conformance import create_session_backend_conformance
from pidrei_agent.harness.session.types import (
    ForkOptions,
    OperationFinishedRecord,
    OperationStartedRecord,
    RecordQuery,
    RunIntent,
    SessionCreateOptions,
    SessionError,
    SessionMetadata,
)
from pidrei_agent.harness.types import FileError, err
from tests.session_helpers import create_temp_dir


def create_repository(root: str) -> JsonlSessionRepo:
    return JsonlSessionRepo(JsonlSessionRepoOptions(fs=LocalExecutionEnv(cwd=root), sessions_root=root))


class _WithDefaultSessionCwd:
    """SessionRepo adapter injecting a default cwd into create/fork options."""

    def __init__(self, repository: JsonlSessionRepo, cwd: str):
        self._repository = repository
        self._cwd = cwd

    async def create(self, options: SessionCreateOptions | None = None) -> Session:
        options = options if options is not None else SessionCreateOptions()
        return await self._repository.create(
            JsonlSessionCreateOptions(id=options.id, parent_session_id=options.parent_session_id, cwd=self._cwd)
        )

    async def open(self, metadata: SessionMetadata) -> Session:
        return await self._repository.open(metadata)

    async def list(self, options: Any = None) -> list[JsonlSessionMetadata]:
        return await self._repository.list()

    async def delete(self, metadata: SessionMetadata) -> None:
        await self._repository.delete(metadata)

    async def fork(
        self, source: SessionMetadata, options: ForkOptions, create: SessionCreateOptions | None = None
    ) -> Session:
        create = create if create is not None else SessionCreateOptions()
        return await self._repository.fork(
            source,
            options,
            JsonlSessionCreateOptions(id=create.id, parent_session_id=create.parent_session_id, cwd=self._cwd),
        )


def expected_session_path(root: str, cwd: str, created_at: int, id: str) -> str:
    directory = f"--{re.sub(r'[/\\\\:]', '-', re.sub(r'^[/\\\\]', '', cwd))}--"
    moment = datetime.fromtimestamp(created_at / 1000, tz=UTC)
    timestamp = f"{moment.strftime('%Y-%m-%dT%H:%M:%S')}.{created_at % 1000:03d}Z".replace(":", "-").replace(".", "-")
    return os.path.join(root, directory, f"{timestamp}_{id}.jsonl")


def _mtime_ms(path: str) -> float:
    return os.lstat(path).st_mtime * 1000


def write_raw_session(root: str, id: str, mutations: list[dict[str, Any]]) -> JsonlSessionMetadata:
    path = os.path.join(root, f"{id}.jsonl")
    created_at = 1
    header = {"kind": "header", "version": 4, "id": id, "createdAt": created_at, "cwd": root}
    with open(path, "w", encoding="utf-8") as file:
        file.write("".join(json.dumps(line) + "\n" for line in [header, *mutations]))
    return JsonlSessionMetadata(
        id=id, created_at=created_at, cwd=root, path=path, modified_at=_mtime_ms(path), source_format=4
    )


def user_message(text: str, timestamp: int = 1):
    from pidrei_ai.types import TextContent, UserMessage

    return UserMessage(content=[TextContent(text=text)], timestamp=timestamp)


class _JsonlFixture:
    def __init__(self) -> None:
        root = create_temp_dir()
        self.repository = _WithDefaultSessionCwd(create_repository(root), root)

    async def dispose(self) -> None:
        pass


async def _create_fixture() -> _JsonlFixture:
    return _JsonlFixture()


CONFORMANCE = create_session_backend_conformance(_create_fixture)


@pytest.mark.tonio
@pytest.mark.parametrize("case", CONFORMANCE, ids=[f"{case.group}: {case.name}" for case in CONFORMANCE])
async def test_jsonl_session_repo_conformance(case):
    await case.run()


@pytest.mark.tonio
async def test_exposes_the_complete_metadata_contract():
    root = create_temp_dir()
    repository = create_repository(root)
    cwd = os.path.join(root, "workspace", "project")
    session = await repository.create(
        JsonlSessionCreateOptions(
            id="metadata",
            cwd=cwd,
            parent_session_id="parent",
            metadata={"owner": "agent", "nested": {"enabled": True}},
        )
    )
    metadata = await session.get_metadata()

    assert metadata == JsonlSessionMetadata(
        id="metadata",
        created_at=metadata.created_at,
        parent_session_id="parent",
        path=expected_session_path(root, metadata.cwd, metadata.created_at, metadata.id),
        cwd=cwd,
        modified_at=_mtime_ms(metadata.path),
        source_format=4,
        metadata={"owner": "agent", "nested": {"enabled": True}},
    )
    assert await repository.list(JsonlSessionListOptions(cwd=cwd)) == [metadata]
    assert await repository.list(JsonlSessionListOptions(cwd=os.path.join(root, "other", "project"))) == []


@pytest.mark.tonio
async def test_rejects_session_ids_that_cannot_be_used_in_coding_agent_filenames():
    root = create_temp_dir()
    repository = create_repository(root)

    with pytest.raises(SessionError) as excinfo:
        await repository.create(JsonlSessionCreateOptions(id="../escape", cwd=root))
    assert excinfo.value.code == "invalid_payload"


@pytest.mark.tonio
async def test_allows_the_same_explicit_session_id_in_different_working_directories():
    root = create_temp_dir()
    repository = create_repository(root)
    first_cwd = os.path.join(root, "workspaces", "first")
    second_cwd = os.path.join(root, "workspaces", "second")

    first = await repository.create(JsonlSessionCreateOptions(id="shared", cwd=first_cwd))
    second = await repository.create(JsonlSessionCreateOptions(id="shared", cwd=second_cwd))

    assert (await first.get_metadata()).cwd == first_cwd
    assert (await second.get_metadata()).cwd == second_cwd
    assert [metadata.id for metadata in await repository.list()] == ["shared", "shared"]


@pytest.mark.tonio
async def test_sorts_listed_sessions_by_current_filesystem_modification_time():
    root = create_temp_dir()
    repository = create_repository(root)
    newest_cwd = os.path.join(root, "workspaces", "newest")
    oldest_cwd = os.path.join(root, "workspaces", "oldest")
    newest = await repository.create(JsonlSessionCreateOptions(id="newest", cwd=newest_cwd))
    newest_metadata = await newest.get_metadata()
    oldest = await repository.create(JsonlSessionCreateOptions(id="oldest", cwd=oldest_cwd))
    oldest_metadata = await oldest.get_metadata()
    os.utime(newest_metadata.path, (1_700_000_002, 1_700_000_002))
    os.utime(oldest_metadata.path, (1_700_000_001, 1_700_000_001))

    listed = await repository.list()

    assert [metadata.id for metadata in listed] == ["newest", "oldest"]
    assert [metadata.id for metadata in await repository.list(JsonlSessionListOptions(cwd=newest_cwd))] == ["newest"]
    assert [metadata.modified_at for metadata in listed] == [
        _mtime_ms(newest_metadata.path),
        _mtime_ms(oldest_metadata.path),
    ]


@pytest.mark.tonio
async def test_writes_one_line_per_mutation_and_restores_the_shared_sequence():
    root = create_temp_dir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=root))
    metadata = await session.get_metadata()
    entry_id = await session.append_custom_entry("note", {"value": 1})
    await session.create_lane("thread", entry_id)
    await session.append_record(
        OperationStartedRecord(
            id="run", lane="thread", source_leaf_id=None, intent=RunIntent(original_prompt=[], initial_messages=[])
        )
    )
    await session.set_name("Example")
    await session.set_label(entry_id, "checkpoint")
    await session.move_lane("main", None)

    with open(metadata.path, encoding="utf-8") as file:
        lines = [json.loads(line) for line in file.read().rstrip("\n").split("\n")]
    assert [line["kind"] for line in lines] == ["header", "entry", "lane", "record", "fact", "fact", "lane"]
    assert [line["seq"] for line in lines[1:]] == [1, 2, 3, 4, 5, 6]

    reopened = await create_repository(root).open(metadata)
    lanes = await reopened.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [("main", None), ("thread", entry_id)]
    assert await reopened.get_name() == "Example"
    assert await reopened.get_label(entry_id) == "checkpoint"
    assert [record.id for record in await reopened.find_records()] == ["run"]
    assert [
        record.id for record in await reopened.find_records(RecordQuery(type="operation_started", operation_kind="run"))
    ] == ["run"]
    assert [record.id for record in await reopened.find_open_operations("thread", limit=2)] == ["run"]
    assert [item.seq for item in await reopened.get_log()] == [1, 2, 3, 4, 5, 6]
    finished = await reopened.append_record(
        OperationFinishedRecord(id="finish", lane="thread", run_id="run", outcome="completed")
    )
    assert finished.seq == 7
    assert await reopened.find_open_operations("thread", limit=2) == []


@pytest.mark.tonio
async def test_recomputes_fork_message_counts_when_reopening():
    root = create_temp_dir()
    repository = create_repository(root)
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=root))
    await source.append_message(user_message("one", timestamp=1))
    await source.append_message(user_message("two", timestamp=2))
    fork = await repository.fork(
        await source.get_metadata(), ForkOptions(), JsonlSessionCreateOptions(id="fork", cwd=root)
    )
    metadata = await fork.get_metadata()

    reopened = await create_repository(root).open(metadata)
    assert (await reopened.get_stats()).message_count == 2
    await reopened.append_message(user_message("three", timestamp=3))
    assert (await reopened.get_stats()).message_count == 3

    verified = await create_repository(root).open(metadata)
    assert (await verified.get_stats()).message_count == 3


@pytest.mark.tonio
async def test_reopens_a_tree_fork_with_its_lanes_and_facts():
    from pidrei_agent.harness.session.types import CustomEntry, EntryQuery

    root = create_temp_dir()
    repository = create_repository(root)
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=root))
    root_id = await source.append_custom_entry("root")
    await source.create_lane("thread", root_id)
    main_id = await source.append_custom_entry("main")
    thread_entry = await source.append_entry(CustomEntry(id="thread", custom_type="thread"), "thread")
    thread_id = thread_entry.id
    await source.set_name("Source")
    await source.set_label(thread_id, "tip")
    fork = await repository.fork(
        await source.get_metadata(), ForkOptions(scope="tree"), JsonlSessionCreateOptions(id="fork", cwd=root)
    )
    metadata = await fork.get_metadata()

    with open(metadata.path, encoding="utf-8") as file:
        imported_entry_lines = [
            json.loads(line) for line in file.read().rstrip("\n").split("\n") if json.loads(line)["kind"] == "entry"
        ]
    assert [("lane" in line) for line in imported_entry_lines] == [False, False, False]

    reopened = await create_repository(root).open(metadata)
    assert [entry.id for entry in await reopened.find_entries(EntryQuery(order="oldestFirst"))] == [
        root_id,
        main_id,
        thread_id,
    ]
    lanes = await reopened.get_lanes()
    assert [(pointer.lane, pointer.leaf_id) for pointer in lanes] == [("main", main_id), ("thread", thread_id)]
    assert await reopened.get_name() == "Source"
    assert await reopened.get_label(thread_id) == "tip"
    assert await reopened.find_records() == []


class _FailingSecondAppendEnv(LocalExecutionEnv):
    """Once armed, passes the first append_file through and fails the second."""

    def __init__(self, cwd: str):
        super().__init__(cwd=cwd)
        self._append_calls: int | None = None

    def arm(self) -> None:
        self._append_calls = 0

    async def append_file(self, path, content, cancel=None):
        if self._append_calls is not None:
            self._append_calls += 1
            if self._append_calls == 2:
                return err(FileError("unknown", "injected staging failure"))
        return await super().append_file(path, content, cancel)


@pytest.mark.tonio
async def test_does_not_publish_a_partial_fork_when_staging_fails():
    root = create_temp_dir()
    env = _FailingSecondAppendEnv(cwd=root)
    repository = JsonlSessionRepo(JsonlSessionRepoOptions(fs=env, sessions_root=root))
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=root))
    await source.append_message(user_message("one", timestamp=1))
    await source.append_message(user_message("two", timestamp=2))
    source_metadata = await source.get_metadata()
    env.arm()

    with pytest.raises(SessionError) as excinfo:
        await repository.fork(source_metadata, ForkOptions(), JsonlSessionCreateOptions(id="fork", cwd=root))
    assert excinfo.value.code == "storage"

    assert [metadata.id for metadata in await repository.list()] == ["source"]
    directory = os.path.dirname(source_metadata.path)
    assert [name for name in os.listdir(directory) if name.endswith(".tmp")] == []


class _FailingRenameEnv(LocalExecutionEnv):
    async def rename_file(self, source_path, destination_path, cancel=None):
        return err(FileError("unknown", "injected rename failure"))


@pytest.mark.tonio
async def test_does_not_publish_a_fork_when_atomic_rename_fails():
    root = create_temp_dir()
    env = _FailingRenameEnv(cwd=root)
    repository = JsonlSessionRepo(JsonlSessionRepoOptions(fs=env, sessions_root=root))
    source = await repository.create(JsonlSessionCreateOptions(id="source", cwd=root))
    await source.append_message(user_message("one", timestamp=1))
    source_metadata = await source.get_metadata()

    with pytest.raises(SessionError) as excinfo:
        await repository.fork(source_metadata, ForkOptions(), JsonlSessionCreateOptions(id="fork", cwd=root))
    assert excinfo.value.code == "storage"

    assert [metadata.id for metadata in await repository.list()] == ["source"]
    directory = os.path.dirname(source_metadata.path)
    assert [name for name in os.listdir(directory) if name.endswith(".tmp")] == []


@pytest.mark.tonio
async def test_repairs_a_valid_final_line_missing_its_newline():
    from pidrei_agent.harness.session.types import EntryQuery

    root = create_temp_dir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=root))
    metadata = await session.get_metadata()
    first_id = await session.append_custom_entry("first")
    with open(metadata.path, encoding="utf-8") as file:
        unterminated = file.read().rstrip("\n")
    with open(metadata.path, "w", encoding="utf-8") as file:
        file.write(unterminated)

    reopened = await create_repository(root).open(metadata)
    with open(metadata.path, encoding="utf-8") as file:
        assert file.read() == f"{unterminated}\n"
    second_id = await reopened.append_custom_entry("second")

    verified = await create_repository(root).open(metadata)
    assert [entry.id for entry in await verified.find_entries(EntryQuery(order="oldestFirst"))] == [
        first_id,
        second_id,
    ]


class _FailingFirstAppendEnv(LocalExecutionEnv):
    def __init__(self, cwd: str, error: FileError):
        super().__init__(cwd=cwd)
        self._error: FileError | None = error

    async def append_file(self, path, content, cancel=None):
        if self._error is not None:
            error, self._error = self._error, None
            return err(error)
        return await super().append_file(path, content, cancel)


@pytest.mark.tonio
async def test_fails_to_open_when_repairing_a_missing_final_newline_fails():
    root = create_temp_dir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=root))
    metadata = await session.get_metadata()
    await session.append_custom_entry("first")
    with open(metadata.path, encoding="utf-8") as file:
        content = file.read().rstrip("\n")
    with open(metadata.path, "w", encoding="utf-8") as file:
        file.write(content)

    env = _FailingFirstAppendEnv(cwd=root, error=FileError("permission_denied", "repair denied", metadata.path))
    failing_repository = JsonlSessionRepo(JsonlSessionRepoOptions(fs=env, sessions_root=root))

    with pytest.raises(SessionError) as excinfo:
        await failing_repository.open(metadata)
    assert excinfo.value.code == "storage"
    assert excinfo.value.__cause__ is not None and excinfo.value.__cause__.code == "permission_denied"


@pytest.mark.tonio
async def test_truncates_a_malformed_final_line():
    root = create_temp_dir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=root))
    metadata = await session.get_metadata()
    await session.append_custom_entry("note", {"value": "kept"})
    with open(metadata.path, encoding="utf-8") as file:
        valid_prefix = file.read()
    with open(metadata.path, "a", encoding="utf-8") as file:
        file.write('{"kind":"entry"')

    reopened = await create_repository(root).open(metadata)
    assert len(await reopened.find_entries()) == 1
    with open(metadata.path, encoding="utf-8") as file:
        assert file.read() == valid_prefix
    appended_id = await reopened.append_custom_entry("after-recovery")
    appended = await reopened.get_entry(appended_id)
    assert appended is not None and appended.seq == 2


@pytest.mark.tonio
async def test_rejects_a_malformed_middle_line_without_modifying_the_file():
    root = create_temp_dir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=root))
    metadata = await session.get_metadata()
    await session.append_custom_entry("first")
    await session.append_custom_entry("second")
    with open(metadata.path, encoding="utf-8") as file:
        lines = file.read().rstrip("\n").split("\n")
    corrupted = f"{lines[0]}\n{lines[1]}\nnot-json\n{lines[2]}\n"
    with open(metadata.path, "w", encoding="utf-8") as file:
        file.write(corrupted)

    with pytest.raises(SessionError) as excinfo:
        await create_repository(root).open(metadata)
    assert excinfo.value.code == "invalid_entry"
    with open(metadata.path, encoding="utf-8") as file:
        assert file.read() == corrupted


@pytest.mark.tonio
async def test_rejects_an_imported_entry_that_references_a_missing_parent():
    root = create_temp_dir()
    metadata = write_raw_session(
        root,
        "missing-parent",
        [
            {
                "kind": "entry",
                "type": "custom",
                "id": "orphan",
                "customType": "note",
                "parentId": "missing",
                "seq": 1,
                "timestamp": 1,
            }
        ],
    )

    repository = create_repository(root)
    with pytest.raises(SessionError) as excinfo:
        await repository.open(metadata)
    assert excinfo.value.code == "invalid_entry"
    assert "references missing parent missing" in excinfo.value.message


@pytest.mark.tonio
async def test_rejects_a_lane_bound_entry_that_does_not_chain_to_the_lane_leaf():
    root = create_temp_dir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="session", cwd=root))
    metadata = await session.get_metadata()
    await session.append_custom_entry("first")
    await session.append_custom_entry("second")

    with open(metadata.path, encoding="utf-8") as file:
        lines = [json.loads(line) for line in file.read().rstrip("\n").split("\n")]
    lines[2]["parentId"] = None
    with open(metadata.path, "w", encoding="utf-8") as file:
        file.write("".join(json.dumps(line) + "\n" for line in lines))

    with pytest.raises(SessionError) as excinfo:
        await create_repository(root).open(metadata)
    assert excinfo.value.code == "invalid_entry"
    assert "does not chain to the lane leaf" in excinfo.value.message


@pytest.mark.tonio
async def test_does_not_move_a_lane_for_an_imported_entry_without_lane_metadata():
    root = create_temp_dir()
    metadata = write_raw_session(
        root,
        "import",
        [
            {
                "kind": "entry",
                "type": "custom",
                "id": "imported",
                "customType": "note",
                "parentId": None,
                "seq": 1,
                "timestamp": 1,
            }
        ],
    )

    imported = await create_repository(root).open(metadata)
    assert await imported.get_leaf_id() is None
    assert [entry.id for entry in await imported.find_entries()] == ["imported"]

    with open(metadata.path, "a", encoding="utf-8") as file:
        file.write(json.dumps({"kind": "lane", "seq": 2, "lane": "main", "leafId": "imported"}) + "\n")
    moved = await create_repository(root).open(metadata)
    assert await moved.get_leaf_id() == "imported"


_REPLAY_REJECTIONS = [
    (
        "a non-consecutive sequence",
        "non-consecutive seq",
        [
            {
                "kind": "entry",
                "type": "custom",
                "id": "entry",
                "customType": "note",
                "parentId": None,
                "seq": 2,
                "timestamp": 1,
            }
        ],
    ),
    (
        "a duplicate entry/record id",
        "duplicate id",
        [
            {
                "kind": "entry",
                "type": "custom",
                "id": "duplicate",
                "customType": "note",
                "parentId": None,
                "seq": 1,
                "timestamp": 1,
            },
            {
                "kind": "record",
                "type": "operation_started",
                "id": "duplicate",
                "lane": "main",
                "seq": 2,
                "timestamp": 2,
                "sourceLeafId": None,
                "intent": {"kind": "run", "originalPrompt": [], "initialMessages": []},
            },
        ],
    ),
    (
        "an entry with a missing parent",
        "missing parent",
        [
            {
                "kind": "entry",
                "type": "custom",
                "id": "entry",
                "customType": "note",
                "parentId": "missing",
                "seq": 1,
                "timestamp": 1,
            }
        ],
    ),
    (
        "an entry referencing a missing lane",
        "missing lane",
        [
            {
                "kind": "entry",
                "lane": "thread",
                "type": "custom",
                "id": "entry",
                "customType": "note",
                "parentId": None,
                "seq": 1,
                "timestamp": 1,
            }
        ],
    ),
    (
        "a record referencing a missing lane",
        "missing lane",
        [
            {
                "kind": "record",
                "type": "operation_started",
                "id": "run",
                "lane": "thread",
                "seq": 1,
                "timestamp": 1,
                "sourceLeafId": None,
                "intent": {"kind": "run", "originalPrompt": [], "initialMessages": []},
            }
        ],
    ),
    (
        "a lane move referencing a missing entry",
        "missing lane target",
        [{"kind": "lane", "lane": "thread", "leafId": "missing", "seq": 1}],
    ),
    (
        "a label referencing a missing entry",
        "missing label target",
        [{"kind": "fact", "fact": "label", "targetId": "missing", "label": "checkpoint", "seq": 1}],
    ),
]


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("name", "message", "mutations"), _REPLAY_REJECTIONS, ids=[case[0] for case in _REPLAY_REJECTIONS]
)
async def test_rejects_invalid_mutations_during_replay(name, message, mutations):
    root = create_temp_dir()
    metadata = write_raw_session(root, re.sub(r"[^A-Za-z0-9._-]", "-", name), mutations)

    with pytest.raises(SessionError) as excinfo:
        await create_repository(root).open(metadata)
    assert excinfo.value.code == "invalid_entry"
    assert message in excinfo.value.message


@pytest.mark.tonio
async def test_rejects_a_complete_malformed_interior_mutation_without_modifying_the_file():
    root = create_temp_dir()
    metadata = write_raw_session(
        root,
        "malformed-interior",
        [
            {
                "kind": "record",
                "type": "operation_started",
                "id": "run",
                "lane": "main",
                "seq": 1,
                "timestamp": 1,
                "sourceLeafId": None,
            },
            {"kind": "fact", "fact": "name", "name": "after", "seq": 2},
        ],
    )
    with open(metadata.path, encoding="utf-8") as file:
        corrupted = file.read()

    with pytest.raises(SessionError) as excinfo:
        await create_repository(root).open(metadata)
    assert excinfo.value.code == "invalid_entry"
    with open(metadata.path, encoding="utf-8") as file:
        assert file.read() == corrupted


class _TruncatingWriteEnv(LocalExecutionEnv):
    """First write_file call truncates the target, then reports failure."""

    def __init__(self, cwd: str):
        super().__init__(cwd=cwd)
        self._failed = False

    async def write_file(self, path, content, cancel=None):
        if not self._failed:
            self._failed = True
            damaged = await super().write_file(path, "")
            if not damaged.ok:
                return damaged
            return err(FileError("unknown", "repair interrupted after truncation", path))
        return await super().write_file(path, content, cancel)


@pytest.mark.tonio
async def test_preserves_the_session_when_staging_torn_tail_repair_fails():
    root = create_temp_dir()
    repository = create_repository(root)
    session = await repository.create(JsonlSessionCreateOptions(id="repair-failure", cwd=root))
    metadata = await session.get_metadata()
    await session.append_custom_entry("kept")
    with open(metadata.path, "a", encoding="utf-8") as file:
        file.write('{"kind":"entry"')
    with open(metadata.path, encoding="utf-8") as file:
        original = file.read()

    env = _TruncatingWriteEnv(cwd=root)
    failing_repository = JsonlSessionRepo(JsonlSessionRepoOptions(fs=env, sessions_root=root))

    with pytest.raises(SessionError) as excinfo:
        await failing_repository.open(metadata)
    assert excinfo.value.code == "storage"
    with open(metadata.path, encoding="utf-8") as file:
        assert file.read() == original
    assert not os.path.exists(f"{metadata.path}.tmp")
