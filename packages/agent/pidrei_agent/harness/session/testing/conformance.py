"""Session backend conformance cases (port of pi `session/testing/conformance.ts`).

The acceptance spec every `SessionRepo` backend must pass. Cases are
runner-independent: `create_session_backend_conformance(factory)` returns plain
case objects; test modules register them with pytest. Where pi feeds
JS-impossible payloads (`undefined`, `1n`, `Map`), the port substitutes the
Python values the serializability guard must reject (non-finite floats, sets,
arbitrary objects, bytes, cycles, non-string keys).
"""

import dataclasses
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import AssistantMessage, TextContent, ToolResultMessage, Usage, UsageCost, UserMessage

from ..types import (
    BranchQuery,
    CompactionEntry,
    CompactionIntent,
    CustomEntry,
    Entry,
    EntryCursor,
    EntryQuery,
    ForkOptions,
    LanePointer,
    LogOptions,
    MessageEntry,
    NameFactLogItem,
    NavigationIntent,
    OperationFinishedRecord,
    OperationKind,
    OperationStartedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    RecordQuery,
    RunIntent,
    SessionCreateOptions,
    SessionError,
    SessionErrorCode,
    SessionRepo,
    SessionStats,
    StepAttemptRecord,
    ToolStartedRecord,
    UsageRecord,
)
from .types import SessionBackendConformanceCase, SessionBackendFixtureFactory


def _create_user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


def _create_assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(),
        stop_reason="stop",
        timestamp=1,
    )


def _operation_started(id: str, *, lane: str, kind: OperationKind) -> OperationStartedRecord:
    if kind == "run":
        intent: RunIntent | CompactionIntent | NavigationIntent = RunIntent(original_prompt=[], initial_messages=[])
    elif kind == "compaction":
        intent = CompactionIntent(result_entry_id=f"{id}-result")
    else:
        intent = NavigationIntent(target_id=None, summarize=False)
    return OperationStartedRecord(id=id, lane=lane, source_leaf_id=None, intent=intent)


async def _entry_ids(entries: Awaitable[list[Entry]]) -> list[str]:
    return [entry.id for entry in await entries]


async def _rejects_with_code(operation: Coroutine[Any, Any, Any], code: SessionErrorCode) -> None:
    try:
        await operation
    except SessionError as error:
        assert error.code == code, f"Expected SessionError with code {code}, got {error.code}"
        return
    raise AssertionError(f"Expected SessionError with code {code}")


type _ConformanceTest = Callable[[SessionRepo], Awaitable[None]]


@dataclass(slots=True)
class _Case:
    group: str
    name: str
    _factory: SessionBackendFixtureFactory
    _test: _ConformanceTest

    async def run(self) -> None:
        fixture = await self._factory()
        try:
            await self._test(fixture.repository)
        finally:
            await fixture.dispose()


async def _case_assigns_parents_and_sequence(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    root = await session.append_entry(MessageEntry(id="root", message=_create_user_message("root")), "main")
    await session.create_lane("thread", root.id)
    child = await session.append_entry(CustomEntry(id="child", custom_type="note", data={"value": 1}), "thread")
    record = await session.append_record(_operation_started("run", lane="thread", kind="run"))
    await session.set_name("Example")
    await session.set_label(root.id, "checkpoint")
    await session.move_lane("main", child.id)

    assert (root.parent_id, root.seq) == (None, 1)
    assert (child.parent_id, child.seq) == ("root", 3)
    assert record.seq == 4
    for timestamp in (root.timestamp, child.timestamp, record.timestamp):
        assert isinstance(timestamp, int) and not isinstance(timestamp, bool) and timestamp >= 0, (
            "storage-assigned timestamps must be Unix milliseconds"
        )
    assert [(item.kind, item.seq) for item in await session.get_log()] == [
        ("entry", 1),
        ("lane", 2),
        ("entry", 3),
        ("record", 4),
        ("fact", 5),
        ("fact", 6),
        ("lane", 7),
    ]
    assert await session.get_lanes() == [
        LanePointer(lane="main", leaf_id="child"),
        LanePointer(lane="thread", leaf_id="child"),
    ]


async def _case_commits_records_and_lane_moves(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    root = await session.append_entry(MessageEntry(id="root", message=_create_user_message("root")), "main")
    finished = await session.append_record(
        OperationFinishedRecord(id="finish", lane="main", run_id="run", outcome="completed")
    )

    assert finished.seq == 2
    assert await session.get_lanes() == [LanePointer(lane="main", leaf_id="root")]
    await session.move_lane("main", None)
    assert await session.get_lanes() == [LanePointer(lane="main", leaf_id=None)]
    log = await session.get_log()
    assert [item.kind for item in log] == ["entry", "record", "lane"]
    assert (log[0].seq, log[0].entry) == (1, root)
    assert (log[1].seq, log[1].record) == (2, finished)
    assert (log[2].seq, log[2].lane, log[2].leaf_id) == (3, "main", None)

    await _rejects_with_code(session.move_lane("main", "missing"), "not_found")
    assert len(await session.find_records()) == 1
    assert [item.seq for item in await session.get_log()] == [1, 2, 3]


async def _case_rejects_duplicate_ids(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.append_entry(MessageEntry(id="shared", message=_create_user_message("root")), "main")
    await _rejects_with_code(
        session.append_record(_operation_started("shared", lane="main", kind="run")), "already_exists"
    )
    await session.append_record(_operation_started("run", lane="main", kind="run"))
    await _rejects_with_code(session.append_entry(CustomEntry(id="run", custom_type="note"), "main"), "already_exists")
    assert [item.seq for item in await session.get_log()] == [1, 2]


async def _case_isolates_lanes(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.append_entry(MessageEntry(id="root", message=_create_user_message("root")), "main")
    await session.create_lane("thread", "root")
    await session.append_entry(MessageEntry(id="main-child", message=_create_user_message("main")), "main")
    await session.append_entry(MessageEntry(id="thread-child", message=_create_user_message("thread")), "thread")

    assert await session.get_lanes() == [
        LanePointer(lane="main", leaf_id="main-child"),
        LanePointer(lane="thread", leaf_id="thread-child"),
    ]
    assert await _entry_ids(session.find_entries_on_branch(BranchQuery(start="main-child", order="oldestFirst"))) == [
        "root",
        "main-child",
    ]
    assert await _entry_ids(session.find_entries_on_branch(BranchQuery(start="thread-child", order="oldestFirst"))) == [
        "root",
        "thread-child",
    ]


async def _case_rejects_invalid_queries(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="invalid-queries"))
    await session.create_lane("thread", None)
    thread = session.view("thread")

    await _rejects_with_code(session.find_entries(EntryQuery(limit=0)), "invalid_query")
    await _rejects_with_code(session.find_entry(EntryQuery(limit=0)), "invalid_query")
    await _rejects_with_code(session.find_entries_on_branch(BranchQuery(limit=0)), "invalid_query")
    await _rejects_with_code(
        thread.find_entries_on_branch(BranchQuery(cursor=EntryCursor(after_seq=-1))), "invalid_query"
    )
    await _rejects_with_code(thread.find_entry_on_branch(BranchQuery(limit=0)), "invalid_query")
    await _rejects_with_code(session.find_records(RecordQuery(limit=0)), "invalid_query")
    await _rejects_with_code(session.find_records(RecordQuery(operation_kind="run")), "invalid_query")
    await _rejects_with_code(
        session.find_records(RecordQuery(type="step_attempt", operation_kind="run")), "invalid_query"
    )
    await _rejects_with_code(session.find_open_operations("main", limit=0), "invalid_query")
    await _rejects_with_code(session.find_open_operations("main", limit=-1), "invalid_query")
    await _rejects_with_code(session.get_log(LogOptions(after_seq=-1)), "invalid_query")


async def _case_bounded_filtered_cursor_queries(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.append_entry(MessageEntry(id="root", message=_create_user_message("root")), "main")
    await session.append_entry(CustomEntry(id="old-note", custom_type="note", data=1), "main")
    await session.append_entry(
        CompactionEntry(id="compact", summary="summary", retained_tail=[], tokens_before=10), "main"
    )
    await session.append_entry(CustomEntry(id="new-note", custom_type="note", data=2), "main")
    await session.append_entry(MessageEntry(id="tail", message=_create_assistant_message("tail")), "main")

    assert await _entry_ids(session.find_entries()) == ["tail", "new-note", "compact", "old-note", "root"]
    assert await _entry_ids(
        session.find_entries(EntryQuery(order="oldestFirst", cursor=EntryCursor(after_seq=2), limit=2))
    ) == ["compact", "new-note"]
    assert await _entry_ids(session.find_entries(EntryQuery(custom_type="note"))) == ["new-note", "old-note"]
    assert await _entry_ids(session.find_entries_on_branch(BranchQuery(start="tail", custom_type="note", limit=1))) == [
        "new-note"
    ]
    assert await _entry_ids(
        session.find_entries_on_branch(BranchQuery(start="tail", stop_at_type="compaction", type="message"))
    ) == ["tail"]
    assert (
        await _entry_ids(session.find_entries_on_branch(BranchQuery(start="tail", stop_at_id="tail", type="custom")))
        == []
    )
    assert await _entry_ids(
        session.find_entries_on_branch(BranchQuery(start="tail", stop_at_type="custom", order="oldestFirst"))
    ) == ["root", "old-note"]
    await _rejects_with_code(session.find_entries(EntryQuery(limit=0)), "invalid_query")
    await _rejects_with_code(session.find_entries_on_branch(BranchQuery(start="missing")), "not_found")


async def _case_keeps_lane_names_permanent(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.create_lane("thread", None)
    await session.append_record(_operation_started("old-run", lane="thread", kind="run"))
    await session.append_record(
        QueueEnqueuedRecord(
            id="old-next-run",
            lane="thread",
            queue="nextRun",
            target=MessageEntry(id="queued-message", message=_create_user_message("queued")),
        )
    )

    assert [record.id for record in await session.find_records(RecordQuery(lane="thread"))] == [
        "old-next-run",
        "old-run",
    ]
    assert [item.record.id for item in await session.get_log() if item.kind == "record"] == [
        "old-run",
        "old-next-run",
    ]
    await _rejects_with_code(session.create_lane("thread", None), "already_exists")


async def _case_persists_queue_cancellation(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    enqueued = await session.append_record(
        QueueEnqueuedRecord(
            id="enqueue",
            lane="main",
            queue="nextRun",
            target=MessageEntry(id="queued-message", message=_create_user_message("queued")),
        )
    )
    cancelled = await session.append_record(QueueCancelledRecord(id="cancel", lane="main", entry_id="queued-message"))
    assert (cancelled.seq, cancelled.entry_id) == (2, "queued-message")
    assert cancelled.run_id is None
    assert await session.get_entry("queued-message") is None
    cancellations = await session.find_records(RecordQuery(type="queue_cancelled"))
    assert cancellations and cancellations[0].entry_id == "queued-message"
    assert cancellations == [cancelled]
    log = await session.get_log()
    assert [item.kind for item in log] == ["record", "record"]
    assert (log[0].seq, log[0].record) == (enqueued.seq, enqueued)
    assert (log[1].seq, log[1].record) == (cancelled.seq, cancelled)


async def _case_filters_records(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.append_record(_operation_started("run-1", lane="main", kind="run"))
    await session.append_record(
        StepAttemptRecord(
            id="attempt-1", lane="main", run_id="run-1", step="assistant", attempt=1, result_entry_id="assistant-1"
        )
    )
    await session.create_lane("thread", None)
    await session.append_record(_operation_started("run-2", lane="thread", kind="run"))
    await session.append_record(
        StepAttemptRecord(
            id="attempt-2", lane="thread", run_id="run-2", step="assistant", attempt=1, result_entry_id="assistant-2"
        )
    )

    assert [record.id for record in await session.find_records(RecordQuery(lane="thread"))] == ["attempt-2", "run-2"]
    assert [
        record.id for record in await session.find_records(RecordQuery(type="step_attempt", order="oldestFirst"))
    ] == [
        "attempt-1",
        "attempt-2",
    ]
    assert [record.id for record in await session.find_records(RecordQuery(run_id="run-1", after_seq=1))] == [
        "attempt-1"
    ]
    assert [record.id for record in await session.find_records(RecordQuery(limit=1))] == ["attempt-2"]


async def _case_filters_operation_starts_by_kind(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.append_record(_operation_started("run-old", lane="main", kind="run"))
    await session.append_record(
        OperationFinishedRecord(id="run-old-finished", lane="main", run_id="run-old", outcome="completed")
    )
    await session.append_record(_operation_started("compaction", lane="main", kind="compaction"))
    await session.append_record(
        OperationFinishedRecord(id="compaction-finished", lane="main", run_id="compaction", outcome="completed")
    )
    await session.append_record(_operation_started("navigation", lane="main", kind="navigation"))
    await session.append_record(
        OperationFinishedRecord(id="navigation-finished", lane="main", run_id="navigation", outcome="completed")
    )
    await session.append_record(_operation_started("run-new", lane="main", kind="run"))

    assert [
        record.id
        for record in await session.find_records(
            RecordQuery(type="operation_started", operation_kind="run", order="oldestFirst")
        )
    ] == ["run-old", "run-new"]
    assert [
        record.id
        for record in await session.find_records(RecordQuery(type="operation_started", operation_kind="compaction"))
    ] == ["compaction"]
    assert [
        record.id
        for record in await session.find_records(RecordQuery(type="operation_started", operation_kind="navigation"))
    ] == ["navigation"]
    assert [
        record.id
        for record in await session.find_records(RecordQuery(type="operation_started", operation_kind="run", limit=1))
    ] == ["run-new"]


async def _case_enforces_one_open_operation(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    assert await session.find_open_operations("main", limit=2) == []

    first = await session.append_record(_operation_started("first", lane="main", kind="run"))
    assert await session.find_open_operations("main", limit=2) == [first]
    await _rejects_with_code(session.append_record(_operation_started("second", lane="main", kind="run")), "storage")
    assert await session.find_open_operations("main", limit=2) == [first]

    await session.append_record(
        OperationFinishedRecord(id="finish-first", lane="main", run_id=first.id, outcome="completed")
    )
    assert await session.find_open_operations("main", limit=2) == []


async def _case_earlier_finish_does_not_close_later_start(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.append_record(
        OperationFinishedRecord(id="finish-before-start", lane="main", run_id="run", outcome="completed")
    )
    started = await session.append_record(_operation_started("run", lane="main", kind="run"))
    assert await session.find_open_operations("main", limit=2) == [started]


async def _case_scopes_open_operations(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.create_lane("thread", None)
    main_run = await session.append_record(_operation_started("main-run", lane="main", kind="run"))
    thread_navigation = await session.append_record(
        _operation_started("thread-navigation", lane="thread", kind="navigation")
    )

    assert await session.find_open_operations("main") == [main_run]
    assert await session.find_open_operations("main", limit=1) == [main_run]
    assert await session.find_open_operations("thread", limit=2) == [thread_navigation]


async def _case_immutable_open_operations(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    committed = await session.append_record(_operation_started("run", lane="main", kind="run"))
    open_operations = await session.find_open_operations("main")
    read = open_operations[0] if open_operations else None
    assert read is not None and read.intent.kind == "run", "Expected an open run operation"
    read.intent.original_prompt.append(_create_user_message("mutated"))

    assert await session.find_open_operations("main") == [committed]


async def _case_facts_and_statistics(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    # Step 2 relaxation (PROPER_MT_DESIGN.md): messages are frozen values now,
    # so the usage is attached by construction instead of mutation.
    assistant = dataclasses.replace(
        _create_assistant_message("answer"),
        usage=Usage(
            input=10,
            output=5,
            cache_read=3,
            cache_write=2,
            total_tokens=20,
            cost=UsageCost(input=1, output=2, cache_read=3, cache_write=4, total=10),
        ),
    )
    await session.append_entry(MessageEntry(id="user", message=_create_user_message("question")), "main")
    await session.append_entry(MessageEntry(id="assistant", message=assistant), "main")
    await session.append_record(
        UsageRecord(
            id="assistant-usage",
            lane="main",
            cause="assistant",
            run_id="run",
            entry_id="assistant",
            attempt=1,
            stop_reason="stop",
            usage=assistant.usage,
        )
    )
    await session.append_record(
        UsageRecord(
            id="deferred-usage",
            lane="main",
            cause="deferred_fetch",
            run_id="run",
            entry_id="deferred-result",
            attempt=1,
            stop_reason="deferred",
            usage=Usage(),
        )
    )
    await session.create_lane("thread", "assistant")
    await session.append_record(
        UsageRecord(
            id="correction",
            lane="thread",
            cause="adjustment",
            details={"reason": "provider correction"},
            usage=Usage(
                input=-2,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=-2,
                cost=UsageCost(input=-0.5, output=0, cache_read=0, cache_write=0, total=-0.5),
            ),
        )
    )
    await session.set_name("First")
    await session.set_name("Second")
    await session.set_label("user", "keep")
    await session.set_label("user", None)
    await _rejects_with_code(session.set_label("missing", "checkpoint"), "not_found")

    assert await session.get_name() == "Second"
    assert await session.get_label("user") is None
    usage_records = await session.find_records(RecordQuery(type="usage", order="oldestFirst"))
    assert [record.cause for record in usage_records] == ["assistant", "deferred_fetch", "adjustment"]
    deferred_usage = next((record for record in usage_records if record.cause == "deferred_fetch"), None)
    assert deferred_usage is not None, "Expected deferred usage record"
    assert deferred_usage.stop_reason == "deferred"
    assert await session.get_stats() == SessionStats(
        message_count=2, cached_tokens=3, uncached_tokens=10, total_tokens=18, cost_total=9.5
    )


async def _case_clears_session_names(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.set_name("Temporary")
    await session.set_name(None)

    assert await session.get_name() is None
    assert await session.get_log() == [
        NameFactLogItem(seq=1, name="Temporary"),
        NameFactLogItem(seq=2, name=None),
    ]

    metadata = await session.get_metadata()
    reopened = await repository.open(metadata)
    assert await reopened.get_name() is None
    assert await reopened.get_log() == [
        NameFactLogItem(seq=1, name="Temporary"),
        NameFactLogItem(seq=2, name=None),
    ]

    fork = await repository.fork(metadata, ForkOptions(), SessionCreateOptions(id="fork"))
    assert await fork.get_name() is None


async def _case_immutable_reads(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="immutable"))
    metadata = await session.get_metadata()
    data = {"nested": {"value": 1}}
    await session.append_entry(CustomEntry(id="custom", custom_type="note", data=data), "main")
    data["nested"]["value"] = 50
    read = await session.get_entry("custom")
    assert read is not None and read.type == "custom", "Expected custom entry"
    read.data["nested"]["value"] = 99
    read_metadata = await session.get_metadata()
    read_metadata.id = "changed"
    log = await session.get_log()
    assert log and log[0].kind == "entry" and log[0].entry.type == "custom", "Expected entry log"
    log[0].entry.data["nested"]["value"] = 100

    assert await session.get_metadata() == metadata
    assert await session.get_entry("custom") == CustomEntry(
        id="custom",
        custom_type="note",
        data={"nested": {"value": 1}},
        parent_id=None,
        seq=1,
        timestamp=read.timestamp,
    )


async def _case_validates_lane_lifecycle(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await _rejects_with_code(session.create_lane("main", None), "already_exists")
    await _rejects_with_code(session.create_lane("thread", "missing"), "not_found")
    await _rejects_with_code(session.move_lane("missing", None), "invalid_lane")


async def _case_binds_lane_views(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    root = await session.append_message(_create_user_message("root"))
    await session.create_lane("thread", root)
    thread = session.view("thread")
    writes = [
        tonio.spawn(session.append_message(_create_user_message("main"))),
        tonio.spawn(thread.append_message(_create_user_message("thread"))),
    ]
    main_child = await writes[0]
    thread_child = await writes[1]

    assert await session.get_leaf_id() == main_child
    assert await thread.get_leaf_id() == thread_child
    assert await _entry_ids(session.find_entries_on_branch(BranchQuery(order="oldestFirst"))) == [root, main_child]
    assert await _entry_ids(thread.find_entries_on_branch(BranchQuery(order="oldestFirst"))) == [root, thread_child]
    empty = await repository.create(SessionCreateOptions(id="empty"))
    assert await empty.find_entries_on_branch() == []


async def _case_appends_provisioned_entries(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    entry = await session.append_entry(CustomEntry(id="provisioned", custom_type="note", data={"value": 1}), "main")

    assert entry.custom_type == "note"
    assert (entry.id, entry.parent_id, entry.seq) == ("provisioned", None, 1)
    assert await session.get_leaf_id() == "provisioned"


async def _case_persists_termination_decisions(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    entry = await session.append_entry(
        MessageEntry(
            id="tool-result",
            message=ToolResultMessage(
                tool_call_id="call-1",
                tool_name="example",
                content=[TextContent(text="done")],
                is_error=False,
                timestamp=1,
            ),
            terminate=True,
        ),
        "main",
    )

    assert entry.terminate is True
    stored = await session.get_entry(entry.id)
    assert stored is not None and stored.type == "message", "Expected message entry"
    assert stored.terminate is True
    assert await session.find_entries() == [entry]
    log = await session.get_log()
    assert [item.kind for item in log] == ["entry"]
    assert (log[0].seq, log[0].entry) == (entry.seq, entry)


async def _case_rejects_non_json_entries(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic

    for data in (
        {"value": float("nan")},
        [float("inf")],
        {"value": set()},
        {"value": object()},
        {"value": b"bytes"},
        {1: "non-string key"},
        cyclic,
    ):
        await _rejects_with_code(session.append_custom_entry("invalid", data), "invalid_payload")

    assert await session.get_leaf_id() is None
    assert await session.find_entries() == []
    assert await session.get_log() == []
    valid_id = await session.append_custom_entry("valid", {"value": 1})
    valid = await session.get_entry(valid_id)
    assert valid is not None and valid.seq == 1


async def _case_rejects_non_json_records(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    for id, value in (("set-record", set()), ("nan-record", float("nan"))):
        await _rejects_with_code(
            session.append_record(
                ToolStartedRecord(
                    id=id,
                    lane="main",
                    run_id="run",
                    assistant_entry_id="assistant",
                    tool_index=0,
                    tool_call_id="call",
                    tool_name="example",
                    effective_args={"value": value},
                    result_entry_id="result",
                    replay="never",
                )
            ),
            "invalid_payload",
        )

    assert await session.find_records() == []
    assert await session.get_log() == []
    valid = await session.append_record(_operation_started("valid-record", lane="main", kind="run"))
    assert valid.seq == 1


async def _case_linearizes_concurrent_writes(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="session"))
    await session.append_entry(MessageEntry(id="root", message=_create_user_message("root")), "main")
    await session.create_lane("thread", "root")
    completion_order: list[str] = []

    async def tracked(entry: CustomEntry, lane: str) -> CustomEntry:
        committed = await session.append_entry(entry, lane)
        completion_order.append(committed.id)
        return committed

    writes = [
        tonio.spawn(tracked(CustomEntry(id="main-1", custom_type="note"), "main")),
        tonio.spawn(tracked(CustomEntry(id="thread-1", custom_type="note"), "thread")),
        tonio.spawn(tracked(CustomEntry(id="main-2", custom_type="note"), "main")),
        tonio.spawn(tracked(CustomEntry(id="thread-2", custom_type="note"), "thread")),
    ]
    entries = [await write for write in writes]
    commit_order = [entry.id for entry in sorted(entries, key=lambda entry: entry.seq)]

    assert len({entry.seq for entry in entries}) == len(entries)
    assert completion_order == commit_order
    concurrent_ids = {entry.id for entry in entries}
    assert [
        item.entry.id for item in await session.get_log() if item.kind == "entry" and item.entry.id in concurrent_ids
    ] == commit_order
    sequences = [item.seq for item in await session.get_log()]
    assert sequences == sorted(sequences)


async def _case_creates_lists_and_opens(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="one"))
    entry_id = await session.append_message(_create_user_message("persisted"))
    metadata = await session.get_metadata()

    listed = await repository.list()
    assert len(listed) == 1
    assert listed[0].id == metadata.id
    assert listed[0].created_at == metadata.created_at
    assert listed[0].parent_session_id == metadata.parent_session_id
    assert await _entry_ids((await repository.open(metadata)).find_entries()) == [entry_id]
    await _rejects_with_code(repository.create(SessionCreateOptions(id="one")), "already_exists")


async def _case_deletes_idempotently(repository: SessionRepo) -> None:
    session = await repository.create(SessionCreateOptions(id="one"))
    metadata = await session.get_metadata()

    await repository.delete(metadata)
    await _rejects_with_code(repository.open(metadata), "not_found")
    await repository.delete(metadata)


async def _case_forks_one_branch(repository: SessionRepo) -> None:
    source = await repository.create(SessionCreateOptions(id="source"))
    root = await source.append_message(_create_user_message("root"))
    shared = await source.append_message(_create_assistant_message("shared"))
    await source.create_lane("thread", shared)
    thread_child = await source.view("thread").append_message(_create_user_message("thread"))
    main_child = await source.append_message(_create_user_message("main"))
    await source.set_name("Source")
    await source.set_label(shared, "copied")
    await source.set_label(thread_child, "excluded")
    await source.append_record(_operation_started("run", lane="main", kind="run"))
    await source.append_record(
        UsageRecord(
            id="source-usage",
            lane="main",
            cause="adjustment",
            usage=Usage(
                input=10,
                output=5,
                cache_read=3,
                cache_write=2,
                total_tokens=20,
                cost=UsageCost(input=1, output=2, cache_read=3, cache_write=4, total=10),
            ),
        )
    )

    fork = await repository.fork(
        await source.get_metadata(),
        ForkOptions(scope="branch", entry_id=main_child, position="at"),
        SessionCreateOptions(id="branch-fork"),
    )

    assert await _entry_ids(fork.find_entries(EntryQuery(order="oldestFirst"))) == [root, shared, main_child]
    assert await fork.get_lanes() == [LanePointer(lane="main", leaf_id=main_child)]
    assert await fork.get_name() == "Source"
    assert await fork.get_label(shared) == "copied"
    assert await fork.get_label(thread_child) is None
    assert await fork.find_records() == []
    assert await fork.get_stats() == SessionStats(
        message_count=3, cached_tokens=0, uncached_tokens=0, total_tokens=0, cost_total=0
    )
    await fork.append_message(_create_user_message("after fork"))
    assert (await fork.get_stats()).message_count == 4
    metadata = await fork.get_metadata()
    assert (metadata.id, metadata.parent_session_id) == ("branch-fork", "source")


async def _case_forks_complete_tree(repository: SessionRepo) -> None:
    source = await repository.create(SessionCreateOptions(id="source"))
    root = await source.append_message(_create_user_message("root"))
    await source.create_lane("thread", root)
    main_child = await source.append_message(_create_user_message("main"))
    thread_child = await source.view("thread").append_message(_create_user_message("thread"))
    await source.set_label(thread_child, "thread-tip")

    fork = await repository.fork(
        await source.get_metadata(), ForkOptions(scope="tree"), SessionCreateOptions(id="tree-fork")
    )
    assert await _entry_ids(fork.find_entries(EntryQuery(order="oldestFirst"))) == [root, main_child, thread_child]
    assert await fork.get_lanes() == [
        LanePointer(lane="main", leaf_id=main_child),
        LanePointer(lane="thread", leaf_id=thread_child),
    ]
    assert await fork.get_label(thread_child) == "thread-tip"
    assert (await fork.get_stats()).message_count == 3
    lane_items = [item for item in await fork.get_log() if item.kind == "lane"]
    assert [(item.seq, item.lane, item.leaf_id) for item in lane_items] == [
        (4, "main", main_child),
        (5, "thread", thread_child),
    ]


async def _case_forks_before_entry(repository: SessionRepo) -> None:
    source = await repository.create(SessionCreateOptions(id="source"))
    root = await source.append_message(_create_user_message("root"))
    tail = await source.append_message(_create_user_message("tail"))
    fork = await repository.fork(
        await source.get_metadata(), ForkOptions(entry_id=tail), SessionCreateOptions(id="fork")
    )

    assert await _entry_ids(fork.find_entries(EntryQuery(order="oldestFirst"))) == [root]
    assert await fork.get_leaf_id() == root
    assert await source.get_leaf_id() == tail
    before_default_target = await repository.fork(
        await source.get_metadata(), ForkOptions(position="before"), SessionCreateOptions(id="before-default-target")
    )
    assert await _entry_ids(before_default_target.find_entries(EntryQuery(order="oldestFirst"))) == [root]
    assert await before_default_target.get_leaf_id() == root

    at_default_target = await repository.fork(
        await source.get_metadata(), ForkOptions(position="at"), SessionCreateOptions(id="at-default-target")
    )
    assert await _entry_ids(at_default_target.find_entries(EntryQuery(order="oldestFirst"))) == [root, tail]
    assert await at_default_target.get_leaf_id() == tail
    await _rejects_with_code(
        repository.fork(await source.get_metadata(), ForkOptions(entry_id="missing")), "invalid_fork_target"
    )


async def _case_validates_default_fork_target(repository: SessionRepo) -> None:
    source = await repository.create(SessionCreateOptions(id="source-with-custom-leaf"))
    await source.append_custom_entry("not-a-message")

    await _rejects_with_code(
        repository.fork(await source.get_metadata(), ForkOptions(), SessionCreateOptions(id="fork")),
        "invalid_fork_target",
    )


def create_session_backend_conformance(
    factory: SessionBackendFixtureFactory,
) -> list[SessionBackendConformanceCase]:
    """Creates the session backend conformance cases. Each case creates and disposes its own fixture."""
    specs: list[tuple[str, str, _ConformanceTest]] = [
        (
            "entries and lanes",
            "assigns parents and one sequence across every mutation",
            _case_assigns_parents_and_sequence,
        ),
        (
            "records and log",
            "commits records and lane moves as separate mutations",
            _case_commits_records_and_lane_moves,
        ),
        ("entries and lanes", "rejects duplicate ids without changing state", _case_rejects_duplicate_ids),
        ("entries and lanes", "isolates lanes while sharing the tree", _case_isolates_lanes),
        ("queries and facts", "rejects invalid queries before empty reads", _case_rejects_invalid_queries),
        (
            "queries and facts",
            "supports bounded filtered and cursor-based queries",
            _case_bounded_filtered_cursor_queries,
        ),
        ("records and log", "keeps lane names permanent with their recovery records", _case_keeps_lane_names_permanent),
        (
            "records and log",
            "persists queue cancellation without consuming its target",
            _case_persists_queue_cancellation,
        ),
        ("records and log", "filters records by lane type run sequence and order", _case_filters_records),
        ("records and log", "filters operation starts by operation kind", _case_filters_operation_starts_by_kind),
        ("records and log", "tracks and enforces one open operation per lane", _case_enforces_one_open_operation),
        (
            "records and log",
            "does not let an earlier finish close a later start",
            _case_earlier_finish_does_not_close_later_start,
        ),
        ("records and log", "scopes open operations by lane and limit", _case_scopes_open_operations),
        ("validation and immutability", "returns immutable open-operation records", _case_immutable_open_operations),
        (
            "queries and facts",
            "keeps latest-value facts and computes ledger statistics across lanes",
            _case_facts_and_statistics,
        ),
        ("queries and facts", "clears session names durably", _case_clears_session_names),
        ("validation and immutability", "returns immutable copies from reads", _case_immutable_reads),
        ("entries and lanes", "validates lane lifecycle and targets", _case_validates_lane_lifecycle),
        ("entries and lanes", "binds lane views without caching leaves", _case_binds_lane_views),
        ("entries and lanes", "appends provisioned entries with their existing ids", _case_appends_provisioned_entries),
        ("entries and lanes", "persists tool-result termination decisions", _case_persists_termination_decisions),
        (
            "validation and immutability",
            "rejects non-JSON entries before storage mutation",
            _case_rejects_non_json_entries,
        ),
        (
            "validation and immutability",
            "rejects non-JSON records before storage mutation",
            _case_rejects_non_json_records,
        ),
        ("entries and lanes", "linearizes concurrent writes across two lanes", _case_linearizes_concurrent_writes),
        ("repository and forks", "creates lists and opens sessions", _case_creates_lists_and_opens),
        ("repository and forks", "deletes sessions idempotently", _case_deletes_idempotently),
        ("repository and forks", "forks one branch with selected facts and no records", _case_forks_one_branch),
        ("repository and forks", "forks a complete tree with lanes and facts", _case_forks_complete_tree),
        ("repository and forks", "forks before an entry without modifying the source", _case_forks_before_entry),
        ("repository and forks", "validates the default fork target", _case_validates_default_fork_target),
    ]
    return [_Case(group=group, name=name, _factory=factory, _test=test) for group, name, test in specs]
