"""JSONL v4 per-session storage round-trips (mirror of pi
agent/test/harness/session/jsonl-storage.test.ts)."""

import pytest
import tonio.colored as tonio

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.session.jsonl import JsonlSessionCreateOptions, JsonlSessionRepo, JsonlSessionRepoOptions
from pidrei_agent.harness.session.session import Session
from pidrei_agent.harness.session.types import (
    AbortRequestedRecord,
    ActiveToolsEntry,
    BranchQuery,
    BranchSummaryEntry,
    CompactionEntry,
    CompactionIntent,
    CustomEntry,
    Entry,
    EntryCursor,
    EntryQuery,
    LaneRecord,
    MessageEntry,
    ModelChangeEntry,
    NavigationIntent,
    OperationFinishedRecord,
    OperationStartedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    RecordQuery,
    RunIntent,
    SessionError,
    SessionStats,
    StepAttemptRecord,
    ThinkingLevelEntry,
    ToolStartedRecord,
    UsageRecord,
    WriteDeferredRecord,
)
from pidrei_agent.harness.types import FileError, err
from pidrei_ai.types import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)
from tests.session_helpers import create_temp_dir


def create_repository(root: str) -> JsonlSessionRepo:
    return JsonlSessionRepo(JsonlSessionRepoOptions(fs=LocalExecutionEnv(cwd=root), sessions_root=root))


def user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=1)


def create_usage(multiplier: int) -> Usage:
    return Usage(
        input=multiplier,
        output=multiplier * 2,
        cache_read=multiplier * 3,
        cache_write=multiplier * 4,
        total_tokens=multiplier * 10,
        cost=UsageCost(
            input=multiplier * 0.1,
            output=multiplier * 0.2,
            cache_read=multiplier * 0.3,
            cache_write=multiplier * 0.4,
            total=multiplier * 1.0,
        ),
    )


async def reopen(root: str, session: Session) -> Session:
    return await create_repository(root).open(await session.get_metadata())


@pytest.mark.tonio
async def test_round_trips_every_entry_type_and_bounded_branch_queries():
    root = create_temp_dir()
    session = await create_repository(root).create(JsonlSessionCreateOptions(id="entries", cwd=root))
    committed: list[Entry] = []
    committed.append(await session.append_entry(MessageEntry(id="message", message=user_message("question")), "main"))
    committed.append(
        await session.append_entry(
            MessageEntry(
                id="assistant-tool-call",
                message=AssistantMessage(
                    content=[
                        TextContent(text="I'll inspect it."),
                        ToolCall(id="call-1", name="read", arguments={"path": "README.md"}),
                    ],
                    api="anthropic-messages",
                    provider="anthropic",
                    model="claude-sonnet-4-5",
                    usage=create_usage(1),
                    stop_reason="toolUse",
                    timestamp=2,
                ),
            ),
            "main",
        )
    )
    committed.append(
        await session.append_entry(
            MessageEntry(
                id="tool-result",
                message=ToolResultMessage(
                    tool_call_id="call-1",
                    tool_name="read",
                    content=[TextContent(text="contents")],
                    details={"path": "README.md"},
                    usage=create_usage(2),
                    is_error=False,
                    timestamp=3,
                ),
                terminate=True,
            ),
            "main",
        )
    )
    committed.append(
        await session.append_entry(
            ModelChangeEntry(id="model", provider="anthropic", model_id="claude-sonnet-4-5"), "main"
        )
    )
    committed.append(await session.append_entry(ThinkingLevelEntry(id="thinking", thinking_level="high"), "main"))
    committed.append(
        await session.append_entry(ActiveToolsEntry(id="tools", active_tool_names=["read", "bash"]), "main")
    )
    committed.append(
        await session.append_entry(
            CompactionEntry(
                id="compaction",
                summary="summary",
                retained_tail=[user_message("retained")],
                tokens_before=123,
                details={"source": "test"},
                usage=create_usage(1),
            ),
            "main",
        )
    )
    committed.append(
        await session.append_entry(
            BranchSummaryEntry(
                id="branch-summary",
                from_id="message",
                summary="branch",
                details={"reason": "navigation"},
                usage=create_usage(2),
            ),
            "main",
        )
    )
    committed.append(
        await session.append_entry(CustomEntry(id="custom", custom_type="note", data={"nested": {"value": 1}}), "main")
    )

    restored = await reopen(root, session)
    assert await restored.find_entries(EntryQuery(order="oldestFirst")) == committed
    assert [entry.id for entry in await restored.find_entries_on_branch(BranchQuery(stop_at_type="compaction"))] == [
        "custom",
        "branch-summary",
        "compaction",
    ]
    assert [
        entry.id
        for entry in await restored.find_entries(
            EntryQuery(order="oldestFirst", cursor=EntryCursor(after_seq=committed[5].seq), limit=2)
        )
    ] == ["compaction", "branch-summary"]
    assert [entry.id for entry in await restored.find_entries(EntryQuery(custom_type="note"))] == ["custom"]
    assert await restored.get_stats() == SessionStats(
        message_count=3, cached_tokens=0, uncached_tokens=0, total_tokens=0, cost_total=0
    )

    custom = await restored.get_entry("custom")
    assert custom is not None and custom.type == "custom", "Expected custom entry"
    custom.data["nested"]["value"] = 99
    log_custom = next(
        (item for item in await restored.get_log() if item.kind == "entry" and item.entry.type == "custom"), None
    )
    assert log_custom is not None, "Expected custom entry in log"
    log_custom.entry.data["nested"]["value"] = 100

    assert await restored.get_entry("custom") == committed[-1]
    assert await restored.find_entries(EntryQuery(order="oldestFirst")) == committed


@pytest.mark.tonio
async def test_round_trips_every_record_type_recovery_projection_and_ledger_statistics():
    root = create_temp_dir()
    session = await create_repository(root).create(JsonlSessionCreateOptions(id="records", cwd=root))
    await session.append_custom_entry("anchor")
    records: list[LaneRecord] = []

    async def append(record: LaneRecord) -> None:
        records.append(await session.append_record(record))

    await append(
        OperationStartedRecord(
            id="run",
            lane="main",
            source_leaf_id="anchor",
            intent=RunIntent(
                original_prompt=[user_message("prompt")],
                initial_messages=[MessageEntry(id="initial", message=user_message("initial"))],
                system_prompt_override="system",
                resume_data={"extension": {"version": 1}},
            ),
        )
    )
    await append(
        QueueEnqueuedRecord(
            id="steer",
            lane="main",
            queue="steer",
            run_id="run",
            target=MessageEntry(id="steer-message", message=user_message("steer")),
        )
    )
    await append(
        QueueEnqueuedRecord(
            id="follow-up",
            lane="main",
            queue="followUp",
            run_id="run",
            target=MessageEntry(id="follow-up-message", message=user_message("follow up")),
        )
    )
    await append(
        StepAttemptRecord(
            id="assistant-attempt",
            lane="main",
            run_id="run",
            step="assistant",
            attempt=1,
            result_entry_id="assistant-result",
        )
    )
    await append(
        ToolStartedRecord(
            id="tool",
            lane="main",
            run_id="run",
            assistant_entry_id="assistant-result",
            tool_index=0,
            tool_call_id="call-1",
            tool_name="read",
            effective_args={"path": "README.md"},
            result_entry_id="tool-result",
            replay="safe",
        )
    )
    await append(
        WriteDeferredRecord(
            id="deferred-write",
            lane="main",
            run_id="run",
            target=CustomEntry(id="deferred-entry", custom_type="fact", data={"value": True}),
        )
    )
    await append(
        UsageRecord(
            id="assistant-usage",
            lane="main",
            cause="assistant",
            run_id="run",
            entry_id="assistant-result",
            attempt=1,
            stop_reason="stop",
            usage=create_usage(1),
        )
    )
    await append(
        UsageRecord(
            id="deferred-usage",
            lane="main",
            cause="deferred_fetch",
            run_id="run",
            entry_id="deferred-result",
            attempt=1,
            stop_reason="deferred",
            usage=create_usage(2),
        )
    )
    await append(
        UsageRecord(
            id="tool-usage",
            lane="main",
            cause="tool",
            run_id="run",
            entry_id="tool-result",
            tool_call_id="call-1",
            usage=create_usage(3),
        )
    )
    await append(
        UsageRecord(
            id="hook-usage", lane="main", cause="hook", run_id="run", entry_id="hook-result", usage=create_usage(4)
        )
    )
    await append(
        UsageRecord(
            id="adjustment", lane="main", cause="adjustment", details={"reason": "correction"}, usage=create_usage(5)
        )
    )
    await append(AbortRequestedRecord(id="abort", lane="main", run_id="run"))
    await append(OperationFinishedRecord(id="run-finished", lane="main", run_id="run", outcome="aborted"))
    await append(
        QueueEnqueuedRecord(
            id="next-run",
            lane="main",
            queue="nextRun",
            target=MessageEntry(id="next-message", message=user_message("next")),
        )
    )
    await append(QueueCancelledRecord(id="queue-cancelled", lane="main", entry_id="next-message"))
    await append(
        OperationStartedRecord(
            id="compaction",
            lane="main",
            source_leaf_id="anchor",
            intent=CompactionIntent(custom_instructions="short", result_entry_id="compaction-result"),
        )
    )
    await append(
        StepAttemptRecord(
            id="compaction-attempt",
            lane="main",
            run_id="compaction",
            step="compaction",
            attempt=1,
            result_entry_id="compaction-result",
            compaction_reason="manual",
        )
    )
    await append(
        OperationFinishedRecord(id="compaction-finished", lane="main", run_id="compaction", outcome="completed")
    )
    await append(
        OperationStartedRecord(
            id="navigation",
            lane="main",
            source_leaf_id="anchor",
            intent=NavigationIntent(
                target_id=None,
                summarize=True,
                custom_instructions="summarize",
                label="checkpoint",
                summary_entry_id="navigation-summary",
            ),
        )
    )
    await append(
        StepAttemptRecord(
            id="branch-attempt",
            lane="main",
            run_id="navigation",
            step="branch_summary",
            attempt=1,
            result_entry_id="navigation-summary",
        )
    )

    restored = await reopen(root, session)
    assert await restored.find_records(RecordQuery(order="oldestFirst")) == records
    assert [
        record.id
        for record in await restored.find_records(RecordQuery(type="operation_started", operation_kind="run", limit=1))
    ] == ["run"]
    assert [
        record.id for record in await restored.find_records(RecordQuery(run_id="compaction", order="oldestFirst"))
    ] == ["compaction", "compaction-attempt", "compaction-finished"]
    assert [
        record.id
        for record in await restored.find_records(RecordQuery(type="usage", after_seq=records[6].seq, limit=2))
    ] == ["adjustment", "hook-usage"]
    assert [record.id for record in await restored.find_open_operations("main", limit=2)] == ["navigation"]
    assert await restored.get_stats() == SessionStats(
        message_count=0, cached_tokens=45, uncached_tokens=75, total_tokens=150, cost_total=15
    )

    started_records = await restored.find_records(RecordQuery(type="operation_started", operation_kind="run"))
    started = started_records[0] if started_records else None
    assert started is not None and started.intent.kind == "run", "Expected restored run record"
    started.intent.original_prompt.append(user_message("mutated"))
    assert await restored.find_records(RecordQuery(order="oldestFirst")) == records


@pytest.mark.tonio
async def test_persists_concurrent_cross_lane_writes_in_shared_sequence_order():
    root = create_temp_dir()
    session = await create_repository(root).create(JsonlSessionCreateOptions(id="concurrent", cwd=root))
    root_entry = await session.append_entry(CustomEntry(id="root", custom_type="root"), "main")
    await session.create_lane("thread", root_entry.id)

    writes = [
        tonio.spawn(session.append_entry(CustomEntry(id="main-1", custom_type="note"), "main")),
        tonio.spawn(session.append_entry(CustomEntry(id="thread-1", custom_type="note"), "thread")),
        tonio.spawn(session.append_entry(CustomEntry(id="main-2", custom_type="note"), "main")),
        tonio.spawn(session.append_entry(CustomEntry(id="thread-2", custom_type="note"), "thread")),
    ]
    entries = [await write for write in writes]
    commit_order = [entry.id for entry in sorted(entries, key=lambda entry: entry.seq)]

    restored = await reopen(root, session)
    restored_concurrent_entries = [
        item.entry for item in await restored.get_log() if item.kind == "entry" and item.entry.id != "root"
    ]
    assert [entry.id for entry in restored_concurrent_entries] == commit_order
    assert len({entry.seq for entry in restored_concurrent_entries}) == len(entries)
    assert [item.seq for item in await restored.get_log()] == [1, 2, 3, 4, 5, 6]


@pytest.mark.tonio
async def test_rejects_non_json_payloads_without_changing_the_durable_prefix():
    root = create_temp_dir()
    session = await create_repository(root).create(JsonlSessionCreateOptions(id="validation", cwd=root))
    metadata = await session.get_metadata()
    with open(metadata.path, encoding="utf-8") as file:
        prefix = file.read()
    cyclic: dict = {}
    cyclic["self"] = cyclic

    with pytest.raises(SessionError) as entry_error:
        await session.append_custom_entry("invalid", cyclic)
    assert entry_error.value.code == "invalid_payload"
    # pi guards `undefined` values, which JSON.stringify would silently omit; the
    # Python guard rejects the values json.dumps cannot represent at all.
    with pytest.raises(SessionError) as record_error:
        await session.append_record(
            ToolStartedRecord(
                id="invalid-record",
                lane="main",
                run_id="run",
                assistant_entry_id="assistant",
                tool_index=0,
                tool_call_id="call",
                tool_name="read",
                effective_args={"value": set()},
                result_entry_id="result",
                replay="never",
            )
        )
    assert record_error.value.code == "invalid_payload"
    with open(metadata.path, encoding="utf-8") as file:
        assert file.read() == prefix

    restored = await reopen(root, session)
    assert await restored.get_log() == []
    valid = await restored.append_entry(CustomEntry(id="valid", custom_type="note", data={"value": 1}), "main")
    assert valid.seq == 1
    assert await (await reopen(root, restored)).get_entry("valid") == valid


class _FailingFirstAppendEnv(LocalExecutionEnv):
    def __init__(self, cwd: str):
        super().__init__(cwd=cwd)
        self._failed = False

    async def append_file(self, path, content, cancel=None):
        if not self._failed:
            self._failed = True
            return err(FileError("unknown", "injected append failure"))
        return await super().append_file(path, content, cancel)


@pytest.mark.tonio
async def test_does_not_advance_state_or_poison_the_write_queue_after_an_append_failure():
    root = create_temp_dir()
    env = _FailingFirstAppendEnv(cwd=root)
    repository = JsonlSessionRepo(JsonlSessionRepoOptions(fs=env, sessions_root=root))
    session = await repository.create(JsonlSessionCreateOptions(id="append-failure", cwd=root))

    with pytest.raises(SessionError) as excinfo:
        await session.append_custom_entry("rejected")
    assert excinfo.value.code == "storage"
    assert await session.get_log() == []
    committed = await session.append_entry(CustomEntry(id="committed", custom_type="note"), "main")
    assert committed.seq == 1

    reopened = await create_repository(root).open(await session.get_metadata())
    log = await reopened.get_log()
    assert [item.kind for item in log] == ["entry"]
    assert (log[0].seq, log[0].entry) == (1, committed)
