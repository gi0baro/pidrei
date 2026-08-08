"""Crash-recovery reducer (mirror of pi agent/test/harness/reducer.test.ts)."""

import copy
from dataclasses import replace
from typing import Any

import pytest

from pidrei_agent.harness.reducer import (
    EffectiveLaneConfiguration,
    LaneReductionInput,
    ModelRef,
    OperationTargets,
    RecordLogCorruption,
    RecordLogSlice,
    StepState,
    reduce_lane_state,
    validate_record_log,
)
from pidrei_agent.harness.session.types import (
    AbortRequestedRecord,
    ActiveToolsEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CompactionIntent,
    Entry,
    LaneRecord,
    MessageEntry,
    ModelChangeEntry,
    NavigationIntent,
    OperationFinishedRecord,
    OperationStartedRecord,
    QueueCancelledRecord,
    QueueEnqueuedRecord,
    RunIntent,
    StepAttemptRecord,
    ThinkingLevelEntry,
    ToolStartedRecord,
    UsageRecord,
    WriteDeferredRecord,
)
from pidrei_ai.types import (
    AssistantMessage,
    DeferredHandle,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


USAGE = Usage(input=1, output=1, total_tokens=2)


def user_message(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=1)


def assistant_message(content: list[Any], stop_reason: str = "stop") -> AssistantMessage:
    return AssistantMessage(
        content=content,
        api="openai-responses",
        provider="openai",
        model="test-model",
        usage=USAGE,
        stop_reason=stop_reason,
        timestamp=1,
        deferred=(
            DeferredHandle(provider="openai", model_id="test-model", api="openai-responses", id="deferred-1")
            if stop_reason == "deferred"
            else None
        ),
    )


def tool_result_message(tool_call_id: str = "call-1", tool_name: str = "tool-1") -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        content=[TextContent(text="result")],
        is_error=False,
        timestamp=1,
    )


def message_target(id: str, message: Any) -> MessageEntry:
    return MessageEntry(id=id, message=message)


def persisted_entry(target: Entry, seq: int, parent_id: str | None = None) -> Entry:
    return replace(target, parent_id=parent_id, seq=seq, timestamp=seq)


def run_started(
    seq: int = 1, *, id: str = "run-1", initial_messages: list[Entry] | None = None
) -> OperationStartedRecord:
    return OperationStartedRecord(
        id=id,
        lane="main",
        seq=seq,
        timestamp=seq,
        source_leaf_id=None,
        intent=RunIntent(original_prompt=[], initial_messages=initial_messages if initial_messages is not None else []),
    )


def compaction_started(seq: int, result_entry_id: str = "compaction-1") -> OperationStartedRecord:
    return OperationStartedRecord(
        id="compact-1",
        lane="main",
        seq=seq,
        timestamp=seq,
        source_leaf_id="source",
        intent=CompactionIntent(result_entry_id=result_entry_id),
    )


def navigation_started(seq: int, summary_entry_id: str = "summary-1") -> OperationStartedRecord:
    return OperationStartedRecord(
        id="navigate-1",
        lane="main",
        seq=seq,
        timestamp=seq,
        source_leaf_id="source",
        intent=NavigationIntent(target_id="target", summarize=True, summary_entry_id=summary_entry_id),
    )


def attempt(
    seq: int,
    run_id: str,
    step: str,
    attempt_number: int,
    result_entry_id: str,
    compaction_reason: str | None = None,
) -> StepAttemptRecord:
    return StepAttemptRecord(
        id=f"attempt-{seq}",
        lane="main",
        seq=seq,
        timestamp=seq,
        run_id=run_id,
        step=step,
        attempt=attempt_number,
        result_entry_id=result_entry_id,
        compaction_reason=(compaction_reason if compaction_reason is not None else "manual")
        if step == "compaction"
        else None,
    )


def abort_requested(seq: int, run_id: str = "run-1") -> AbortRequestedRecord:
    return AbortRequestedRecord(id=f"abort-{seq}", lane="main", seq=seq, timestamp=seq, run_id=run_id)


def operation_finished(seq: int, run_id: str = "run-1", outcome: str = "completed") -> OperationFinishedRecord:
    return OperationFinishedRecord(
        id=f"finish-{seq}", lane="main", seq=seq, timestamp=seq, run_id=run_id, outcome=outcome
    )


def tool_started(seq: int, **overrides: Any) -> ToolStartedRecord:
    return ToolStartedRecord(
        id=f"tool-start-{seq}",
        lane="main",
        seq=seq,
        timestamp=seq,
        run_id="run-1",
        assistant_entry_id=overrides.get("assistant_entry_id", "assistant-tools"),
        tool_index=overrides.get("tool_index", 0),
        tool_call_id=overrides.get("tool_call_id", "call-1"),
        tool_name=overrides.get("tool_name", "tool-1"),
        effective_args={},
        result_entry_id=overrides.get("result_entry_id", "tool-result-1"),
        replay="never",
    )


def queue_enqueued(seq: int, target: Entry | None = None, queue: str = "steer") -> QueueEnqueuedRecord:
    if target is None:
        target = message_target("queue-1", user_message("queued"))
    return QueueEnqueuedRecord(
        id=f"queue-{seq}",
        lane="main",
        seq=seq,
        timestamp=seq,
        queue=queue,
        target=target,
        run_id=None if queue == "nextRun" else "run-1",
    )


def queue_cancelled(seq: int, entry_id: str = "queue-1", run_id: str | None = "run-1") -> QueueCancelledRecord:
    return QueueCancelledRecord(
        id=f"cancel-{seq}", lane="main", seq=seq, timestamp=seq, entry_id=entry_id, run_id=run_id
    )


def write_deferred(seq: int, target: Entry | None = None) -> WriteDeferredRecord:
    if target is None:
        target = message_target("write-1", user_message("deferred write"))
    return WriteDeferredRecord(id=f"write-{seq}", lane="main", seq=seq, timestamp=seq, run_id="run-1", target=target)


def usage_record(seq: int, result_entry_id: str, stop_reason: str = "error", attempt_number: int = 1) -> UsageRecord:
    return UsageRecord(
        id=f"usage-{seq}",
        lane="main",
        seq=seq,
        timestamp=seq,
        cause="assistant",
        run_id="run-1",
        entry_id=result_entry_id,
        attempt=attempt_number,
        stop_reason=stop_reason,
        usage=USAGE,
    )


def compaction_entry(id: str, seq: int) -> CompactionEntry:
    return CompactionEntry(
        id=id, parent_id=None, seq=seq, timestamp=seq, summary="summary", retained_tail=[], tokens_before=10
    )


def branch_summary_entry(id: str, seq: int) -> BranchSummaryEntry:
    return BranchSummaryEntry(id=id, parent_id="target", seq=seq, timestamp=seq, from_id="source", summary="summary")


def recovery_slice(records: list[LaneRecord], entries: list[Entry] | None = None) -> RecordLogSlice:
    entries = entries if entries is not None else []
    finished = {record.run_id for record in records if record.type == "operation_finished"}
    open_operations = sorted(
        (record for record in records if record.type == "operation_started" and record.id not in finished),
        key=lambda record: -record.seq,
    )
    return RecordLogSlice(lane="main", open_operations=list(open_operations), records=records, entries=entries)


DEFAULTS = EffectiveLaneConfiguration(
    model=ModelRef(provider="default-provider", model_id="default-model"),
    thinking_level="off",
    active_tool_names=["default-tool"],
)


def reduction_input(
    records: list[LaneRecord],
    own_entries: list[Entry] | None = None,
    *,
    entries: list[Entry] | None = None,
    configuration_entries: list[Entry] | None = None,
    leaf_id: str | None | type(...) = ...,
    defaults: EffectiveLaneConfiguration | None = None,
) -> LaneReductionInput:
    own_entries = own_entries if own_entries is not None else []
    slice = recovery_slice(records, [*own_entries, *(entries if entries is not None else [])])
    if leaf_id is ...:
        leaf_id = own_entries[-1].id if own_entries else None
    return LaneReductionInput(
        lane=slice.lane,
        open_operations=slice.open_operations,
        records=slice.records,
        entries=slice.entries,
        leaf_id=leaf_id,
        own_entries=own_entries,
        configuration_entries=configuration_entries if configuration_entries is not None else [],
        defaults=defaults if defaults is not None else DEFAULTS,
    )


def expect_corruption(input: RecordLogSlice, reason: str) -> None:
    with pytest.raises(RecordLogCorruption) as excinfo:
        validate_record_log(input)
    assert excinfo.value.reason == reason


ASSISTANT_TOOLS_ENTRY = persisted_entry(
    message_target(
        "assistant-tools", assistant_message([ToolCall(id="call-1", name="tool-1", arguments={})], "toolUse")
    ),
    3,
)


CORRUPTION_CASES = [
    (
        "multiple operations are open",
        "multiple_open_operations",
        lambda: recovery_slice([run_started(1), run_started(2, id="run-2")]),
    ),
    (
        "a record references an operation that does not exist",
        "unknown_operation",
        lambda: recovery_slice([abort_requested(1, "missing")]),
    ),
    (
        "a record follows its operation finish",
        "record_after_finish",
        lambda: recovery_slice([run_started(1), operation_finished(2), abort_requested(3)]),
    ),
    (
        "attempt numbers skip within one assistant step",
        "non_consecutive_attempt",
        lambda: recovery_slice(
            [
                run_started(1),
                attempt(2, "run-1", "assistant", 1, "assistant-1"),
                attempt(3, "run-1", "assistant", 3, "assistant-2"),
            ]
        ),
    ),
    (
        "a non-compaction attempt carries compactionReason",
        "invalid_compaction_reason",
        lambda: recovery_slice(
            [run_started(1), replace(attempt(2, "run-1", "assistant", 1, "assistant-1"), compaction_reason="manual")]
        ),
    ),
    (
        "a compaction attempt omits compactionReason",
        "invalid_compaction_reason",
        lambda: recovery_slice(
            [run_started(1), replace(attempt(2, "run-1", "compaction", 1, "compaction-1"), compaction_reason=None)]
        ),
    ),
    (
        "steering is enqueued after abort",
        "queue_after_abort",
        lambda: recovery_slice([run_started(1), abort_requested(2), queue_enqueued(3)]),
    ),
    (
        "a queue cancellation has no enqueue",
        "invalid_queue_cancellation",
        lambda: recovery_slice([run_started(1), queue_cancelled(2)]),
    ),
    (
        "a queue cancellation targets an entry that exists",
        "invalid_queue_cancellation",
        lambda: recovery_slice(
            [run_started(1), queue_enqueued(2), queue_cancelled(4)],
            [persisted_entry(message_target("queue-1", user_message("queued")), 3)],
        ),
    ),
    (
        "structural attempts disagree on resultEntryId",
        "inconsistent_step",
        lambda: recovery_slice(
            [
                run_started(1),
                attempt(2, "run-1", "compaction", 1, "compaction-1", "threshold"),
                attempt(3, "run-1", "compaction", 2, "compaction-2", "threshold"),
            ]
        ),
    ),
    (
        "structural attempts disagree on compactionReason",
        "inconsistent_step",
        lambda: recovery_slice(
            [
                run_started(1),
                attempt(2, "run-1", "compaction", 1, "compaction-1", "threshold"),
                attempt(3, "run-1", "compaction", 2, "compaction-1", "overflow"),
            ]
        ),
    ),
    (
        "tool_started does not match the assistant tool call",
        "tool_call_mismatch",
        lambda: recovery_slice(
            [run_started(1), tool_started(4, tool_call_id="different-call")], [ASSISTANT_TOOLS_ENTRY]
        ),
    ),
    (
        "two tool_started records share an invocation identity",
        "duplicate_tool_invocation",
        lambda: recovery_slice(
            [
                run_started(1),
                tool_started(4),
                replace(tool_started(5, result_entry_id="tool-result-2"), id="tool-start-duplicate"),
            ],
            [ASSISTANT_TOOLS_ENTRY],
        ),
    ),
    (
        "a provisioned id exists with different content",
        "provisioned_entry_mismatch",
        lambda: recovery_slice(
            [run_started(1, initial_messages=[message_target("prompt-1", user_message("expected"))])],
            [persisted_entry(message_target("prompt-1", user_message("different")), 2)],
        ),
    ),
    (
        "a deferred assistant message has no handle",
        "invalid_deferred_handle",
        lambda: recovery_slice(
            [run_started(1)],
            [
                persisted_entry(
                    message_target("assistant-deferred", replace(assistant_message([], "deferred"), deferred=None)), 2
                )
            ],
        ),
    ),
]


@pytest.mark.parametrize(
    ("name", "reason", "build_input"), CORRUPTION_CASES, ids=[case[0] for case in CORRUPTION_CASES]
)
def test_record_log_validity_rejects(name, reason, build_input):
    expect_corruption(build_input(), reason)


def test_does_not_mutate_its_bounded_recovery_inputs():
    target = message_target("prompt-1", user_message("hello"))
    start = run_started(1, initial_messages=[target])
    entry = persisted_entry(target, 2)
    input = RecordLogSlice(lane="main", open_operations=[start], records=[start], entries=[entry])
    before = copy.deepcopy(input)

    assert validate_record_log(input) is None
    assert input.records == before.records
    assert input.entries == before.entries


def valid_prefixes(trace: str, actions: list[tuple[str, Any]]) -> list[tuple[str, RecordLogSlice]]:
    cases = []
    for index in range(len(actions)):
        prefix = actions[: index + 1]
        cases.append(
            (
                f"{trace} after action {index + 1}",
                recovery_slice(
                    [value for kind, value in prefix if kind == "record"],
                    [value for kind, value in prefix if kind == "entry"],
                ),
            )
        )
    return cases


PROMPT_TARGET = message_target("prompt-1", user_message("fix the bug"))
ASSISTANT_TOOL_TARGET = message_target(
    "assistant-tools", assistant_message([ToolCall(id="call-1", name="tool-1", arguments={})], "toolUse")
)
TOOL_RESULT_TARGET = message_target("tool-result-1", tool_result_message())
ASSISTANT_FINAL_TARGET = message_target("assistant-final", assistant_message([TextContent(text="done")]))

VALID_PREFIX_CASES = [
    *valid_prefixes(
        "one-tool run X1-X5",
        [
            ("record", run_started(1, initial_messages=[PROMPT_TARGET])),
            ("entry", persisted_entry(PROMPT_TARGET, 2)),
            ("record", attempt(3, "run-1", "assistant", 1, "assistant-tools")),
            ("entry", persisted_entry(ASSISTANT_TOOL_TARGET, 4, "prompt-1")),
            ("record", tool_started(5)),
            ("entry", persisted_entry(TOOL_RESULT_TARGET, 6, "assistant-tools")),
            ("record", attempt(7, "run-1", "assistant", 1, "assistant-final")),
            ("entry", persisted_entry(ASSISTANT_FINAL_TARGET, 8, "tool-result-1")),
            ("record", operation_finished(9)),
        ],
    ),
    *valid_prefixes(
        "assistant retry",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-attempt-1")),
            ("record", usage_record(3, "assistant-attempt-1")),
            ("record", attempt(4, "run-1", "assistant", 2, "assistant-attempt-2")),
            ("record", usage_record(5, "assistant-attempt-2", "stop", 2)),
            (
                "entry",
                persisted_entry(message_target("assistant-attempt-2", assistant_message([TextContent(text="ok")])), 6),
            ),
        ],
    ),
    *valid_prefixes(
        "terminal assistant failure",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-error")),
            (
                "entry",
                persisted_entry(
                    message_target("assistant-error", replace(assistant_message([], "error"), error_message="failed")),
                    3,
                ),
            ),
            ("record", operation_finished(4, "run-1", "failed")),
        ],
    ),
    *valid_prefixes(
        "overflow compaction and retry",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "discarded-overflow")),
            ("record", usage_record(3, "discarded-overflow", "length")),
            ("record", attempt(4, "run-1", "compaction", 1, "overflow-compaction", "overflow")),
            ("entry", compaction_entry("overflow-compaction", 5)),
            ("record", attempt(6, "run-1", "assistant", 1, "assistant-after-compaction")),
            (
                "entry",
                persisted_entry(
                    message_target("assistant-after-compaction", assistant_message([TextContent(text="fits")])), 7
                ),
            ),
        ],
    ),
    *valid_prefixes(
        "steering acceptance and consumption",
        [
            ("record", run_started(1)),
            ("record", queue_enqueued(2)),
            ("entry", persisted_entry(message_target("queue-1", user_message("queued")), 3)),
        ],
    ),
    *valid_prefixes(
        "queue cancellation",
        [("record", run_started(1)), ("record", queue_enqueued(2)), ("record", queue_cancelled(3))],
    ),
    *valid_prefixes(
        "deferred write acceptance and application",
        [
            ("record", run_started(1)),
            ("record", write_deferred(2)),
            ("entry", persisted_entry(message_target("write-1", user_message("deferred write")), 3)),
        ],
    ),
    *valid_prefixes(
        "abort during a tool",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-tools")),
            ("entry", persisted_entry(ASSISTANT_TOOL_TARGET, 3)),
            ("record", tool_started(4)),
            ("record", abort_requested(5)),
            (
                "entry",
                persisted_entry(
                    message_target(
                        "tool-result-1",
                        replace(tool_result_message(), content=[TextContent(text="interrupted")], is_error=True),
                    ),
                    6,
                ),
            ),
        ],
    ),
    *valid_prefixes(
        "threshold auto-compaction",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "compaction", 1, "threshold-compaction", "threshold")),
            ("entry", compaction_entry("threshold-compaction", 3)),
            ("record", attempt(4, "run-1", "assistant", 1, "assistant-after-threshold")),
        ],
    ),
    *valid_prefixes(
        "manual compaction",
        [
            ("record", compaction_started(1)),
            ("record", attempt(2, "compact-1", "compaction", 1, "compaction-1", "manual")),
            ("entry", compaction_entry("compaction-1", 3)),
            ("record", operation_finished(4, "compact-1")),
        ],
    ),
    *valid_prefixes(
        "move-first navigation summary",
        [
            ("record", navigation_started(1)),
            ("record", attempt(2, "navigate-1", "branch_summary", 1, "summary-1")),
            ("entry", branch_summary_entry("summary-1", 3)),
            ("record", operation_finished(4, "navigate-1")),
        ],
    ),
    *valid_prefixes(
        "blocked tool without an intent record",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-tools")),
            ("entry", persisted_entry(ASSISTANT_TOOL_TARGET, 3)),
            (
                "entry",
                persisted_entry(
                    message_target(
                        "blocked-result",
                        replace(tool_result_message(), content=[TextContent(text="blocked")], is_error=True),
                    ),
                    4,
                ),
            ),
        ],
    ),
    *valid_prefixes(
        "idle next-run cancellation",
        [
            ("record", queue_enqueued(1, message_target("next-1", user_message("later")), "nextRun")),
            ("record", queue_cancelled(2, "next-1", None)),
        ],
    ),
    *valid_prefixes(
        "next-run enqueue after abort",
        [
            ("record", run_started(1)),
            ("record", abort_requested(2)),
            ("record", queue_enqueued(3, message_target("next-1", user_message("later")), "nextRun")),
        ],
    ),
    *valid_prefixes(
        "deferred write applied during abort reconciliation",
        [
            ("record", run_started(1)),
            ("record", write_deferred(2)),
            ("record", abort_requested(3)),
            ("entry", persisted_entry(message_target("write-1", user_message("deferred write")), 4)),
        ],
    ),
    *valid_prefixes(
        "accepted steering killed by abort",
        [("record", run_started(1)), ("record", queue_enqueued(2)), ("record", abort_requested(3))],
    ),
    *valid_prefixes(
        "compaction retry",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "compaction", 1, "threshold-compaction", "threshold")),
            ("record", attempt(3, "run-1", "compaction", 2, "threshold-compaction", "threshold")),
            ("entry", compaction_entry("threshold-compaction", 4)),
        ],
    ),
    *valid_prefixes(
        "hook-supplied manual compaction",
        [
            ("record", compaction_started(1)),
            ("entry", compaction_entry("compaction-1", 2)),
            ("record", operation_finished(3, "compact-1")),
        ],
    ),
    *valid_prefixes(
        "hook-supplied navigation summary",
        [
            ("record", navigation_started(1)),
            ("entry", branch_summary_entry("summary-1", 2)),
            ("record", operation_finished(3, "navigate-1")),
        ],
    ),
    *valid_prefixes(
        "deferred provider suspension and redemption",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-deferred")),
            ("entry", persisted_entry(message_target("assistant-deferred", assistant_message([], "deferred")), 3)),
            (
                "entry",
                persisted_entry(
                    message_target("assistant-redeemed", assistant_message([TextContent(text="ready")])), 4
                ),
            ),
        ],
    ),
    *valid_prefixes(
        "abort of a deferred provider request",
        [
            ("record", run_started(1)),
            ("record", attempt(2, "run-1", "assistant", 1, "assistant-deferred")),
            ("entry", persisted_entry(message_target("assistant-deferred", assistant_message([], "deferred")), 3)),
            ("record", abort_requested(4)),
        ],
    ),
]


@pytest.mark.parametrize(("name", "input"), VALID_PREFIX_CASES, ids=[case[0] for case in VALID_PREFIX_CASES])
def test_accepts_valid_durable_prefixes(name, input):
    assert validate_record_log(input) is None


def test_reduces_an_idle_lane_to_pending_next_run_input_and_default_configuration():
    pending = message_target("next-pending", user_message("pending"))
    cancelled = message_target("next-cancelled", user_message("cancelled"))
    consumed = message_target("next-consumed", user_message("consumed"))
    input = reduction_input(
        [
            queue_enqueued(1, pending, "nextRun"),
            queue_enqueued(2, cancelled, "nextRun"),
            queue_cancelled(3, cancelled.id, None),
            queue_enqueued(4, consumed, "nextRun"),
        ],
        [],
        entries=[persisted_entry(consumed, 5)],
        leaf_id="idle-leaf",
    )

    result = reduce_lane_state(input)
    assert result.lane_state.lane == "main"
    assert result.lane_state.leaf_id == "idle-leaf"
    assert result.lane_state.operation is None
    assert result.lane_state.pending_next_run == [pending]
    assert result.effective_configuration == DEFAULTS
    assert result.terminal_failure is None


def test_folds_persisted_configuration_over_copied_defaults_in_sequence():
    configuration_entries: list[Entry] = [
        ModelChangeEntry(
            id="model-change",
            parent_id=None,
            seq=1,
            timestamp=1,
            provider="persisted-provider",
            model_id="persisted-model",
        ),
        ThinkingLevelEntry(id="thinking-change", parent_id="model-change", seq=2, timestamp=2, thinking_level="high"),
        ActiveToolsEntry(
            id="tools-change", parent_id="thinking-change", seq=3, timestamp=3, active_tool_names=["persisted-tool"]
        ),
    ]
    input = reduction_input([], [], configuration_entries=configuration_entries)

    assert reduce_lane_state(input).effective_configuration == EffectiveLaneConfiguration(
        model=ModelRef(provider="persisted-provider", model_id="persisted-model"),
        thinking_level="high",
        active_tool_names=["persisted-tool"],
    )
    assert input.defaults == DEFAULTS


def test_applies_committed_operation_owned_configuration_after_the_anchor():
    assistant = persisted_entry(
        message_target(
            "assistant-config",
            replace(
                assistant_message([TextContent(text="response")]),
                provider="response-provider",
                model="response-model",
            ),
        ),
        2,
    )
    tools = ActiveToolsEntry(
        id="operation-tools", parent_id=assistant.id, seq=3, timestamp=3, active_tool_names=["operation-tool"]
    )
    result = reduce_lane_state(reduction_input([run_started(1)], [assistant, tools]))

    assert result.effective_configuration == EffectiveLaneConfiguration(
        model=ModelRef(provider="response-provider", model_id="response-model"),
        thinking_level="off",
        active_tool_names=["operation-tool"],
    )


def test_keeps_captured_next_run_input_with_the_open_run_instead_of_pending_next_run():
    captured = message_target("next-captured", user_message("captured"))
    later = message_target("next-later", user_message("later"))
    start = run_started(2, initial_messages=[captured])

    result = reduce_lane_state(
        reduction_input([queue_enqueued(1, captured, "nextRun"), start, queue_enqueued(3, later, "nextRun")])
    )

    assert result.lane_state.pending_next_run == [later]
    assert result.lane_state.operation is not None
    assert result.lane_state.operation.missing_initial_messages == [captured]


def test_derives_missing_input_queues_deferred_writes_and_the_unfinished_attempt():
    missing_prompt = message_target("prompt-missing", user_message("missing"))
    committed_prompt = message_target("prompt-committed", user_message("committed"))
    steer = message_target("steer-pending", user_message("steer"))
    consumed_follow_up = message_target("follow-consumed", user_message("follow"))
    next_run = message_target("next-run", user_message("next"))
    pending_write = message_target("write-pending", user_message("write"))
    applied_write = message_target("write-applied", user_message("applied"))
    start = run_started(1, initial_messages=[missing_prompt, committed_prompt])
    committed_prompt_entry = persisted_entry(committed_prompt, 2)
    consumed_follow_up_entry = persisted_entry(consumed_follow_up, 6, committed_prompt.id)
    applied_write_entry = persisted_entry(applied_write, 9, consumed_follow_up.id)
    input = reduction_input(
        [
            start,
            queue_enqueued(3, steer),
            queue_enqueued(4, consumed_follow_up, "followUp"),
            queue_enqueued(5, next_run, "nextRun"),
            write_deferred(7, pending_write),
            write_deferred(8, applied_write),
            attempt(10, start.id, "assistant", 1, "assistant-pending"),
        ],
        [committed_prompt_entry, consumed_follow_up_entry, applied_write_entry],
    )

    result = reduce_lane_state(input)
    assert result.lane_state.pending_next_run == [next_run]
    operation = result.lane_state.operation
    assert operation is not None
    assert operation.id == start.id
    assert operation.kind == "run"
    assert operation.aborting is False
    assert operation.missing_initial_messages == [missing_prompt]
    assert operation.pending_steer == [steer]
    assert operation.pending_follow_up == []
    assert operation.pending_writes == [pending_write]
    assert operation.step == StepState(kind="assistant", attempts=1, result_entry_id="assistant-pending")
    assert operation.newest_own is not None
    assert (operation.newest_own.entry_id, operation.newest_own.type, operation.newest_own.role) == (
        applied_write.id,
        "message",
        "user",
    )


def test_kills_steer_and_follow_up_queues_on_abort_while_preserving_writes_and_next_run_input():
    steer = message_target("steer-aborted", user_message("steer"))
    follow_up = message_target("follow-aborted", user_message("follow"))
    next_run = message_target("next-after-abort", user_message("next"))
    pending_write = message_target("write-after-abort", user_message("write"))
    input = reduction_input(
        [
            run_started(1),
            queue_enqueued(2, steer),
            queue_enqueued(3, follow_up, "followUp"),
            queue_enqueued(4, next_run, "nextRun"),
            write_deferred(5, pending_write),
            abort_requested(6),
        ]
    )

    result = reduce_lane_state(input)
    assert result.lane_state.pending_next_run == [next_run]
    operation = result.lane_state.operation
    assert operation is not None
    assert operation.aborting is True
    assert operation.pending_steer == []
    assert operation.pending_follow_up == []
    assert operation.pending_writes == [pending_write]


@pytest.mark.parametrize(
    ("name", "record", "expected"),
    [
        (
            "assistant",
            attempt(2, "run-1", "assistant", 1, "result"),
            StepState(kind="assistant", attempts=1, result_entry_id="result"),
        ),
        (
            "compaction",
            attempt(2, "run-1", "compaction", 1, "result", "overflow"),
            StepState(kind="compaction", attempts=1, result_entry_id="result", compaction_reason="overflow"),
        ),
        (
            "branch summary",
            attempt(2, "run-1", "branch_summary", 1, "result"),
            StepState(kind="branch_summary", attempts=1, result_entry_id="result"),
        ),
    ],
    ids=["assistant", "compaction", "branch summary"],
)
def test_reduces_an_unfinished_step(name, record, expected):
    result = reduce_lane_state(reduction_input([run_started(1), record]))
    assert result.lane_state.operation is not None
    assert result.lane_state.operation.step == expected


def test_closes_the_newest_attempt_only_when_its_provisioned_result_exists():
    target = message_target("result", assistant_message([TextContent(text="done")]))
    result = reduce_lane_state(
        reduction_input([run_started(1), attempt(2, "run-1", "assistant", 1, target.id)], [persisted_entry(target, 3)])
    )
    assert result.lane_state.operation is not None
    assert result.lane_state.operation.step is None


def test_ignores_unfulfilled_result_ids_from_earlier_attempts():
    target = message_target("attempt-2-result", assistant_message([TextContent(text="done")]))
    result = reduce_lane_state(
        reduction_input(
            [
                run_started(1),
                attempt(2, "run-1", "assistant", 1, "attempt-1-result"),
                attempt(3, "run-1", "assistant", 2, target.id),
            ],
            [persisted_entry(target, 4)],
        )
    )
    assert result.lane_state.operation is not None
    assert result.lane_state.operation.step is None


@pytest.mark.parametrize(
    ("name", "records", "result_entry"),
    [
        ("X1", [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-tools")], None),
        ("X3", [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-tools"), tool_started(4)], None),
        (
            "X5",
            [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-tools"), tool_started(4)],
            replace(persisted_entry(TOOL_RESULT_TARGET, 5, ASSISTANT_TOOLS_ENTRY.id), terminate=True),
        ),
    ],
    ids=["X1", "X3", "X5"],
)
def test_reduces_tool_batch_state(name, records, result_entry):
    own_entries = [ASSISTANT_TOOLS_ENTRY, result_entry] if result_entry is not None else [ASSISTANT_TOOLS_ENTRY]
    reduction = reduce_lane_state(reduction_input(records, own_entries))
    operation = reduction.lane_state.operation
    assert operation is not None and operation.tool_batch is not None
    batch = operation.tool_batch
    call = batch.calls[0]

    assert batch.assistant_entry_id == ASSISTANT_TOOLS_ENTRY.id
    assert batch.truncated is False
    assert batch.unresolved is (result_entry is None)
    assert call.tool_index == 0
    assert (call.tool_call.id, call.tool_call.name) == ("call-1", "tool-1")
    assert call.result_exists is (result_entry is not None)
    if result_entry is not None:
        assert call.terminate is True
    assert (call.started is not None) == any(record.type == "tool_started" for record in records)


def test_does_not_resolve_a_tool_batch_from_a_deferred_write_tool_result():
    assistant = persisted_entry(ASSISTANT_TOOL_TARGET, 3)
    written_result = message_target("written-tool-result", tool_result_message())
    result = reduce_lane_state(
        reduction_input(
            [run_started(1), attempt(2, "run-1", "assistant", 1, assistant.id), write_deferred(4, written_result)],
            [assistant, persisted_entry(written_result, 5, assistant.id)],
        )
    )

    operation = result.lane_state.operation
    assert operation is not None and operation.tool_batch is not None
    assert operation.tool_batch.calls[0].result_exists is False
    assert operation.tool_batch.unresolved is True


def test_matches_blocked_results_without_tool_start_records_and_preserves_source_order():
    assistant = persisted_entry(
        message_target(
            "assistant-two-tools",
            assistant_message(
                [
                    ToolCall(id="call-1", name="tool-1", arguments={}),
                    ToolCall(id="call-2", name="tool-2", arguments={}),
                ],
                "toolUse",
            ),
        ),
        3,
    )
    blocked = persisted_entry(
        message_target(
            "blocked-result",
            replace(tool_result_message("call-1", "tool-1"), content=[TextContent(text="blocked")], is_error=True),
        ),
        4,
        assistant.id,
    )
    second_start = tool_started(
        5,
        assistant_entry_id=assistant.id,
        tool_index=1,
        tool_call_id="call-2",
        tool_name="tool-2",
        result_entry_id="call-2-result",
    )
    result = reduce_lane_state(
        reduction_input(
            [run_started(1), attempt(2, "run-1", "assistant", 1, assistant.id), second_start], [assistant, blocked]
        )
    )

    operation = result.lane_state.operation
    assert operation is not None and operation.tool_batch is not None
    calls = operation.tool_batch.calls
    assert (calls[0].tool_index, calls[0].tool_call.id, calls[0].result_exists) == (0, "call-1", True)
    assert (calls[1].tool_index, calls[1].started, calls[1].result_exists) == (1, second_start, False)


def test_marks_a_length_stopped_tool_batch_as_truncated_without_resolving_it():
    truncated = persisted_entry(
        message_target(
            "assistant-truncated", assistant_message([ToolCall(id="call-1", name="tool-1", arguments={})], "length")
        ),
        3,
    )
    result = reduce_lane_state(
        reduction_input([run_started(1), attempt(2, "run-1", "assistant", 1, truncated.id)], [truncated])
    )
    operation = result.lane_state.operation
    assert operation is not None and operation.tool_batch is not None
    assert operation.tool_batch.truncated is True
    assert operation.tool_batch.unresolved is True


def test_detects_an_unredeemed_deferred_handle_only_at_the_operation_tail():
    deferred_message = assistant_message([], "deferred")
    deferred_entry = persisted_entry(message_target("assistant-deferred", deferred_message), 3)
    pending = reduce_lane_state(
        reduction_input([run_started(1), attempt(2, "run-1", "assistant", 1, deferred_entry.id)], [deferred_entry])
    )
    assert pending.lane_state.operation is not None
    assert pending.lane_state.operation.deferred == deferred_message.deferred

    successor = persisted_entry(
        message_target("assistant-ready", assistant_message([TextContent(text="ready")])), 4, deferred_entry.id
    )
    redeemed = reduce_lane_state(
        reduction_input(
            [run_started(1), attempt(2, "run-1", "assistant", 1, deferred_entry.id)], [deferred_entry, successor]
        )
    )
    assert redeemed.lane_state.operation is not None
    assert redeemed.lane_state.operation.deferred is None


@pytest.mark.parametrize(
    ("name", "records", "own_entries", "expected_source"),
    [
        (
            "step",
            [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-error")],
            [
                persisted_entry(
                    message_target("assistant-error", replace(assistant_message([], "error"), error_message="failed")),
                    3,
                )
            ],
            "step",
        ),
        (
            "deferred fetch",
            [run_started(1), attempt(2, "run-1", "assistant", 1, "assistant-deferred")],
            [
                persisted_entry(message_target("assistant-deferred", assistant_message([], "deferred")), 3),
                persisted_entry(
                    message_target("deferred-error", replace(assistant_message([], "error"), error_message="expired")),
                    4,
                    "assistant-deferred",
                ),
            ],
            "deferred_fetch",
        ),
        (
            "deferred fetch usage record",
            [
                run_started(1),
                UsageRecord(
                    id="deferred-usage",
                    lane="main",
                    seq=3,
                    timestamp=3,
                    cause="deferred_fetch",
                    run_id="run-1",
                    entry_id="deferred-error",
                    attempt=1,
                    stop_reason="error",
                    usage=USAGE,
                ),
            ],
            [
                persisted_entry(
                    message_target("deferred-error", replace(assistant_message([], "error"), error_message="expired")),
                    2,
                )
            ],
            "deferred_fetch",
        ),
    ],
    ids=["step", "deferred fetch", "deferred fetch usage record"],
)
def test_derives_terminal_failure_provenance(name, records, own_entries, expected_source):
    result = reduce_lane_state(reduction_input(records, own_entries))
    assert result.terminal_failure is not None
    assert result.terminal_failure.source == expected_source


def test_does_not_classify_an_error_shaped_deferred_write_as_terminal_failure():
    target = message_target("written-error", replace(assistant_message([], "error"), error_message="note"))
    entry = persisted_entry(target, 3)
    result = reduce_lane_state(reduction_input([run_started(1), write_deferred(2, target)], [entry]))
    assert result.terminal_failure is None


@pytest.mark.parametrize(
    ("name", "records", "entries", "expected"),
    [
        ("manual compaction result", [compaction_started(1)], [], OperationTargets(result=False)),
        (
            "completed manual compaction result",
            [compaction_started(1)],
            [compaction_entry("compaction-1", 2)],
            OperationTargets(result=True),
        ),
        ("missing navigation summary", [navigation_started(1)], [], OperationTargets(summary=False)),
        (
            "navigation summary",
            [navigation_started(1)],
            [branch_summary_entry("summary-1", 2)],
            OperationTargets(summary=True),
        ),
    ],
    ids=[
        "manual compaction result",
        "completed manual compaction result",
        "missing navigation summary",
        "navigation summary",
    ],
)
def test_derives_structural_target_state(name, records, entries, expected):
    result = reduce_lane_state(reduction_input(records, entries))
    assert result.lane_state.operation is not None
    assert result.lane_state.operation.targets == expected


def test_resets_the_overflow_guard_only_after_newer_conversational_input_is_consumed():
    initial = message_target("initial", user_message("initial"))
    steer = message_target("steer", user_message("steer"))
    start = run_started(1, initial_messages=[initial])
    initial_entry = persisted_entry(initial, 2)
    records: list[LaneRecord] = [
        start,
        attempt(3, start.id, "compaction", 1, "overflow-summary", "overflow"),
        queue_enqueued(5, steer),
    ]

    used = reduce_lane_state(reduction_input(records, [initial_entry]))
    assert used.lane_state.operation is not None
    assert used.lane_state.operation.overflow_recovery_used is True

    reset = reduce_lane_state(reduction_input(records, [initial_entry, persisted_entry(steer, 6, initial.id)]))
    assert reset.lane_state.operation is not None
    assert reset.lane_state.operation.overflow_recovery_used is False


def test_is_deterministic_and_does_not_mutate_or_alias_its_inputs():
    pending = message_target("next", user_message("next"))
    input = reduction_input([queue_enqueued(1, pending, "nextRun")])
    before = copy.deepcopy(input)
    first = reduce_lane_state(input)
    second = reduce_lane_state(input)

    assert first == second
    assert input == before
    first.lane_state.pending_next_run[0].id = "mutated-output"
    record = input.records[0]
    assert record.type == "queue_enqueued"
    assert record.target.id == "next"
