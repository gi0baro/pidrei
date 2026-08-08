"""Session v4 vocabulary: entries, lane records, queries, storage contracts.

Port of pi `harness/session/types.ts`. Entries and lane records share one
monotonic sequence; lanes are named pointers into the entry tree. "Provisioned"
entries/records (pi's `ProvisionedEntry`/`NewRecord` Omit-types) are modeled as
the same dataclasses with `seq`/`parent_id`/`timestamp` left at their defaults —
storage assigns them on append and returns the completed value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol


if TYPE_CHECKING:
    from .session import Session

type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

# pi: Exclude<StopReason, "pending"> | "deferred"
type SessionStopReason = Literal["stop", "length", "toolUse", "error", "aborted", "deferred"]


class IdGenerator(Protocol):
    def next(self) -> str: ...


# --- entries --------------------------------------------------------------------
# Base fields: `type` tag, `id`, `seq`, `parent_id`, Unix-millisecond `timestamp`.
# `seq`/`parent_id`/`timestamp` are storage-assigned; their defaults mark a
# provisioned (not yet appended) value.


@dataclass(slots=True, kw_only=True)
class MessageEntry:
    id: str
    message: Any  # AgentMessage
    terminate: Literal[True] | None = None
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["message"] = "message"


@dataclass(slots=True, kw_only=True)
class ModelChangeEntry:
    id: str
    provider: str
    model_id: str
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["model_change"] = "model_change"


@dataclass(slots=True, kw_only=True)
class ThinkingLevelEntry:
    id: str
    thinking_level: str
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["thinking_level_change"] = "thinking_level_change"


@dataclass(slots=True, kw_only=True)
class ActiveToolsEntry:
    id: str
    active_tool_names: list[str]
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["active_tools_change"] = "active_tools_change"


@dataclass(slots=True, kw_only=True)
class CompactionEntry:
    id: str
    summary: str
    retained_tail: list[Any]  # AgentMessage[]
    tokens_before: int
    details: Any = None
    usage: Any = None  # Usage | None
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["compaction"] = "compaction"


@dataclass(slots=True, kw_only=True)
class BranchSummaryEntry:
    id: str
    from_id: str
    summary: str
    details: Any = None
    usage: Any = None  # Usage | None
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["branch_summary"] = "branch_summary"


@dataclass(slots=True, kw_only=True)
class CustomEntry:
    id: str
    custom_type: str
    data: Any = None
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["custom"] = "custom"


type Entry = (
    MessageEntry
    | ModelChangeEntry
    | ThinkingLevelEntry
    | ActiveToolsEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
)


# --- lane records ---------------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class RunIntent:
    # Normalized caller input before before_run; kept for suspended operations and before_resume.
    original_prompt: list[Any]  # AgentMessage[]
    # Captured nextRun items, then the prompt, then before_run injections.
    initial_messages: list[Entry]  # provisioned entries
    system_prompt_override: str | None = None
    resume_data: dict[str, JsonValue] | None = None
    kind: Literal["run"] = "run"


@dataclass(slots=True, kw_only=True)
class CompactionIntent:
    result_entry_id: str
    custom_instructions: str | None = None
    kind: Literal["compaction"] = "compaction"


@dataclass(slots=True, kw_only=True)
class NavigationIntent:
    target_id: str | None
    summarize: bool
    custom_instructions: str | None = None
    label: str | None = None
    summary_entry_id: str | None = None
    kind: Literal["navigation"] = "navigation"


type OperationIntent = RunIntent | CompactionIntent | NavigationIntent

type OperationKind = Literal["run", "compaction", "navigation"]


@dataclass(slots=True, kw_only=True)
class OperationStartedRecord:
    id: str
    lane: str
    source_leaf_id: str | None
    intent: OperationIntent
    seq: int = 0
    timestamp: int = 0
    type: Literal["operation_started"] = "operation_started"


@dataclass(slots=True, kw_only=True)
class AbortRequestedRecord:
    id: str
    lane: str
    run_id: str
    seq: int = 0
    timestamp: int = 0
    type: Literal["abort_requested"] = "abort_requested"


@dataclass(slots=True, kw_only=True)
class OperationFinishedRecord:
    id: str
    lane: str
    run_id: str
    outcome: Literal["completed", "aborted", "failed", "declined"]
    error: dict[str, str] | None = None  # { code, message }
    seq: int = 0
    timestamp: int = 0
    type: Literal["operation_finished"] = "operation_finished"


type CompactionReason = Literal["manual", "threshold", "overflow"]


@dataclass(slots=True, kw_only=True)
class StepAttemptRecord:
    id: str
    lane: str
    run_id: str
    step: Literal["assistant", "branch_summary", "compaction"]
    attempt: int
    result_entry_id: str
    # Persists why compaction summary generation started so recovery resumes the
    # same work. Set exactly when step == "compaction".
    compaction_reason: CompactionReason | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["step_attempt"] = "step_attempt"


@dataclass(slots=True, kw_only=True)
class ToolStartedRecord:
    id: str
    lane: str
    run_id: str
    assistant_entry_id: str
    tool_index: int
    tool_call_id: str
    tool_name: str
    effective_args: dict[str, Any]
    result_entry_id: str
    replay: Literal["never", "safe"]
    seq: int = 0
    timestamp: int = 0
    type: Literal["tool_started"] = "tool_started"


@dataclass(slots=True, kw_only=True)
class QueueEnqueuedRecord:
    id: str
    lane: str
    queue: Literal["steer", "followUp", "nextRun"]
    target: Entry  # provisioned entry
    # Absent exactly when queue == "nextRun".
    run_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["queue_enqueued"] = "queue_enqueued"


@dataclass(slots=True, kw_only=True)
class QueueCancelledRecord:
    id: str
    lane: str
    entry_id: str
    run_id: str | None = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["queue_cancelled"] = "queue_cancelled"


@dataclass(slots=True, kw_only=True)
class WriteDeferredRecord:
    id: str
    lane: str
    run_id: str
    target: Entry  # provisioned entry
    seq: int = 0
    timestamp: int = 0
    type: Literal["write_deferred"] = "write_deferred"


type UsageCause = Literal["assistant", "compaction", "branch_summary", "deferred_fetch", "tool", "hook", "adjustment"]


@dataclass(slots=True, kw_only=True)
class UsageRecord:
    id: str
    lane: str
    cause: UsageCause
    usage: Any  # Usage
    run_id: str | None = None
    entry_id: str | None = None
    # Set for step causes ("assistant", "compaction", "branch_summary", "deferred_fetch").
    attempt: int | None = None
    stop_reason: SessionStopReason | None = None
    # Set for cause "tool".
    tool_call_id: str | None = None
    # Set for cause "adjustment".
    details: JsonValue = None
    seq: int = 0
    timestamp: int = 0
    type: Literal["usage"] = "usage"


type LaneRecord = (
    OperationStartedRecord
    | AbortRequestedRecord
    | OperationFinishedRecord
    | StepAttemptRecord
    | ToolStartedRecord
    | QueueEnqueuedRecord
    | QueueCancelledRecord
    | WriteDeferredRecord
    | UsageRecord
)


# --- queries --------------------------------------------------------------------

type EntryOrder = Literal["newestFirst", "oldestFirst"]


@dataclass(slots=True)
class EntryCursor:
    after_seq: int


@dataclass(slots=True, kw_only=True)
class EntryQuery:
    type: str | None = None
    custom_type: str | None = None
    order: EntryOrder | None = None
    limit: int | None = None
    cursor: EntryCursor | None = None


@dataclass(slots=True, kw_only=True)
class BranchQuery(EntryQuery):
    """EntryQuery plus pi's `BranchBounds`; `start` defaults to the lane leaf."""

    start: str | None = None
    stop_at_type: str | None = None
    stop_at_id: str | None = None


@dataclass(slots=True, kw_only=True)
class RecordQuery:
    lane: str | None = None
    type: str | None = None
    run_id: str | None = None
    # Valid only with type "operation_started".
    operation_kind: OperationKind | None = None
    after_seq: int | None = None
    order: EntryOrder | None = None
    limit: int | None = None


@dataclass(slots=True, kw_only=True)
class SessionMetadata:
    id: str
    created_at: int
    parent_session_id: str | None = None


@dataclass(slots=True)
class SessionStats:
    message_count: int = 0
    cached_tokens: int = 0
    uncached_tokens: int = 0
    total_tokens: int = 0
    cost_total: float = 0


@dataclass(slots=True)
class LanePointer:
    lane: str
    leaf_id: str | None


# --- log ------------------------------------------------------------------------


@dataclass(slots=True, kw_only=True)
class EntryLogItem:
    seq: int
    entry: Entry
    kind: Literal["entry"] = "entry"


@dataclass(slots=True, kw_only=True)
class RecordLogItem:
    seq: int
    record: LaneRecord
    kind: Literal["record"] = "record"


@dataclass(slots=True, kw_only=True)
class LaneLogItem:
    seq: int
    lane: str
    leaf_id: str | None
    kind: Literal["lane"] = "lane"


@dataclass(slots=True, kw_only=True)
class NameFactLogItem:
    seq: int
    name: str
    kind: Literal["fact"] = "fact"
    fact: Literal["name"] = "name"


@dataclass(slots=True, kw_only=True)
class LabelFactLogItem:
    seq: int
    target_id: str
    label: str | None
    kind: Literal["fact"] = "fact"
    fact: Literal["label"] = "label"


type LogItem = EntryLogItem | RecordLogItem | LaneLogItem | NameFactLogItem | LabelFactLogItem


@dataclass(slots=True, kw_only=True)
class LogOptions:
    after_seq: int | None = None
    limit: int | None = None


# --- storage and tree contracts -------------------------------------------------


class SessionStorage(Protocol):
    async def get_metadata(self) -> SessionMetadata: ...

    # Lanes
    async def get_lanes(self) -> list[LanePointer]: ...
    async def create_lane(self, lane: str, at: str | None) -> None: ...
    async def move_lane(self, lane: str, to: str | None) -> None: ...

    # Entries and Records
    async def append_entry[TEntry: Entry](self, entry: TEntry, lane: str) -> TEntry: ...
    async def append_record[TRecord: LaneRecord](self, record: TRecord) -> TRecord: ...

    # Reads
    async def get_entry(self, id: str) -> Entry | None: ...
    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...
    async def find_entries_on_branch(self, query: BranchQuery) -> list[Entry]:
        """`start` is mandatory here; defaulting to a lane's leaf is view sugar."""
        ...

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]: ...
    async def find_open_operations(self, lane: str, limit: int | None = None) -> list[OperationStartedRecord]:
        """Returns unfinished operation starts newest first. Recovery uses `limit=2`:
        zero results mean the lane is idle, one means it is suspended, and two
        mean at least two operations are open, which is corruption. Further
        results provide no additional recovery state.
        """
        ...

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]: ...

    # Global facts
    async def get_name(self) -> str | None: ...
    async def set_name(self, name: str) -> None: ...
    async def get_label(self, id: str) -> str | None: ...
    async def set_label(self, id: str, label: str | None) -> None: ...
    async def get_stats(self) -> SessionStats: ...


class SessionTree(Protocol):
    async def get_leaf_id(self) -> str | None: ...
    async def get_entry(self, id: str) -> Entry | None: ...
    async def get_stats(self) -> SessionStats: ...
    async def get_name(self) -> str | None: ...
    async def set_name(self, name: str) -> None: ...
    async def get_label(self, target_id: str) -> str | None: ...
    async def set_label(self, target_id: str, label: str | None) -> None: ...
    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...
    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None: ...
    async def find_entries_on_branch(self, query: BranchQuery | None = None) -> list[Entry]: ...
    async def find_entry_on_branch(self, query: BranchQuery | None = None) -> Entry | None: ...
    async def append_message(self, message: Any) -> str: ...
    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str: ...


@dataclass(slots=True, kw_only=True)
class SessionCreateOptions:
    id: str | None = None
    parent_session_id: str | None = None


@dataclass(slots=True, kw_only=True)
class ForkOptions:
    scope: Literal["branch", "tree"] = "branch"
    # Branch scope only; ignored for tree forks.
    entry_id: str | None = None
    position: Literal["before", "at"] | None = None


class SessionRepo(Protocol):
    async def create(self, options: SessionCreateOptions | None = None) -> Session: ...
    async def open(self, metadata: SessionMetadata) -> Session:
        """Opens the session for writing and acquires any backend writer claim."""
        ...

    async def list(self, options: Any = None) -> list[SessionMetadata]:
        """Lists session metadata without opening sessions or acquiring writer claims."""
        ...

    async def delete(self, metadata: SessionMetadata) -> None: ...
    async def fork(
        self, source: SessionMetadata, options: ForkOptions, create: SessionCreateOptions | None = None
    ) -> Session: ...


type SessionErrorCode = Literal[
    "not_found",
    "already_exists",
    "invalid_entry",
    "invalid_payload",
    "invalid_lane",
    "invalid_query",
    "invalid_fork_target",
    "storage",
]


class SessionError(Exception):
    def __init__(self, code: SessionErrorCode, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if cause is not None:
            self.__cause__ = cause


__all__ = [
    "AbortRequestedRecord",
    "ActiveToolsEntry",
    "BranchQuery",
    "BranchSummaryEntry",
    "CompactionEntry",
    "CompactionIntent",
    "CompactionReason",
    "CustomEntry",
    "Entry",
    "EntryCursor",
    "EntryLogItem",
    "EntryOrder",
    "EntryQuery",
    "ForkOptions",
    "IdGenerator",
    "JsonValue",
    "LabelFactLogItem",
    "LaneLogItem",
    "LanePointer",
    "LaneRecord",
    "LogItem",
    "LogOptions",
    "MessageEntry",
    "ModelChangeEntry",
    "NameFactLogItem",
    "NavigationIntent",
    "OperationFinishedRecord",
    "OperationIntent",
    "OperationKind",
    "OperationStartedRecord",
    "QueueCancelledRecord",
    "QueueEnqueuedRecord",
    "RecordLogItem",
    "RecordQuery",
    "RunIntent",
    "SessionCreateOptions",
    "SessionError",
    "SessionErrorCode",
    "SessionMetadata",
    "SessionRepo",
    "SessionStats",
    "SessionStopReason",
    "SessionStorage",
    "SessionTree",
    "StepAttemptRecord",
    "ThinkingLevelEntry",
    "ToolStartedRecord",
    "UsageCause",
    "UsageRecord",
    "WriteDeferredRecord",
]
