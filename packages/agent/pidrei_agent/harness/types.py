"""Harness-level types (port of pi `agent/src/harness/types.ts`).

This module currently carries the Result machinery, error types, and the
execution-environment capability surface needed by the built-in tools; the
harness/session/compaction types land together with their module ports.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

from pidrei_ai.utils.cancel import CancelToken

from ..types import AgentToolResult, AgentToolUpdateCallback, PrepareArguments, ToolExecutionMode


# --- Result -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Ok[TValue]:
    value: TValue
    ok: ClassVar[Literal[True]] = True


@dataclass(frozen=True, slots=True)
class Err[TError]:
    error: TError
    ok: ClassVar[Literal[False]] = False


# Result of a fallible operation. Expected failures are returned as `Err`
# instead of raised.
type Result[TValue, TError] = Ok[TValue] | Err[TError]


def ok[TValue](value: TValue) -> Ok[TValue]:
    """Create a successful `Result`."""
    return Ok(value)


def err[TError](error: TError) -> Err[TError]:
    """Create a failed `Result`."""
    return Err(error)


def get_or_throw[TValue](result: Result[TValue, Any]) -> TValue:
    """Return the success value or raise the failure error.

    Intended for tests and explicit adapter boundaries.
    """
    if not result.ok:
        raise result.error
    return result.value


def get_or_none[TValue](result: Result[TValue, Any]) -> TValue | None:
    """Return the success value or `None` (pi: `getOrUndefined`)."""
    return result.value if result.ok else None


def to_error(error: Any) -> Exception:
    """Normalize unknown raised values into Exception instances (pi: `toError`)."""
    if isinstance(error, Exception):
        return error
    if isinstance(error, str):
        return Exception(error)
    try:
        return Exception(json.dumps(error))
    except TypeError, ValueError:
        return Exception(str(error))


# --- errors -------------------------------------------------------------------

# Stable, backend-independent file error codes returned by file operations.
type FileErrorCode = Literal[
    "aborted",
    "not_found",
    "permission_denied",
    "not_directory",
    "is_directory",
    "invalid",
    "not_supported",
    "unknown",
]


class FileError(Exception):
    """Error returned by `FileSystem` file operations."""

    def __init__(self, code: FileErrorCode, message: str, path: str | None = None, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.path = path
        self.message = message
        if cause is not None:
            self.__cause__ = cause


# Stable, backend-independent execution error codes returned by `Shell.exec`.
type ExecutionErrorCode = Literal[
    "aborted",
    "timeout",
    "shell_unavailable",
    "spawn_error",
    "callback_error",
    "unknown",
]


class ExecutionError(Exception):
    """Error returned by `Shell.exec`."""

    def __init__(self, code: ExecutionErrorCode, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if cause is not None:
            self.__cause__ = cause


# --- filesystem ---------------------------------------------------------------

# Kind of filesystem object as addressed by a `FileSystem`. Symlinks are not
# followed automatically.
type FileKind = Literal["file", "directory", "symlink"]


@dataclass(slots=True)
class FileInfo:
    """Metadata for one filesystem object in a `FileSystem`."""

    # Basename of `path`.
    name: str
    # Absolute, syntactically normalized addressed path. Symlinks are not followed.
    path: str
    # Object kind. Symlink targets are not followed; use `canonical_path` explicitly.
    kind: FileKind
    # Size in bytes for the addressed filesystem object.
    size: int
    # Modification time as milliseconds since Unix epoch.
    mtime_ms: float


class FileSystem(Protocol):
    """Filesystem capability used by the harness.

    Paths passed to methods may be absolute or relative to `cwd`. Operation
    methods must never raise: all filesystem failures, including unexpected
    backend failures, must be encoded in the returned `Result`.
    """

    cwd: str

    async def absolute_path(self, path: str, cancel: CancelToken | None = None) -> Result[str, FileError]: ...
    async def join_path(self, parts: list[str], cancel: CancelToken | None = None) -> Result[str, FileError]: ...
    async def read_text_file(self, path: str, cancel: CancelToken | None = None) -> Result[str, FileError]: ...
    async def read_text_lines(
        self, path: str, max_lines: int | None = None, cancel: CancelToken | None = None
    ) -> Result[list[str], FileError]: ...
    async def read_binary_file(self, path: str, cancel: CancelToken | None = None) -> Result[bytes, FileError]: ...
    async def write_file(
        self, path: str, content: str | bytes, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...
    async def append_file(
        self, path: str, content: str | bytes, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...
    async def file_info(self, path: str, cancel: CancelToken | None = None) -> Result[FileInfo, FileError]: ...
    async def list_dir(self, path: str, cancel: CancelToken | None = None) -> Result[list[FileInfo], FileError]: ...
    async def canonical_path(self, path: str, cancel: CancelToken | None = None) -> Result[str, FileError]: ...
    async def exists(self, path: str, cancel: CancelToken | None = None) -> Result[bool, FileError]: ...
    async def create_dir(
        self, path: str, recursive: bool = True, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...
    async def remove(
        self, path: str, recursive: bool = False, force: bool = False, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...
    async def create_temp_dir(
        self, prefix: str = "tmp-", cancel: CancelToken | None = None
    ) -> Result[str, FileError]: ...
    async def create_temp_file(
        self, prefix: str = "", suffix: str = "", cancel: CancelToken | None = None
    ) -> Result[str, FileError]: ...
    async def cleanup(self) -> None:
        """Release filesystem resources. Best-effort; must not raise."""
        ...


@dataclass(slots=True)
class ShellExecResult:
    stdout: str
    stderr: str
    exit_code: int


@dataclass(slots=True)
class ShellExecOptions:
    """Options for `Shell.exec`."""

    # Working directory for the command. Relative paths resolve against the
    # environment `cwd`, which is also the default.
    cwd: str | None = None
    # Environment variables for the command. Values override inherited defaults
    # when `inherit_env` is true.
    env: dict[str, str] = field(default_factory=dict)
    # Whether to inherit the execution environment's default variables.
    inherit_env: bool = True
    # Timeout in seconds. Defaults to no timeout.
    timeout: float | None = None
    # Cancel token used to terminate the command (pi: `abortSignal`).
    cancel: CancelToken | None = None
    # Called with stdout chunks as they are produced.
    on_stdout: Callable[[str], None] | None = None
    # Called with stderr chunks as they are produced.
    on_stderr: Callable[[str], None] | None = None


class Shell(Protocol):
    """Shell execution capability used by the harness."""

    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[ShellExecResult, ExecutionError]: ...
    async def cleanup(self) -> None:
        """Release shell resources. Best-effort; must not raise."""
        ...


class ExecutionEnv(FileSystem, Shell, Protocol):
    """Filesystem and process execution environment used by the harness."""


# --- harness tools ------------------------------------------------------------


class AgentHarnessTool[TContext, TDetails]:
    """Tool definition executed by an `AgentHarness` with an application-defined context.

    pi builds tools as object literals typed `AgentHarnessTool`; the Python
    port uses a base class whose factories (`create_bash_tool`, ...) return
    configured instances.
    """

    name: str
    label: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (pi: TypeBox TSchema)
    # pi-ai `Tool` field carried through (adapters read it when converting tools).
    constrained_sampling: Any = None
    execution_mode: ToolExecutionMode | None = None
    prepare_arguments: PrepareArguments | None = None

    async def execute(
        self,
        tool_call_id: str,
        params: Any,
        cancel: CancelToken | None,
        on_update: AgentToolUpdateCallback[TDetails] | None,
        context: TContext,
    ) -> AgentToolResult[TDetails]:
        """Execute the tool call with the context resolved for the current turn snapshot."""
        raise NotImplementedError


# --- session / harness errors ---------------------------------------------------

# Stable compaction error codes returned by compaction helpers.
type CompactionErrorCode = Literal["aborted", "summarization_failed", "invalid_session", "unknown"]


class CompactionError(Exception):
    """Error returned by compaction helpers."""

    def __init__(self, code: CompactionErrorCode, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if cause is not None:
            self.__cause__ = cause


# Stable branch-summary error codes returned by branch summarization helpers.
type BranchSummaryErrorCode = Literal["aborted", "summarization_failed", "invalid_session"]


class BranchSummaryError(Exception):
    """Error returned by branch summarization helpers."""

    def __init__(self, code: BranchSummaryErrorCode, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if cause is not None:
            self.__cause__ = cause


type SessionErrorCode = Literal[
    "not_found",
    "invalid_session",
    "invalid_entry",
    "invalid_fork_target",
    "storage",
    "unknown",
]


class SessionError(Exception):
    """Error raised by session storage, repositories, and session tree operations."""

    def __init__(self, code: SessionErrorCode, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if cause is not None:
            self.__cause__ = cause


type AgentHarnessErrorCode = Literal[
    "busy",
    "invalid_state",
    "invalid_argument",
    "session",
    "hook",
    "auth",
    "compaction",
    "branch_summary",
    "unknown",
]


class AgentHarnessError(Exception):
    """Public AgentHarness failure with a stable top-level classification."""

    def __init__(self, code: AgentHarnessErrorCode, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if cause is not None:
            self.__cause__ = cause


# --- session tree entries -------------------------------------------------------
# All entries carry pi's base fields: `type` tag, `id`, `parent_id`, ISO `timestamp`.


@dataclass(slots=True, kw_only=True)
class MessageEntry:
    id: str
    parent_id: str | None
    timestamp: str
    message: Any  # AgentMessage
    type: Literal["message"] = "message"


@dataclass(slots=True, kw_only=True)
class ThinkingLevelChangeEntry:
    id: str
    parent_id: str | None
    timestamp: str
    thinking_level: str
    type: Literal["thinking_level_change"] = "thinking_level_change"


@dataclass(slots=True, kw_only=True)
class ModelChangeEntry:
    id: str
    parent_id: str | None
    timestamp: str
    provider: str
    model_id: str
    type: Literal["model_change"] = "model_change"


@dataclass(slots=True, kw_only=True)
class ActiveToolsChangeEntry:
    id: str
    parent_id: str | None
    timestamp: str
    active_tool_names: list[str]
    type: Literal["active_tools_change"] = "active_tools_change"


@dataclass(slots=True, kw_only=True)
class CompactionEntry:
    id: str
    parent_id: str | None
    timestamp: str
    summary: str
    tokens_before: int
    first_kept_entry_id: str | None = None
    retained_tail: list[Any] | None = None  # AgentMessage[]
    details: Any = None
    usage: Any = None  # Usage | None
    from_hook: bool | None = None
    type: Literal["compaction"] = "compaction"


@dataclass(slots=True, kw_only=True)
class BranchSummaryEntry:
    id: str
    parent_id: str | None
    timestamp: str
    from_id: str
    summary: str
    details: Any = None
    usage: Any = None  # Usage | None
    from_hook: bool | None = None
    type: Literal["branch_summary"] = "branch_summary"


@dataclass(slots=True, kw_only=True)
class CustomEntry:
    id: str
    parent_id: str | None
    timestamp: str
    custom_type: str
    data: Any = None
    type: Literal["custom"] = "custom"


@dataclass(slots=True, kw_only=True)
class CustomMessageEntry:
    id: str
    parent_id: str | None
    timestamp: str
    custom_type: str
    content: Any  # str | list[TextContent | ImageContent]
    display: bool
    details: Any = None
    type: Literal["custom_message"] = "custom_message"


@dataclass(slots=True, kw_only=True)
class LabelEntry:
    id: str
    parent_id: str | None
    timestamp: str
    target_id: str
    label: str | None
    type: Literal["label"] = "label"


@dataclass(slots=True, kw_only=True)
class SessionInfoEntry:
    # Legacy name, kept for backwards compatibility.
    id: str
    parent_id: str | None
    timestamp: str
    name: str | None = None
    type: Literal["session_info"] = "session_info"


@dataclass(slots=True, kw_only=True)
class LeafEntry:
    id: str
    parent_id: str | None
    timestamp: str
    target_id: str | None
    type: Literal["leaf"] = "leaf"


type SessionTreeEntry = (
    MessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | ActiveToolsChangeEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry
    | CustomMessageEntry
    | LabelEntry
    | SessionInfoEntry
    | LeafEntry
)


@dataclass(slots=True)
class SessionModelRef:
    provider: str
    model_id: str


@dataclass(slots=True)
class SessionContext:
    messages: list[Any]  # AgentMessage[]
    thinking_level: str
    model: SessionModelRef | None
    active_tool_names: list[str] | None


@dataclass(slots=True)
class SessionStats:
    message_count: int
    cached_tokens: int
    uncached_tokens: int
    total_tokens: int
    cost_total: float


@dataclass(slots=True, kw_only=True)
class SessionMetadata:
    id: str
    created_at: str


@dataclass(slots=True)
class SessionEntryCursorOptions:
    after_entry_seq: int | None = None
    limit: int | None = None


class SessionStorage(Protocol):
    """Backend-independent session tree storage."""

    async def get_metadata(self) -> SessionMetadata: ...
    async def get_leaf_id(self) -> str | None: ...
    async def set_leaf_id(self, leaf_id: str | None) -> None:
        """Persist a leaf entry that records the active session-tree leaf."""
        ...

    async def create_entry_id(self) -> str: ...
    async def append_entry(self, entry: SessionTreeEntry) -> None: ...
    async def get_entry(self, id: str) -> SessionTreeEntry | None: ...
    async def find_entries(self, type: str) -> list[SessionTreeEntry]: ...
    async def get_label(self, id: str) -> str | None: ...
    async def get_session_name(self) -> str | None: ...
    async def get_session_stats(self) -> SessionStats: ...
    async def get_path_to_root_or_compaction(self, leaf_id: str | None) -> list[SessionTreeEntry]: ...
    async def get_entries(self, options: SessionEntryCursorOptions | None = None) -> list[SessionTreeEntry]: ...


@dataclass(slots=True)
class SessionCreateOptions:
    id: str | None = None


@dataclass(slots=True)
class SessionForkOptions:
    entry_id: str | None = None
    position: Literal["before", "at"] | None = None
    id: str | None = None


# --- resources ------------------------------------------------------------------


@dataclass(slots=True)
class Skill:
    """Skill loaded from a `SKILL.md` file or provided by an application.

    `name`, `description`, and `file_path` are inserted into the system prompt
    in an XML-formatted block as suggested by agentskills.io.
    """

    # Stable skill name used for lookup and model-visible listings.
    name: str
    # Short model-visible description of when to use the skill.
    description: str
    # Full skill instructions.
    content: str
    # Absolute path to the skill file.
    file_path: str
    # Exclude this skill from model-visible skill lists while still allowing
    # explicit application invocation.
    disable_model_invocation: bool = False


@dataclass(slots=True)
class AgentHarnessResources:
    """Resources made available to explicit invocation methods and system-prompt callbacks."""

    # Prompt templates available for explicit invocation.
    prompt_templates: list[Any] | None = None  # list[PromptTemplate]
    # Skills available to the model and explicit skill invocation.
    skills: list[Skill] | None = None


# --- harness stream options / events --------------------------------------------


@dataclass(slots=True)
class AgentHarnessStreamOptions:
    """Curated provider request options owned by the harness and snapshotted per turn."""

    # Preferred transport forwarded to the stream function.
    transport: Any = None
    # Provider request timeout in milliseconds.
    timeout_ms: float | None = None
    # Maximum provider retry attempts.
    max_retries: int | None = None
    # Optional cap for provider-requested retry delays.
    max_retry_delay_ms: float | None = None
    # Additional request headers merged with auth and lifecycle headers.
    headers: dict[str, str] | None = None
    # Provider metadata forwarded with requests.
    metadata: dict[str, Any] | None = None
    # Provider cache retention hint.
    cache_retention: Any = None


# Per-request stream option patch returned by provider hooks. A plain dict so
# key *presence* carries meaning (pi uses `Object.hasOwn`): a present key
# replaces the option; `headers`/`metadata` sub-dicts merge with None values
# deleting keys, and an explicit `"headers": None` clears all headers.
type AgentHarnessStreamOptionsPatch = dict[str, Any]

type AgentHarnessPhase = Literal["idle", "turn", "compaction", "branch_summary", "retry"]


@dataclass(slots=True)
class QueueUpdateEvent:
    steer: list[Any]
    follow_up: list[Any]
    next_turn: list[Any]
    type: Literal["queue_update"] = "queue_update"


@dataclass(slots=True)
class SavePointEvent:
    had_pending_mutations: bool
    type: Literal["save_point"] = "save_point"


@dataclass(slots=True)
class HarnessAbortEvent:
    cleared_steer: list[Any]
    cleared_follow_up: list[Any]
    type: Literal["abort"] = "abort"


@dataclass(slots=True)
class SettledEvent:
    next_turn_count: int
    type: Literal["settled"] = "settled"


@dataclass(slots=True)
class BeforeAgentStartEvent:
    prompt: str
    system_prompt: str
    resources: AgentHarnessResources
    images: list[Any] | None = None
    type: Literal["before_agent_start"] = "before_agent_start"


@dataclass(slots=True)
class HarnessContextEvent:
    messages: list[Any]
    type: Literal["context"] = "context"


@dataclass(slots=True)
class BeforeProviderRequestEvent:
    model: Any
    session_id: str
    stream_options: AgentHarnessStreamOptions
    type: Literal["before_provider_request"] = "before_provider_request"


@dataclass(slots=True)
class BeforeProviderPayloadEvent:
    model: Any
    payload: Any
    type: Literal["before_provider_payload"] = "before_provider_payload"


@dataclass(slots=True)
class AfterProviderResponseEvent:
    status: int
    headers: dict[str, str]
    type: Literal["after_provider_response"] = "after_provider_response"


@dataclass(slots=True)
class HarnessToolCallEvent:
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


@dataclass(slots=True)
class HarnessToolResultEvent:
    tool_call_id: str
    tool_name: str
    input: dict[str, Any]
    content: list[Any]
    details: Any
    is_error: bool
    usage: Any = None
    type: Literal["tool_result"] = "tool_result"


@dataclass(slots=True)
class SessionBeforeCompactEvent:
    preparation: Any  # CompactionPreparation
    branch_entries: list[SessionTreeEntry]
    signal: CancelToken
    custom_instructions: str | None = None
    type: Literal["session_before_compact"] = "session_before_compact"


@dataclass(slots=True)
class SessionCompactEvent:
    compaction_entry: CompactionEntry
    from_hook: bool
    type: Literal["session_compact"] = "session_compact"


@dataclass(slots=True)
class TreePreparation:
    target_id: str
    old_leaf_id: str | None
    common_ancestor_id: str | None
    entries_to_summarize: list[SessionTreeEntry]
    user_wants_summary: bool
    custom_instructions: str | None = None
    replace_instructions: bool | None = None
    label: str | None = None


@dataclass(slots=True)
class SessionBeforeTreeEvent:
    preparation: TreePreparation
    signal: CancelToken
    type: Literal["session_before_tree"] = "session_before_tree"


@dataclass(slots=True)
class SessionTreeEvent:
    new_leaf_id: str | None
    old_leaf_id: str | None
    summary_entry: BranchSummaryEntry | None = None
    from_hook: bool | None = None
    type: Literal["session_tree"] = "session_tree"


@dataclass(slots=True)
class RetryScheduledEvent:
    operation: Literal["compaction", "branch_summary"]
    attempt: int
    max_attempts: int
    delay_ms: float
    error_message: str
    type: Literal["retry_scheduled"] = "retry_scheduled"


@dataclass(slots=True)
class RetryAttemptStartEvent:
    operation: Literal["compaction", "branch_summary"]
    type: Literal["retry_attempt_start"] = "retry_attempt_start"


@dataclass(slots=True)
class RetryFinishedEvent:
    operation: Literal["compaction", "branch_summary"]
    type: Literal["retry_finished"] = "retry_finished"


@dataclass(slots=True)
class ModelUpdateEvent:
    model: Any
    previous_model: Any
    source: Literal["set", "restore"]
    type: Literal["model_update"] = "model_update"


@dataclass(slots=True)
class ThinkingLevelUpdateEvent:
    level: str
    previous_level: str
    type: Literal["thinking_level_update"] = "thinking_level_update"


@dataclass(slots=True)
class ToolsUpdateEvent:
    tool_names: list[str]
    previous_tool_names: list[str]
    active_tool_names: list[str]
    previous_active_tool_names: list[str]
    source: Literal["set", "restore"]
    type: Literal["tools_update"] = "tools_update"


@dataclass(slots=True)
class ResourcesUpdateEvent:
    resources: AgentHarnessResources
    previous_resources: AgentHarnessResources
    type: Literal["resources_update"] = "resources_update"


# --- harness hook results -------------------------------------------------------


@dataclass(slots=True)
class BeforeAgentStartResult:
    messages: list[Any] | None = None
    system_prompt: str | None = None


@dataclass(slots=True)
class ContextResult:
    messages: list[Any]


@dataclass(slots=True)
class BeforeProviderRequestResult:
    stream_options: AgentHarnessStreamOptionsPatch | None = None


@dataclass(slots=True)
class BeforeProviderPayloadResult:
    payload: Any


@dataclass(slots=True)
class HarnessToolCallResult:
    block: bool | None = None
    reason: str | None = None


@dataclass(slots=True)
class HarnessToolResultPatch:
    content: list[Any] | None = None
    details: Any = None
    is_error: bool | None = None
    usage: Any = None
    terminate: bool | None = None


@dataclass(slots=True)
class SessionBeforeCompactResult:
    cancel: bool | None = None
    compaction: Any = None  # CompactionResult


@dataclass(slots=True)
class BranchSummaryOverride:
    summary: str
    details: Any = None
    # Usage from the LLM call that generated this summary, if available.
    usage: Any = None


@dataclass(slots=True)
class SessionBeforeTreeResult:
    cancel: bool | None = None
    summary: BranchSummaryOverride | None = None
    custom_instructions: str | None = None
    replace_instructions: bool | None = None
    label: str | None = None


@dataclass(slots=True)
class AbortResult:
    cleared_steer: list[Any]
    cleared_follow_up: list[Any]


@dataclass(slots=True)
class NavigateTreeResult:
    cancelled: bool
    editor_text: str | None = None
    summary_entry: BranchSummaryEntry | None = None


@dataclass(slots=True, kw_only=True)
class JsonlSessionMetadata(SessionMetadata):
    cwd: str
    path: str
    parent_session_path: str | None = None
    metadata: dict[str, Any] | None = None
