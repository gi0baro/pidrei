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
    async def rename_file(
        self, source_path: str, destination_path: str, cancel: CancelToken | None = None
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
type CompactionErrorCode = Literal["aborted", "summarization_failed"]


class CompactionError(Exception):
    """Error returned by compaction helpers."""

    def __init__(self, code: CompactionErrorCode, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if cause is not None:
            self.__cause__ = cause


# Stable branch-summary error codes returned by branch summarization helpers.
type BranchSummaryErrorCode = Literal["aborted", "summarization_failed"]


class BranchSummaryError(Exception):
    """Error returned by branch summarization helpers."""

    def __init__(self, code: BranchSummaryErrorCode, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        if cause is not None:
            self.__cause__ = cause


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
