"""AgentHarness v2 scaffold (port of pi `harness/agent-harness.ts`).

Upstream deleted the v1 harness runtime and replaced it with this
compile-complete scaffold over the v4 session model: the API surface, error
vocabulary, and configuration accessors are real, but every orchestration
method raises `HarnessNotImplemented` until the v2 runtime lands. pidrei
mirrors that state 1:1 (sans upstream's telemetry `context` option — telemetry
is not ported).
"""

import copy
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from pidrei_ai.utils.retry import RetryPolicy

from ..types import QueueMode, ThinkingLevel
from .compaction.compaction import CompactionSettings
from .result import TaggedError
from .session.session import Session
from .session.types import Entry, JsonValue, RecordQuery, SessionTree
from .types import AgentHarnessResources


class LaneBusy(TaggedError):
    pass


class MissingIdentities(TaggedError):
    pass


class NoActiveRun(TaggedError):
    pass


class NoActiveOperation(TaggedError):
    pass


class NothingToResume(TaggedError):
    pass


class InvalidMessage(TaggedError):
    pass


class UnknownSkill(TaggedError):
    pass


class UnknownTemplate(TaggedError):
    pass


class UnknownTarget(TaggedError):
    pass


class UnknownQueueItem(TaggedError):
    pass


class LaneExists(TaggedError):
    pass


class InvalidLane(TaggedError):
    pass


class NothingToCompact(TaggedError):
    pass


class Closed(TaggedError):
    pass


class HarnessFault(Exception):
    def __init__(self, message: str, cause: Any):
        super().__init__(message)
        self.cause = cause


class HarnessClosed(Exception):
    def __init__(self) -> None:
        super().__init__("AgentHarness was closed while the operation was active")


class HarnessNotImplemented(Exception):
    def __init__(self, operation: str):
        super().__init__(f"AgentHarness.{operation} is not implemented yet")
        self.operation = operation


@dataclass(slots=True, kw_only=True)
class OperationError:
    code: str
    message: str


type OperationKind = Literal["run", "compaction", "navigation"]


@dataclass(slots=True, kw_only=True)
class RunOutcome:
    kind: Literal["completed", "aborted", "failed", "suspended"]
    leaf_id: str
    final_entry_id: str | None = None
    final_message: Any = None  # AssistantMessage
    error: OperationError | None = None
    deferred: Any = None  # DeferredHandle


@dataclass(slots=True, kw_only=True)
class CompactionOutcome:
    kind: Literal["completed", "declined", "aborted", "failed"]
    leaf_id: str
    entry: Any = None  # CompactionEntry
    error: OperationError | None = None


@dataclass(slots=True, kw_only=True)
class NavigationOutcome:
    kind: Literal["completed", "declined", "aborted", "failed"]
    new_leaf_id: str | None = None
    leaf_id: str | None = None
    summary_entry: Any = None  # BranchSummaryEntry
    error: OperationError | None = None


@dataclass(slots=True, kw_only=True)
class NavigateOptions:
    summarize: bool | None = None
    custom_instructions: str | None = None
    label: str | None = None


@dataclass(slots=True, kw_only=True)
class SuspendedOperation:
    lane: str
    kind: OperationKind
    id: str
    started_at: int
    reason: Literal["crash", "deferred"]
    prompt: list[Any] | None = None  # AgentMessage[]
    deferred: Any = None  # DeferredHandle
    aborting: dict[str, list[Any]] | None = None  # { steer, followUp }
    missing: dict[str, list[str]] = field(default_factory=lambda: {"tools": [], "models": []})


@dataclass(slots=True, kw_only=True)
class LaneOperationInfo:
    id: str
    kind: OperationKind
    status: Literal["running", "suspended", "aborting"]


@dataclass(slots=True, kw_only=True)
class LaneInfo:
    name: str
    leaf_id: str | None
    operation: LaneOperationInfo | None


@dataclass(slots=True, kw_only=True)
class QueuedItem:
    entry_id: str
    message: Any  # AgentMessage


@dataclass(slots=True, kw_only=True)
class LaneSnapshot:
    lane: str
    transcript: list[Entry]
    leaf_id: str | None
    operation: LaneOperationInfo | None
    queues: dict[str, list[QueuedItem]]  # steer / followUp / nextRun
    pending_writes: list[dict[str, Any]]  # { id, entry }
    faulted: bool


@dataclass(slots=True, kw_only=True)
class SessionSnapshot:
    lanes: list[LaneInfo]
    faulted: bool


type HookName = Literal[
    "before_run",
    "before_resume",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
]


class Hooks(Protocol):
    def on(self, name: HookName, handler: Callable[[Any], Any], id: str | None = None) -> Callable[[], None]: ...


class Events(Protocol):
    def on(self, type: str, listener: Callable[[Any], Any]) -> Callable[[], None]: ...


class _UnavailableRegistry:
    def __init__(self, operation: str, is_closed: Callable[[], bool]):
        self._operation = operation
        self._is_closed = is_closed

    def on(self, name: str, handler: Callable[[Any], Any], id: str | None = None) -> Callable[[], None]:
        raise HarnessClosed() if self._is_closed() else HarnessNotImplemented(self._operation)


# AgentTool plus the replay classification the durable runtime records per call.
type HarnessTool = Any
type Resources = AgentHarnessResources
type StreamOptions = Any  # SimpleStreamOptions
type EntryProjector = Callable[[Entry], Awaitable[list[Any]]]


@dataclass(slots=True, kw_only=True)
class AgentHarnessOptions:
    session: Session
    models: Any  # Models
    model: Any  # Model
    thinking_level: ThinkingLevel | None = None
    active_tool_names: list[str] | None = None
    tools: list[HarnessTool] | None = None
    tool_context: Any = None
    system_prompt: Any = None  # str | () -> Awaitable[str]
    resources: Resources | None = None
    stream_options: StreamOptions | None = None
    retry: RetryPolicy | None = None
    compaction: CompactionSettings | None = None
    steering_mode: QueueMode | None = None
    follow_up_mode: QueueMode | None = None
    tool_execution: Literal["sequential", "parallel"] | None = None
    drive: Literal["automatic", "manual"] | None = None
    to_provider_messages: Any = None
    entry_projectors: dict[str, EntryProjector] | None = None


class AgentHarness:
    name = "main"

    def __init__(self, options: AgentHarnessOptions):
        self._durable_session = options.session
        self.session: SessionTree = options.session
        self._closed = False
        self.hooks: Hooks = _UnavailableRegistry("hooks.on", lambda: self._closed)
        self.events: Events = _UnavailableRegistry("events.on", lambda: self._closed)
        self._model = options.model
        self._thinking_level: ThinkingLevel = options.thinking_level if options.thinking_level is not None else "off"
        if options.active_tool_names is not None:
            self._active_tool_names = list(options.active_tool_names)
        elif options.tools is not None:
            self._active_tool_names = [tool.name for tool in options.tools]
        else:
            self._active_tool_names = []
        self._tools: list[HarnessTool] = list(options.tools) if options.tools is not None else []
        self._resources = AgentHarnessResources(
            skills=list(options.resources.skills)
            if options.resources is not None and options.resources.skills
            else None,
            prompt_templates=(
                list(options.resources.prompt_templates)
                if options.resources is not None and options.resources.prompt_templates
                else None
            ),
        )
        self._stream_options = copy.copy(options.stream_options) if options.stream_options is not None else None
        self._retry_policy = (
            options.retry
            if options.retry is not None
            else RetryPolicy(enabled=False, max_retries=0, base_delay_ms=1000)
        )
        self._compaction_settings = (
            options.compaction
            if options.compaction is not None
            else CompactionSettings(enabled=True, reserve_tokens=16384, keep_recent_tokens=20000)
        )
        self._steering_mode: QueueMode = options.steering_mode if options.steering_mode is not None else "one-at-a-time"
        self._follow_up_mode: QueueMode = (
            options.follow_up_mode if options.follow_up_mode is not None else "one-at-a-time"
        )

    @staticmethod
    async def create(options: AgentHarnessOptions) -> tuple[AgentHarness, list[SuspendedOperation]]:
        records = await options.session.find_records(RecordQuery(limit=1))
        if records:
            raise HarnessNotImplemented("create.restore")
        return AgentHarness(options), []

    def _unavailable(self, operation: str) -> Any:
        raise HarnessClosed() if self._closed else HarnessNotImplemented(operation)

    async def get_leaf_id(self) -> str | None:
        return await self._durable_session.get_leaf_id()

    async def prompt(self, input: Any, images: list[Any] | None = None) -> Any:
        return self._unavailable("prompt")

    async def skill(self, name: str, additional_instructions: str | None = None) -> Any:
        return self._unavailable("skill")

    async def prompt_from_template(self, name: str, args: list[str] | None = None) -> Any:
        return self._unavailable("prompt_from_template")

    async def compact(self, custom_instructions: str | None = None) -> Any:
        return self._unavailable("compact")

    async def navigate_tree(self, target_id: str | None, options: NavigateOptions | None = None) -> Any:
        return self._unavailable("navigate_tree")

    async def resume(self) -> Any:
        return self._unavailable("resume")

    async def abort(self) -> Any:
        return self._unavailable("abort")

    async def steer(self, input: Any, images: list[Any] | None = None) -> Any:
        return self._unavailable("steer")

    async def follow_up(self, input: Any, images: list[Any] | None = None) -> Any:
        return self._unavailable("follow_up")

    async def next_run(self, input: Any, images: list[Any] | None = None) -> Any:
        return self._unavailable("next_run")

    async def cancel_queued(self, entry_id: str) -> Any:
        return self._unavailable("cancel_queued")

    async def record_usage(self, usage: Any, entry_id: str | None = None, details: JsonValue = None) -> Any:
        return self._unavailable("record_usage")

    async def wait_for_idle(self) -> None:
        return self._unavailable("wait_for_idle")

    async def run_when_idle(self, callback: Callable[[], Any]) -> None:
        return self._unavailable("run_when_idle")

    async def peek_action(self) -> Any:
        return self._unavailable("peek_action")

    async def execute_action(self) -> Any:
        return self._unavailable("execute_action")

    async def run_to_completion(self) -> None:
        return self._unavailable("run_to_completion")

    async def get_model(self) -> Any:
        return self._model

    async def set_model(self, model: Any) -> None:
        self._model = model

    async def get_thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    async def set_thinking_level(self, level: ThinkingLevel) -> None:
        self._thinking_level = level

    async def get_active_tools(self) -> list[str]:
        return list(self._active_tool_names)

    async def set_active_tools(self, names: list[str]) -> None:
        self._active_tool_names = list(names)

    async def watch(self) -> Any:
        return self._unavailable("watch")

    async def lane(self, name: str) -> Any:
        return self._unavailable("lane")

    async def create_lane(self, name: str, at: str | None) -> Any:
        return self._unavailable("create_lane")

    async def lanes(self) -> list[LaneInfo]:
        return self._unavailable("lanes")

    async def get_tools(self) -> list[HarnessTool]:
        return list(self._tools)

    async def set_tools(self, tools: list[HarnessTool], active_names: list[str] | None = None) -> None:
        self._tools = list(tools)
        self._active_tool_names = list(active_names) if active_names is not None else [tool.name for tool in tools]

    async def get_resources(self) -> Resources:
        return AgentHarnessResources(
            skills=list(self._resources.skills) if self._resources.skills else None,
            prompt_templates=list(self._resources.prompt_templates) if self._resources.prompt_templates else None,
        )

    async def set_resources(self, resources: Resources) -> None:
        self._resources = AgentHarnessResources(
            skills=list(resources.skills) if resources.skills else None,
            prompt_templates=list(resources.prompt_templates) if resources.prompt_templates else None,
        )

    async def get_stream_options(self) -> StreamOptions:
        return copy.copy(self._stream_options)

    async def set_stream_options(self, options: StreamOptions) -> None:
        self._stream_options = copy.copy(options)

    async def get_retry_policy(self) -> RetryPolicy:
        return replace(self._retry_policy)

    async def set_retry_policy(self, policy: RetryPolicy) -> None:
        self._retry_policy = replace(policy)

    async def get_compaction_settings(self) -> CompactionSettings:
        return replace(self._compaction_settings)

    async def set_compaction_settings(self, settings: CompactionSettings) -> None:
        self._compaction_settings = replace(settings)

    async def get_steering_mode(self) -> QueueMode:
        return self._steering_mode

    async def set_steering_mode(self, mode: QueueMode) -> None:
        self._steering_mode = mode

    async def get_follow_up_mode(self) -> QueueMode:
        return self._follow_up_mode

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_mode = mode

    async def watch_session(self) -> Any:
        return self._unavailable("watch_session")

    async def close(self) -> None:
        self._closed = True
