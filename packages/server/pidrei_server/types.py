"""Service contract for durable sessions (port of pi server `types.ts`).

pi's `MaybePromise<T>` unions are not ported: per the repo's async-only
callback rule, every contract method that may suspend returns an awaitable
(`PiSessionRuntime.snapshot`, `ByteConnection.close`). Runtime events are
tagged dataclasses instead of anonymous object literals.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from pidrei_protocol import (
    ModelMetadata,
    ModelRef,
    SessionMetadata,
    SessionPhase,
    SessionSnapshot,
    ThinkingLevel,
    TranscriptProgress,
)

from .errors import PiServerError


if TYPE_CHECKING:
    from .listener import PiServerListener


@dataclass(slots=True, frozen=True)
class PiServerOptions:
    listeners: list[PiServerListener]
    max_frame_length: int | None = None
    handshake_timeout_ms: int | None = None
    server_id: str | None = None
    on_error: Callable[[Exception], None] | None = None


@dataclass(slots=True, frozen=True)
class PromptInput:
    text: str


type SteerInput = PromptInput


@dataclass(slots=True, frozen=True)
class CreateSessionOptions:
    # A collision-resistant ID assigned by PiServer. The service must persist this exact ID.
    id: str
    cwd: str | None = None
    name: str | None = None
    model: ModelRef | None = None
    thinking_level: ThinkingLevel | None = None


@dataclass(slots=True, frozen=True)
class SessionRuntimeSnapshotEvent:
    type: Literal["snapshot"] = "snapshot"


@dataclass(slots=True, frozen=True)
class SessionRuntimeProgressEvent:
    progress: TranscriptProgress
    type: Literal["progress"] = "progress"


@dataclass(slots=True, frozen=True)
class SessionRuntimeErrorEvent:
    error: PiServerError
    type: Literal["error"] = "error"


type PiSessionRuntimeEvent = SessionRuntimeSnapshotEvent | SessionRuntimeProgressEvent | SessionRuntimeErrorEvent


class PiSessionRuntime(Protocol):
    """One acquired durable session. Conflicting operations must reject rather than queue."""

    def snapshot(self) -> Awaitable[SessionSnapshot]: ...
    def get_phase(self) -> SessionPhase: ...
    def prompt(self, input: PromptInput) -> Awaitable[None]: ...
    def steer(self, input: SteerInput) -> Awaitable[None]: ...
    def abort(self) -> Awaitable[None]: ...
    def set_model(self, model: ModelRef) -> Awaitable[None]: ...
    def set_thinking(self, thinking_level: ThinkingLevel) -> Awaitable[None]: ...
    def subscribe(self, listener: Callable[[PiSessionRuntimeEvent], None]) -> Callable[[], None]: ...
    def dispose(self) -> Awaitable[None]: ...


class PiServerService(Protocol):
    """Service boundary for durable sessions and exclusively acquired runtimes."""

    def list_sessions(self) -> Awaitable[list[SessionMetadata]]: ...
    def list_models(self) -> Awaitable[list[ModelMetadata]]: ...
    def create_session(self, options: CreateSessionOptions) -> Awaitable[PiSessionRuntime]: ...
    def open_session(self, session_id: str) -> Awaitable[PiSessionRuntime]: ...


type SessionRuntime = PiSessionRuntime
type SessionRuntimeEvent = PiSessionRuntimeEvent
