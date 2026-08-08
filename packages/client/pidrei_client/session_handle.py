"""Session lease handle (port of pi client `session-handle.ts`).

Command methods are sync prologues returning awaitables: pi's async methods
issue their request synchronously (run-to-first-await), and lease/ordering
semantics depend on that. Synchronous failures surface as rejected awaitables,
matching how a JS async function converts throws into rejections.
`Symbol.asyncDispose` maps to the async context-manager protocol.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, Self

from pidrei_protocol import Command, CommandResult, ModelRef, ServerEvent, SessionSnapshot, ThinkingLevel

from .promise import rejected
from .types import Unsubscribe


type SessionLeaseMode = Literal["shared", "exclusive"]


@dataclass(slots=True, frozen=True)
class AcquireSessionOptions:
    mode: SessionLeaseMode


@dataclass(slots=True, frozen=True)
class SessionHandleCallbacks:
    is_attached: Callable[[], bool]
    get_snapshot: Callable[[], SessionSnapshot | None]
    subscribe: Callable[[Callable[[SessionSnapshot], None]], Unsubscribe]
    on_event: Callable[[Callable[[ServerEvent], None]], Unsubscribe]
    detach: Callable[[], Awaitable[None]]
    dispose: Callable[[], Awaitable[None]]
    request: Callable[[Command], Awaitable[CommandResult]]


class SessionHandle:
    def __init__(self, session_id: str, callbacks: SessionHandleCallbacks) -> None:
        self.id = session_id
        self._callbacks = callbacks

    @property
    def attached(self) -> bool:
        return self._callbacks.is_attached()

    @property
    def active(self) -> bool:
        return self.attached

    @property
    def snapshot(self) -> SessionSnapshot | None:
        return self._callbacks.get_snapshot()

    def subscribe(self, listener: Callable[[SessionSnapshot], None]) -> Unsubscribe:
        return self._callbacks.subscribe(listener)

    def on_event(self, listener: Callable[[ServerEvent], None]) -> Unsubscribe:
        return self._callbacks.on_event(listener)

    def detach(self) -> Awaitable[None]:
        return self._callbacks.detach()

    def dispose(self) -> Awaitable[None]:
        return self._callbacks.dispose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.dispose()

    def prompt(self, text: str) -> Awaitable[SessionSnapshot]:
        return self._request_session({"command": "prompt", "sessionId": self.id, "text": text})

    def steer(self, text: str) -> Awaitable[SessionSnapshot]:
        return self._request_session({"command": "steer", "sessionId": self.id, "text": text})

    def abort(self) -> Awaitable[SessionSnapshot]:
        return self._request_session({"command": "abort", "sessionId": self.id})

    def set_model(self, model: ModelRef) -> Awaitable[SessionSnapshot]:
        return self._request_session({"command": "set_model", "sessionId": self.id, "model": model})

    def set_thinking(self, thinking_level: ThinkingLevel) -> Awaitable[SessionSnapshot]:
        return self._request_session({"command": "set_thinking", "sessionId": self.id, "thinkingLevel": thinking_level})

    def _request_session(self, command: Command) -> Awaitable[SessionSnapshot]:
        try:
            result = self._callbacks.request(command)
        except Exception as error:
            return rejected(error)
        return _session_of(result)


async def _session_of(result: Awaitable[CommandResult]) -> SessionSnapshot:
    return (await result)["session"]


type SessionLease = SessionHandle
type PiSessionHandle = SessionHandle
