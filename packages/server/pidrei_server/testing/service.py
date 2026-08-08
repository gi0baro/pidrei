"""In-memory test service and runtime (port of pi server `testing/service.ts`).

pi's testing `Deferred` is re-exported from the shared promise module (await
`deferred` where upstream awaits `.promise`). `structuredClone` becomes
`copy.deepcopy`.
"""

import copy
from collections.abc import Callable
from dataclasses import dataclass

from pidrei_protocol import (
    ModelMetadata,
    ModelRef,
    SessionMetadata,
    SessionPhase,
    SessionSnapshot,
    ThinkingLevel,
    TranscriptProgress,
)

from ..errors import PiServerError
from ..promise import Deferred
from ..types import (
    CreateSessionOptions,
    PiSessionRuntimeEvent,
    PromptInput,
    SessionRuntimeErrorEvent,
    SessionRuntimeProgressEvent,
    SessionRuntimeSnapshotEvent,
)


TEST_MODEL: ModelMetadata = {
    "provider": "test",
    "id": "small",
    "name": "Test Small",
    "api": "test-api",
    "reasoning": True,
    "input": ["text", "image"],
    "contextWindow": 16_000,
    "maxTokens": 2_000,
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "supportedThinkingLevels": ["off", "medium", "high"],
    "authenticated": True,
}


@dataclass(slots=True)
class _StoredSession:
    snapshot: SessionSnapshot


@dataclass(slots=True)
class _PendingPrompt:
    input: PromptInput
    done: Deferred


class TestSessionRuntime:
    __test__ = False  # not a pytest class, despite the upstream name

    def __init__(self, stored: _StoredSession, on_dispose: Callable[[], None]) -> None:
        self.disposed = Deferred()
        self.dispose_count = 0
        self.steers: list[PromptInput] = []
        self._stored = stored
        self._on_dispose = on_dispose
        self._listeners: set[Callable[[PiSessionRuntimeEvent], None]] = set()
        self._pending_prompt: _PendingPrompt | None = None

    async def snapshot(self) -> SessionSnapshot:
        return copy.deepcopy(self._stored.snapshot)

    def get_phase(self) -> SessionPhase:
        return self._stored.snapshot["phase"]

    async def prompt(self, input: PromptInput) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "A prompt is already running")
        done = Deferred()
        self._pending_prompt = _PendingPrompt(input=input, done=done)
        revision = self._stored.snapshot["revision"]
        self._update(
            {
                "phase": "turn",
                "transcript": [
                    *self._stored.snapshot["transcript"],
                    {
                        "id": f"user-{revision + 1}",
                        "role": "user",
                        "content": [{"type": "text", "text": input.text}],
                        "timestamp": revision + 1,
                    },
                ],
            }
        )
        outcome = await done
        revision = self._stored.snapshot["revision"]
        if outcome == "complete":
            assistant = {
                "id": f"assistant-{revision + 1}",
                "role": "assistant",
                "content": [{"type": "text", "text": f"reply:{input.text}"}],
                "status": "complete",
                "model": self._stored.snapshot["model"],
                "stopReason": "stop",
                "timestamp": revision + 1,
            }
        else:
            assistant = {
                "id": f"assistant-{revision + 1}",
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
                "status": "aborted",
                "model": self._stored.snapshot["model"],
                "stopReason": "aborted",
                "timestamp": revision + 1,
            }
        self._update({"phase": "idle", "transcript": [*self._stored.snapshot["transcript"], assistant]})
        self._pending_prompt = None

    async def steer(self, input: PromptInput) -> None:
        if self.get_phase() == "idle":
            raise PiServerError("busy", "There is no active prompt to steer")
        self.steers.append(input)
        revision = self._stored.snapshot["revision"]
        self._update(
            {
                "queuedSteerCount": self._stored.snapshot["queuedSteerCount"] + 1,
                "queuedSteer": [
                    *self._stored.snapshot["queuedSteer"],
                    {
                        "id": f"steer-{revision + 1}",
                        "role": "user",
                        "content": [{"type": "text", "text": input.text}],
                        "timestamp": revision + 1,
                    },
                ],
            }
        )

    async def abort(self) -> None:
        if self._pending_prompt is None:
            raise PiServerError("busy", "There is no active prompt to abort")
        self._pending_prompt.done.resolve("aborted")

    async def set_model(self, model: ModelRef) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "Session is busy")
        self._update({"model": model})

    async def set_thinking(self, thinking_level: ThinkingLevel) -> None:
        if self.get_phase() != "idle":
            raise PiServerError("busy", "Session is busy")
        self._update({"thinkingLevel": thinking_level})

    def subscribe(self, listener: Callable[[PiSessionRuntimeEvent], None]) -> Callable[[], None]:
        self._listeners.add(listener)
        return lambda: self._listeners.discard(listener)

    async def dispose(self) -> None:
        self.dispose_count += 1
        self._on_dispose()
        self.disposed.resolve(None)

    def set_phase(self, phase: SessionPhase) -> None:
        self._stored.snapshot = {**self._stored.snapshot, "phase": phase}

    def finish_prompt(self) -> None:
        if self._pending_prompt is None:
            raise Exception("No prompt is pending")
        self._pending_prompt.done.resolve("complete")

    def emit_progress(self, progress: TranscriptProgress) -> None:
        for listener in list(self._listeners):
            listener(SessionRuntimeProgressEvent(progress=progress))

    def emit_error(self, error: PiServerError) -> None:
        for listener in list(self._listeners):
            listener(SessionRuntimeErrorEvent(error=error))

    def emit_snapshot(self) -> None:
        for listener in list(self._listeners):
            listener(SessionRuntimeSnapshotEvent())

    def _update(self, updates: dict) -> None:
        self._stored.snapshot = {
            **self._stored.snapshot,
            **updates,
            "revision": self._stored.snapshot["revision"] + 1,
            "updatedAt": self._stored.snapshot["updatedAt"] + 1,
        }
        self.emit_snapshot()


@dataclass(slots=True, frozen=True)
class ListDelay:
    entered: Deferred
    release: Deferred


class TestServerService:
    __test__ = False  # not a pytest class, despite the upstream name

    def __init__(self) -> None:
        self.sessions: dict[str, _StoredSession] = {}
        self.runtimes: dict[str, list[TestSessionRuntime]] = {}
        self.locked: set[str] = set()
        self.last_created_id: str | None = None
        self._next_list_delay: ListDelay | None = None

    async def list_sessions(self) -> list[SessionMetadata]:
        delay = self._next_list_delay
        if delay is not None:
            self._next_list_delay = None
            delay.entered.resolve(None)
            await delay.release
        return [
            {
                "id": stored.snapshot["id"],
                "createdAt": stored.snapshot["createdAt"],
                "updatedAt": stored.snapshot["updatedAt"],
                "sessionName": stored.snapshot["name"],
                "cwd": stored.snapshot["cwd"],
            }
            for stored in self.sessions.values()
        ]

    async def list_models(self) -> list[ModelMetadata]:
        return [TEST_MODEL]

    async def create_session(self, options: CreateSessionOptions) -> TestSessionRuntime:
        self.last_created_id = options.id
        if options.id in self.sessions:
            raise PiServerError("session_locked", "Session already exists")
        self.seed(options.id, options.name, options.cwd, options.model, options.thinking_level)
        return self._acquire(options.id)

    async def open_session(self, session_id: str) -> TestSessionRuntime:
        if session_id not in self.sessions:
            raise PiServerError("not_found", f"Unknown session: {session_id}")
        if session_id in self.locked:
            raise PiServerError("session_locked", f"Session is locked: {session_id}")
        return self._acquire(session_id)

    def seed(
        self,
        session_id: str = "session-1",
        name: str | None = None,
        cwd: str | None = None,
        model: ModelRef | None = None,
        thinking_level: ThinkingLevel | None = None,
    ) -> None:
        self.sessions[session_id] = _StoredSession(
            snapshot={
                "id": session_id,
                "name": name if name is not None else f"Session {session_id}",
                "cwd": cwd if cwd is not None else "/tmp/pi-server-conformance",  # noqa: S108 (upstream literal)
                "createdAt": 1,
                "updatedAt": 1,
                "phase": "idle",
                "model": model if model is not None else {"provider": TEST_MODEL["provider"], "id": TEST_MODEL["id"]},
                "thinkingLevel": thinking_level if thinking_level is not None else "off",
                "attached": False,
                "locked": False,
                "revision": 0,
                "transcript": [],
                "queuedSteer": [],
                "queuedSteerCount": 0,
            }
        )

    def delay_next_list(self) -> ListDelay:
        delay = ListDelay(entered=Deferred(), release=Deferred())
        self._next_list_delay = delay
        return delay

    def latest_runtime(self, session_id: str) -> TestSessionRuntime:
        runtimes = self.runtimes.get(session_id)
        if not runtimes:
            raise Exception(f"No runtime for {session_id}")
        return runtimes[-1]

    def _acquire(self, session_id: str) -> TestSessionRuntime:
        stored = self.sessions.get(session_id)
        if stored is None:
            raise Exception(f"Unknown session: {session_id}")
        self.locked.add(session_id)
        runtime = TestSessionRuntime(stored, lambda: self.locked.discard(session_id))
        self.runtimes.setdefault(session_id, []).append(runtime)
        return runtime
