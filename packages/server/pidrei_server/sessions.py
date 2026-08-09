"""Live session lifecycle (port of pi server `sessions.ts`).

The state-machine-entangled paths keep pi's run-to-first-await semantics via
sync prologues: `_terminate` flips `terminal` and `_maybe_dispose` claims
`disposing` at call time, with the awaited remainder driven on the runtime so
it completes even when voided (`scheduleMaybeDispose`). Everything awaited
directly at its call site stays a plain coroutine.

Being synchronous is not the same as being atomic. pi's prologues run to
completion before any other task observes them because there is one event
loop; here two tasks reach the same prologue on different threads, read the
same "not claimed yet" state and both proceed. `_lifecycle_guard` covers every
check-and-claim on a `_LiveSession` — the terminal flip and the three places
that take ownership of `disposing`. CI caught the missing one as a runtime
disposed twice after a terminal error.
"""

import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import tonio.colored as tonio

from pidrei_protocol import Command, CommandResult, ServerMessage, SessionMetadata, SessionSnapshot

from .connection import ByteConnection, ConnectionState
from .errors import PiServerError
from .promise import Deferred, all_settled, driven, gather, resolved
from .types import CreateSessionOptions, PiServerService, PiSessionRuntime, PiSessionRuntimeEvent, PromptInput


@dataclass(slots=True, eq=False)
class _LiveSession:
    id: str
    runtime: PiSessionRuntime
    unsubscribe: Callable[[], None]
    connections: set[ConnectionState] = field(default_factory=set)
    operation_count: int = 0
    ready: bool = False
    terminal: bool = False
    disposing: Deferred | None = None


@dataclass(slots=True, frozen=True)
class LiveSessionManagerOptions:
    service: PiServerService
    is_closing: Callable[[], bool]
    send_message: Callable[[ConnectionState, ServerMessage], Awaitable[bool]]
    close_connection: Callable[[ByteConnection], Awaitable[None]]
    disconnect: Callable[[ConnectionState], Awaitable[None]]
    broadcast_server_snapshot: Callable[[], None]
    report_error: Callable[[object], None]


def _to_metadata(snapshot: SessionSnapshot) -> SessionMetadata:
    return {
        "id": snapshot["id"],
        "createdAt": snapshot["createdAt"],
        "updatedAt": snapshot["updatedAt"],
        "sessionName": snapshot.get("name"),
        "cwd": snapshot["cwd"],
    }


def _without_absent(metadata: SessionMetadata) -> SessionMetadata:
    """Drop None-valued keys, mirroring undefined omission on the JS wire."""
    return {key: value for key, value in metadata.items() if value is not None}


class LiveSessionManager:
    def __init__(self, options: LiveSessionManagerOptions) -> None:
        self._options = options
        self._live_sessions: dict[str, _LiveSession] = {}
        self._opening_sessions: dict[str, Deferred] = {}
        self._lifecycle_guard = threading.Lock()

    async def execute_command(self, connection: ConnectionState, command: Command) -> CommandResult:
        name = command["command"]
        if name == "list":
            return {"command": "list", "sessions": await self.list_metadata()}
        if name == "create":
            session_id = str(uuid.uuid4())
            options = CreateSessionOptions(
                id=session_id,
                cwd=command.get("cwd"),
                name=command.get("name"),
                model=command.get("model"),
                thinking_level=command.get("thinkingLevel"),
            )
            live = await self._acquire(session_id, lambda: self._options.service.create_session(options))
            await self._attach(connection, live)
            session = self._for_connection(await self._broadcast_snapshot(live), connection)
            self._options.broadcast_server_snapshot()
            return {"command": "create", "session": session}
        if name == "attach":
            session_id = command["sessionId"]
            live = await self._acquire(session_id, lambda: self._options.service.open_session(session_id))
            await self._attach(connection, live)
            session = self._for_connection(await self._broadcast_snapshot(live), connection)
            self._options.broadcast_server_snapshot()
            return {"command": "attach", "session": session}
        if name == "detach":
            session_id = command["sessionId"]
            live_or_none = self._live_sessions.get(session_id)
            if session_id in connection.session_ids:
                connection.session_ids.discard(session_id)
                if live_or_none is not None:
                    live_or_none.connections.discard(connection)
                    if live_or_none.connections and not live_or_none.terminal and live_or_none.disposing is None:
                        await self._broadcast_snapshot(live_or_none)
                    await self._maybe_dispose(live_or_none)
                self._options.broadcast_server_snapshot()
            return {"command": "detach", "sessionId": session_id}
        if name == "prompt":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(
                connection, live, lambda: live.runtime.prompt(PromptInput(text=command["text"]))
            )
            return {"command": "prompt", "session": session}
        if name == "steer":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(
                connection, live, lambda: live.runtime.steer(PromptInput(text=command["text"]))
            )
            return {"command": "steer", "session": session}
        if name == "abort":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(connection, live, lambda: live.runtime.abort())
            return {"command": "abort", "session": session}
        if name == "set_model":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(connection, live, lambda: live.runtime.set_model(command["model"]))
            return {"command": "set_model", "session": session}
        if name == "set_thinking":
            live = self._require_attached(connection, command["sessionId"])
            session = await self._run_operation(
                connection, live, lambda: live.runtime.set_thinking(command["thinkingLevel"])
            )
            return {"command": "set_thinking", "session": session}
        raise PiServerError("invalid_request", f"Unknown command: {name}")

    async def disconnect(self, connection: ConnectionState) -> None:
        sessions = [
            live
            for live in (self._live_sessions.get(session_id) for session_id in connection.session_ids)
            if live is not None
        ]
        connection.session_ids.clear()
        for live in sessions:
            live.connections.discard(connection)
        results = await all_settled([self._maybe_dispose(live) for live in sessions])
        for _, error in results:
            if error is not None:
                self._options.report_error(error)

    async def list_metadata(self) -> list[SessionMetadata]:
        stored = await self._options.service.list_sessions()
        lives = [live for live in self._live_sessions.values() if live.disposing is None]
        snapshots = await gather([self._normalized_snapshot(live) for live in lives])
        live_by_id = {live.id: snapshot for live, snapshot in zip(lives, snapshots, strict=True)}
        metadata: list[SessionMetadata] = []
        for item in stored:
            snapshot = live_by_id.get(item["id"])
            if snapshot is None:
                metadata.append(item)
                continue
            del live_by_id[item["id"]]
            metadata.append(_without_absent({**item, **_to_metadata(snapshot)}))
        for snapshot in live_by_id.values():
            metadata.append(_without_absent(_to_metadata(snapshot)))
        return metadata

    async def close(self) -> None:
        opening_results = await all_settled(list(self._opening_sessions.values()))
        for _, error in opening_results:
            if error is not None:
                self._options.report_error(error)
        sessions = list(self._live_sessions.values())
        self._live_sessions.clear()

        async def _close_one(live: _LiveSession) -> None:
            # Same check-and-claim as `_maybe_dispose`, and it has to publish
            # its claim for the same reason: a dispose already in flight there
            # must be awaited, not repeated.
            with self._lifecycle_guard:
                existing = live.disposing
                claimed = Deferred() if existing is None else None
                if claimed is not None:
                    live.disposing = claimed
            if claimed is None:
                assert existing is not None
                await existing
                return
            live.unsubscribe()
            try:
                await live.runtime.dispose()
            except BaseException as error:
                claimed.reject(error)
                raise
            claimed.resolve(None)

        await gather([_close_one(live) for live in sessions])

    async def _run_operation(
        self, connection: ConnectionState, live: _LiveSession, operation: Callable[[], Awaitable[None]]
    ) -> SessionSnapshot:
        live.operation_count += 1
        try:
            await operation()
            return self._for_connection(await self._broadcast_snapshot(live), connection)
        finally:
            live.operation_count -= 1
            self._schedule_maybe_dispose(live)

    async def _acquire(
        self, session_id: str, acquire_runtime: Callable[[], Awaitable[PiSessionRuntime]]
    ) -> _LiveSession:
        while True:
            existing = self._live_sessions.get(session_id)
            if existing is not None:
                if existing.terminal:
                    raise PiServerError("session_locked", f"Session runtime is terminating: {session_id}")
                if existing.disposing is not None:
                    await existing.disposing
                    continue
                return existing
            opening = self._opening_sessions.get(session_id)
            if opening is not None:
                return await opening
            pending = Deferred()
            self._opening_sessions[session_id] = pending
            try:
                try:
                    live = await self._create(session_id, acquire_runtime)
                except Exception as error:
                    pending.reject(error)
                    raise
                pending.resolve(live)
                return live
            finally:
                if self._opening_sessions.get(session_id) is pending:
                    del self._opening_sessions[session_id]

    async def _create(
        self, session_id: str, acquire_runtime: Callable[[], Awaitable[PiSessionRuntime]]
    ) -> _LiveSession:
        runtime = await acquire_runtime()
        if self._options.is_closing():
            await runtime.dispose()
            raise Exception("PiServer closed while acquiring a session runtime")
        live: _LiveSession | None = None
        try:
            snapshot = await runtime.snapshot()
            if snapshot["id"] != session_id:
                raise PiServerError(
                    "invalid_request",
                    f"Service returned session {snapshot['id']} for server-assigned session {session_id}",
                )
            live = _LiveSession(id=session_id, runtime=runtime, unsubscribe=lambda: None)
            started = live
            live.unsubscribe = runtime.subscribe(lambda event: self._handle_runtime_event(started, event))
            self._live_sessions[session_id] = live
            live.ready = True
            return live
        except Exception:
            if live is not None:
                live.unsubscribe()
            try:
                await runtime.dispose()
            except Exception as dispose_error:
                self._options.report_error(dispose_error)
            raise

    def _handle_runtime_event(self, live: _LiveSession, event: PiSessionRuntimeEvent) -> None:
        if event.type == "error":
            self._watch(self._terminate(live, event.error))
            return
        if event.type == "progress":
            envelope: ServerMessage = {
                "type": "event",
                "event": {"type": "session_progress", "sessionId": live.id, "progress": event.progress},
            }
            for connection in list(live.connections):
                self._options.send_message(connection, envelope)
        else:
            self._watch(self._broadcast_snapshot(live))
        self._schedule_maybe_dispose(live)

    def _terminate(self, live: _LiveSession, error: PiServerError) -> Awaitable[None]:
        with self._lifecycle_guard:
            if live.terminal:
                return resolved(None)
            live.terminal = True
        self._options.report_error(error)
        live.unsubscribe()
        connections = list(live.connections)

        async def _finish() -> None:
            await gather([self._options.close_connection(connection.connection) for connection in connections])
            await gather([self._options.disconnect(connection) for connection in connections])
            await self._maybe_dispose(live)

        return driven(_finish())

    async def _normalized_snapshot(self, live: _LiveSession) -> SessionSnapshot:
        snapshot = await live.runtime.snapshot()
        if snapshot["id"] != live.id:
            raise PiServerError("invalid_request", f"Runtime session ID changed from {live.id} to {snapshot['id']}")
        return {
            **snapshot,
            "phase": live.runtime.get_phase(),
            "attached": len(live.connections) > 0,
            "locked": True,
        }

    def _for_connection(self, snapshot: SessionSnapshot, connection: ConnectionState) -> SessionSnapshot:
        return {**snapshot, "attached": snapshot["id"] in connection.session_ids}

    async def _broadcast_snapshot(self, live: _LiveSession) -> SessionSnapshot:
        snapshot = await self._normalized_snapshot(live)
        envelope: ServerMessage = {"type": "event", "event": {"type": "session_snapshot", "snapshot": snapshot}}
        for connection in list(live.connections):
            self._options.send_message(connection, envelope)
        return snapshot

    async def _attach(self, connection: ConnectionState, live: _LiveSession) -> None:
        if connection.disconnected or connection.stage != "ready" or connection.connection.closed:
            await self._maybe_dispose(live)
            raise PiServerError("invalid_request", "Connection closed while attaching to a session")
        connection.session_ids.add(live.id)
        live.connections.add(connection)

    def _require_attached(self, connection: ConnectionState, session_id: str) -> _LiveSession:
        if session_id not in connection.session_ids:
            raise PiServerError("invalid_request", f"Connection is not attached to session {session_id}")
        live = self._live_sessions.get(session_id)
        if live is None or live.terminal or live.disposing is not None:
            raise PiServerError("not_found", f"Session is not live: {session_id}")
        return live

    def _schedule_maybe_dispose(self, live: _LiveSession) -> None:
        self._watch(self._maybe_dispose(live))

    def _maybe_dispose(self, live: _LiveSession) -> Awaitable[None]:
        with self._lifecycle_guard:
            if (
                self._options.is_closing()
                or not live.ready
                or live.disposing is not None
                or len(live.connections) > 0
                or live.operation_count > 0
                or (not live.terminal and live.runtime.get_phase() != "idle")
            ):
                return live.disposing if live.disposing is not None else resolved(None)
            # Claim before releasing the guard, so a concurrent caller sees the
            # claim rather than an unclaimed session. The claim is a bare
            # Deferred so `unsubscribe()` and the runtime's own `dispose()`
            # — both service-supplied — never run under the lock.
            claimed = Deferred()
            live.disposing = claimed
        live.unsubscribe()

        async def _dispose() -> None:
            try:
                try:
                    await live.runtime.dispose()
                finally:
                    if self._live_sessions.get(live.id) is live:
                        del self._live_sessions[live.id]
            except BaseException as error:
                claimed.reject(error)
                return
            claimed.resolve(None)

        driven(_dispose())

        async def _finish() -> None:
            await claimed
            if not self._options.is_closing():
                self._options.broadcast_server_snapshot()

        return driven(_finish())

    def _watch(self, awaitable: Awaitable[None]) -> None:
        """pi's `void promise.catch(reportError)` for autonomously-driven work."""

        async def _run() -> None:
            try:
                await awaitable
            except Exception as error:
                self._options.report_error(error)

        tonio.spawn.without_tracking(_run())
