"""Remote session coordinator (port of pi coding-agent `client/remote-session.ts`).

Binds one `pidrei_client` session lease at a time, projects its transcript
through the pure reducer in `transcript.py`, and serializes mutations through
a lifecycle (`unbound`/`ready`/`busy`/`disposed`).

pi's async methods run synchronously to their first await, and the tests
depend on that (a `submit()` call must put the prompt request on the wire
before the caller awaits). The port mirrors pi's phasing with sync prologues
returning awaitables; operation bodies are spawned via `driven` so they make
progress whether or not the raced operation awaitable is consumed — pi's
promise semantics, needed when `dispose()` preempts an in-flight attachment.

pi has both a static and an instance `open`/`create` on `RemoteSession`;
Python cannot overload on dispatch, so the static factories are the module
functions `open_remote_session` / `create_remote_session`.

`Promise.race([running, disposeSignal])` maps to a race `Deferred` per
operation: a watcher mirrors the operation's outcome into it, and `dispose()`
rejects every outstanding race synchronously (pi's resolved dispose signal).
The losing operation keeps running as its own task, exactly like an unsettled
promise losing a JS race. (`tonio.select` would fit but leaks a
never-awaited-coroutine warning when it cancels a branch that has not started
yet — see TONIO_BUGS.md.) `AggregateError` maps to `ExceptionGroup` with pi's
messages.

Lifecycle states are identity-compared (`eq=False`), mirroring pi's object
literals in a `Set`; compare `status`/`operation` attributes, not instances.
"""

import inspect
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal, Self

import tonio.colored as tonio

from pidrei_client import (
    AcquireSessionOptions,
    ConnectionState,
    ConnectionStateChange,
    CreateSessionOptions,
    PiClient,
    SessionLease,
    Unsubscribe,
)
from pidrei_protocol import (
    ModelMetadata,
    ModelRef,
    ServerEvent,
    SessionMetadata,
    SessionPhase,
    SessionSnapshot,
    ThinkingLevel,
    TranscriptItem,
)

from .promise import Deferred, driven, rejected, resolved
from .transcript import (
    TranscriptState,
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)


type RemoteSessionOperation = Literal["open", "create", "submit", "abort", "setModel", "setThinking", "reconnect"]


@dataclass(slots=True, eq=False)
class RemoteSessionLifecycle:
    """Identity-compared lifecycle marker (pi's `{ status, operation }` literals)."""

    status: Literal["unbound", "ready", "busy", "disposed"]
    operation: RemoteSessionOperation | None = None


@dataclass(slots=True, frozen=True)
class RemoteSessionState:
    lifecycle: RemoteSessionLifecycle
    snapshot: SessionSnapshot | None
    transcript: list[TranscriptItem]


@dataclass(slots=True, frozen=True)
class CreateRemoteSessionOptions:
    cwd: str
    model: ModelRef | None = None
    thinking_level: ThinkingLevel | None = None


@dataclass(slots=True, frozen=True)
class RemoteSessionOptions:
    on_listener_error: Callable[[Exception], None] | None = None


class RemoteSessionDisposedError(Exception):
    def __init__(self) -> None:
        super().__init__("Remote session is disposed")


async def _settle_remote_session_disposal(cleanup: list[Awaitable[None]]) -> None:
    errors: list[Exception] = []
    for awaitable in [aw if isinstance(aw, Deferred) else driven(_as_coroutine(aw)) for aw in cleanup]:
        try:
            await awaitable
        except RemoteSessionDisposedError:
            continue
        except Exception as error:
            errors.append(error)
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise ExceptionGroup("Failed to dispose remote session", errors)


def _as_coroutine(awaitable: Awaitable[Any]) -> Coroutine[Any, Any, Any]:
    if inspect.iscoroutine(awaitable):
        return awaitable

    async def _await() -> Any:
        return await awaitable

    return _await()


def open_remote_session(
    client: PiClient, session_id: str, options: RemoteSessionOptions | None = None
) -> Awaitable[RemoteSession]:
    """pi's static `RemoteSession.open`: open a session, disposing on failure."""
    session = RemoteSession(client, options)
    return _finish_factory(session, session.open(session_id))


def create_remote_session(
    client: PiClient, create_options: CreateRemoteSessionOptions, options: RemoteSessionOptions | None = None
) -> Awaitable[RemoteSession]:
    """pi's static `RemoteSession.create`: create a session, disposing on failure."""
    session = RemoteSession(client, options)
    return _finish_factory(session, session.create(create_options))


async def _finish_factory(session: RemoteSession, operation: Awaitable[None]) -> RemoteSession:
    try:
        await operation
    except BaseException:
        await session.dispose()
        raise
    return session


class RemoteSession:
    def __init__(self, client: PiClient, options: RemoteSessionOptions | None = None) -> None:
        self._client = client
        self._on_listener_error = options.on_listener_error if options is not None else None
        self._lifecycle = RemoteSessionLifecycle(status="unbound")
        self._handle: SessionLease | None = None
        self._transcript: TranscriptState | None = None
        self._unsubscribe_snapshot: Unsubscribe | None = None
        self._unsubscribe_events: Unsubscribe | None = None
        self._listeners: set[Callable[[RemoteSessionState], None]] = set()
        self._pending_attachment_operations: set[Deferred] = set()
        self._active_operation_states: set[RemoteSessionLifecycle] = set()
        self._operation_races: set[Deferred] = set()
        self._dispose_promise: Deferred | None = None

    @property
    def id(self) -> str | None:
        return self._handle.id if self._handle is not None else None

    @property
    def state(self) -> RemoteSessionState:
        return RemoteSessionState(
            lifecycle=self._lifecycle,
            snapshot=self._transcript.snapshot if self._transcript is not None else None,
            transcript=select_transcript(self._transcript) if self._transcript is not None else [],
        )

    @property
    def snapshot(self) -> SessionSnapshot | None:
        return self._transcript.snapshot if self._transcript is not None else None

    @property
    def phase(self) -> SessionPhase | None:
        snapshot = self.snapshot
        return snapshot["phase"] if snapshot is not None else None

    @property
    def operation(self) -> RemoteSessionOperation | None:
        return self._lifecycle.operation if self._lifecycle.status == "busy" else None

    @property
    def models(self) -> list[ModelMetadata]:
        snapshot = self._client.snapshot
        return snapshot["models"] if snapshot is not None else []

    @property
    def sessions(self) -> list[SessionMetadata]:
        snapshot = self._client.snapshot
        return snapshot["sessions"] if snapshot is not None else []

    @property
    def connection_state(self) -> ConnectionState:
        return self._client.connection_state

    @property
    def disposed(self) -> bool:
        return self._lifecycle.status == "disposed"

    def subscribe(self, listener: Callable[[RemoteSessionState], None]) -> Unsubscribe:
        self._assert_not_disposed()
        self._listeners.add(listener)
        self._call_listener(listener, self.state)
        return lambda: self._listeners.discard(listener)

    def on_connection_state_change(self, listener: Callable[[ConnectionStateChange], None]) -> Unsubscribe:
        self._assert_not_disposed()
        return self._client.on_connection_state_change(listener)

    def open(self, session_id: str) -> Awaitable[None]:
        try:
            if self._handle is not None and self._handle.id == session_id and self._lifecycle.status == "ready":
                return resolved(None)
            return self._replace(
                "open",
                lambda: self._client.acquire_session(session_id, AcquireSessionOptions(mode="exclusive")),
            )
        except Exception as error:
            return rejected(error)

    def create(self, options: CreateRemoteSessionOptions) -> Awaitable[None]:
        try:
            return self._replace(
                "create",
                lambda: self._client.create_session(
                    CreateSessionOptions(cwd=options.cwd, model=options.model, thinking_level=options.thinking_level)
                ),
            )
        except Exception as error:
            return rejected(error)

    def submit(self, text: str) -> Awaitable[None]:
        try:
            normalized = text.strip()
            if not normalized:
                return resolved(None)
            self._assert_available()
            handle = self._require_handle()
            if self.phase not in ("idle", "turn"):
                phase = self.phase if self.phase is not None else "unknown"
                raise Exception(f"Session cannot accept input during {phase} phase")
            return self._run_operation(
                "submit",
                lambda: _discard_result(
                    handle.prompt(normalized) if self.phase == "idle" else handle.steer(normalized)
                ),
            )
        except Exception as error:
            return rejected(error)

    def abort(self) -> Awaitable[None]:
        try:
            preempting_submit = self._lifecycle.status == "busy" and self._lifecycle.operation == "submit"
            if preempting_submit:
                self._assert_not_disposed()
            else:
                self._assert_available()
            handle = self._require_handle()
            if self.phase == "idle" and not preempting_submit:
                return resolved(None)
            return self._run_operation("abort", lambda: _discard_result(handle.abort()), preempting_submit)
        except Exception as error:
            return rejected(error)

    def set_model(self, model: ModelRef) -> Awaitable[None]:
        try:
            return self._run_idle_operation(
                "setModel", "change model", lambda: _discard_result(self._require_handle().set_model(model))
            )
        except Exception as error:
            return rejected(error)

    def set_thinking(self, thinking_level: ThinkingLevel) -> Awaitable[None]:
        try:
            return self._run_idle_operation(
                "setThinking",
                "change thinking level",
                lambda: _discard_result(self._require_handle().set_thinking(thinking_level)),
            )
        except Exception as error:
            return rejected(error)

    def reconnect(self) -> Awaitable[None]:
        try:
            self._assert_available()
            session_id = self._require_handle().id
            return self._run_operation(
                "reconnect",
                lambda: self._track_attachment_operation(lambda: self._reconnect_attachment(session_id)),
            )
        except Exception as error:
            return rejected(error)

    def dispose(self) -> Awaitable[None]:
        if self._dispose_promise is not None:
            return self._dispose_promise
        handle = self._handle
        self._lifecycle = RemoteSessionLifecycle(status="disposed")
        # pi resolves its dispose signal here; the race Deferreds are that
        # signal's fan-out, rejected with the message pi's raced arm throws.
        for race in list(self._operation_races):
            race.reject(Exception("Remote session is disposed"))
        self._clear_subscriptions()
        self._handle = None
        self._transcript = None
        cleanup: list[Awaitable[None]] = list(self._pending_attachment_operations)
        if handle is not None:
            cleanup.append(handle.dispose())
        self._dispose_promise = driven(_settle_remote_session_disposal(cleanup))
        self._notify()
        self._listeners.clear()
        return self._dispose_promise

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.dispose()

    def _replace(
        self, operation: Literal["open", "create"], prepare: Callable[[], Awaitable[SessionLease]]
    ) -> Awaitable[None]:
        self._assert_available()
        if self._handle is not None and self.phase != "idle":
            phase = self.phase if self.phase is not None else "unavailable"
            raise Exception(f"Cannot {operation} a session while session is {phase}")
        return self._run_operation(
            operation,
            lambda: self._track_attachment_operation(lambda: self._prepare_replacement(operation, prepare)),
        )

    def _track_attachment_operation(self, run: Callable[[], Awaitable[None]]) -> Awaitable[None]:
        pending = driven(_as_coroutine(run()))
        self._pending_attachment_operations.add(pending)
        return self._finish_tracked_operation(pending)

    async def _finish_tracked_operation(self, pending: Deferred) -> None:
        try:
            await pending
        finally:
            self._pending_attachment_operations.discard(pending)

    def _prepare_replacement(
        self, operation: Literal["open", "create"], prepare: Callable[[], Awaitable[SessionLease]]
    ) -> Awaitable[None]:
        previous = self._handle
        preparing = prepare()
        return self._finish_replacement(operation, previous, preparing)

    async def _finish_replacement(
        self, operation: Literal["open", "create"], previous: SessionLease | None, preparing: Awaitable[SessionLease]
    ) -> None:
        next_handle = await preparing
        await self._assert_not_disposed_after_await(next_handle)
        snapshot = next_handle.snapshot
        if snapshot is None:
            await self._detach(next_handle)
            raise Exception(f"Session {next_handle.id} did not provide a snapshot")
        replacing = previous is not None and previous.id != next_handle.id and previous.attached
        if replacing and self.phase != "idle":
            await self._detach(next_handle)
            phase = self.phase if self.phase is not None else "unavailable"
            raise Exception(f"Cannot {operation} a session while session is {phase}")
        if replacing:
            try:
                await previous.detach()
            except Exception as error:
                try:
                    await self._detach(next_handle)
                except Exception as cleanup_error:
                    raise ExceptionGroup(
                        "Failed to replace remote session attachment", [error, cleanup_error]
                    ) from None
                raise
        await self._assert_not_disposed_after_await(next_handle)
        self._bind(next_handle, snapshot)

    def _run_idle_operation(
        self,
        operation: Literal["setModel", "setThinking"],
        description: str,
        run: Callable[[], Awaitable[None]],
    ) -> Awaitable[None]:
        self._assert_available()
        self._require_handle()
        if self.phase != "idle":
            phase = self.phase if self.phase is not None else "unavailable"
            raise Exception(f"Cannot {description} while session is {phase}")
        return self._run_operation(operation, run)

    def _run_operation(
        self, operation: RemoteSessionOperation, run: Callable[[], Awaitable[None]], preempt: bool = False
    ) -> Awaitable[None]:
        if preempt:
            self._assert_not_disposed()
        else:
            self._assert_available()
        previous = self._lifecycle
        busy = RemoteSessionLifecycle(status="busy", operation=operation)
        self._lifecycle = busy
        self._active_operation_states.add(busy)
        self._notify()
        running = run()
        running_deferred = running if isinstance(running, Deferred) else driven(_as_coroutine(running))
        race = Deferred()
        self._operation_races.add(race)

        async def _mirror_running() -> None:
            try:
                result = await running_deferred
            except Exception as error:
                race.reject(error)
            else:
                race.resolve(result)
            finally:
                self._operation_races.discard(race)

        tonio.spawn.without_tracking(_mirror_running())
        return self._finish_run_operation(previous, busy, race, preempt)

    async def _finish_run_operation(
        self, previous: RemoteSessionLifecycle, busy: RemoteSessionLifecycle, race: Deferred, preempt: bool
    ) -> None:
        try:
            await race
        finally:
            self._active_operation_states.discard(busy)
            if not self.disposed and self._lifecycle is busy:
                if preempt and previous in self._active_operation_states:
                    self._lifecycle = previous
                elif self._handle is not None:
                    self._lifecycle = RemoteSessionLifecycle(status="ready")
                else:
                    self._lifecycle = RemoteSessionLifecycle(status="unbound")
                self._notify()

    def _reconnect_attachment(self, session_id: str) -> Awaitable[None]:
        reconnecting = self._client.reconnect()
        return self._finish_reconnect_attachment(session_id, reconnecting)

    async def _finish_reconnect_attachment(self, session_id: str, reconnecting: Awaitable[Any]) -> None:
        await reconnecting
        handle = await self._client.acquire_session(session_id, AcquireSessionOptions(mode="exclusive"))
        await self._assert_not_disposed_after_await(handle)
        self._bind(handle)

    def _bind(self, handle: SessionLease, known_snapshot: SessionSnapshot | None = None) -> None:
        snapshot = known_snapshot if known_snapshot is not None else handle.snapshot
        if snapshot is None:
            raise Exception(f"Session {handle.id} did not provide a snapshot")
        self._clear_subscriptions()
        self._handle = handle
        self._transcript = create_transcript_state(snapshot)

        def on_snapshot(next_snapshot: SessionSnapshot) -> None:
            if self._transcript is None:
                return
            self._transcript = apply_transcript_snapshot(self._transcript, next_snapshot)
            self._notify()

        self._unsubscribe_snapshot = handle.subscribe(on_snapshot)
        self._unsubscribe_events = handle.on_event(self._handle_event)

    def _handle_event(self, event: ServerEvent) -> None:
        if event["type"] == "session_removed":
            self._clear_subscriptions()
            self._handle = None
            self._transcript = None
            if self._lifecycle.status != "busy":
                self._lifecycle = RemoteSessionLifecycle(status="unbound")
            self._notify()
            return
        if event["type"] != "session_progress" or self._transcript is None:
            return
        self._transcript = apply_transcript_progress(self._transcript, event["progress"])
        self._notify()

    def _notify(self) -> None:
        state = self.state
        for listener in list(self._listeners):
            self._call_listener(listener, state)

    def _call_listener(self, listener: Callable[[RemoteSessionState], None], state: RemoteSessionState) -> None:
        try:
            listener(state)
        except Exception as error:
            self._report_listener_error(error)

    def _report_listener_error(self, error: Exception) -> None:
        if self._on_listener_error is None:
            return
        try:
            self._on_listener_error(error)
        except Exception:
            # Diagnostics cannot affect session or transport state.
            pass

    def _clear_subscriptions(self) -> None:
        if self._unsubscribe_snapshot is not None:
            self._unsubscribe_snapshot()
        if self._unsubscribe_events is not None:
            self._unsubscribe_events()
        self._unsubscribe_snapshot = None
        self._unsubscribe_events = None

    def _require_handle(self) -> SessionLease:
        if self._handle is None:
            raise Exception("No remote session is attached")
        return self._handle

    def _assert_available(self) -> None:
        self._assert_not_disposed()
        if self._lifecycle.status == "busy":
            raise Exception(f"Remote session is busy with {self._lifecycle.operation}")

    def _assert_not_disposed(self) -> None:
        if self.disposed:
            raise Exception("Remote session is disposed")

    async def _assert_not_disposed_after_await(self, handle: SessionLease) -> None:
        if not self.disposed:
            return
        await self._detach(handle)
        raise RemoteSessionDisposedError()

    async def _detach(self, handle: SessionLease) -> None:
        await handle.dispose()


async def _discard_result(awaitable: Awaitable[Any]) -> None:
    """pi's `.then(() => undefined)` on session commands."""
    await awaitable
