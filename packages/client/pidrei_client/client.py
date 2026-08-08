"""PiClient with session-lease bookkeeping (port of pi client `client.ts`).

pi's public methods are async functions whose bodies run synchronously up to
the first await; the request/lease bookkeeping (frame sent at call time, lease
reserved before returning, `releasing` entered when `detach()` is called)
lives in that synchronous prefix. The port mirrors it with sync prologues that
return awaitables, spawning the awaited remainder on the tonio runtime so it
progresses whether or not the caller awaits — exactly like a JS promise chain.
Synchronous throws become rejected awaitables, as JS async functions would.
The one intentional narrowing: a pending detachment or cleanup reconciliation
makes `acquire_session` suspend before pi would have issued the follow-up
request; observable request ordering is unchanged.

pi's `static connect()` is `PiClient.open()` here — Python cannot overload the
instance method of the same name.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Self

import tonio.colored as tonio

from pidrei_protocol import (
    Command,
    CommandResult,
    FrameDecoderOptions,
    ProtocolValidationError,
    ServerEvent,
    ServerMessage,
    ServerSnapshot,
    SessionMetadata,
    encode_client_message,
)

from .connection import Connection, ConnectionOptions
from .errors import (
    PiClientDisposedError,
    PiDisconnectedError,
    PiServerError,
    PiSessionDetachedError,
    PiSessionOwnershipError,
)
from .promise import Deferred, rejected, resolved
from .session_handle import (
    AcquireSessionOptions,
    SessionHandle,
    SessionHandleCallbacks,
    SessionLeaseMode,
)
from .state import ClientState
from .types import ConnectionState, ConnectionStateChange, CreateSessionOptions, PiClientOptions, Unsubscribe


@dataclass(slots=True, frozen=True)
class _SessionLeaseToken:
    mode: SessionLeaseMode


@dataclass(slots=True, frozen=True)
class _PendingRequest:
    command: Command
    deferred: Deferred


class PiClient:
    def __init__(self, options: PiClientOptions) -> None:
        self._options = options
        self._state = ClientState(options.on_listener_error)
        self._connection = Connection(
            ConnectionOptions(
                transport_factory=options.transport_factory,
                max_frame_length=options.max_frame_length,
                on_handshake=self._state.apply_server_snapshot,
                on_message=self._handle_message,
                on_state_change=self._handle_connection_state_change,
            )
        )
        self._pending_requests: dict[str, _PendingRequest] = {}
        self._session_lease_counts: dict[str, int] = {}
        self._exclusive_session_leases: dict[str, _SessionLeaseToken] = {}
        self._session_lease_generations: dict[str, int] = {}
        self._session_attachments: dict[str, Deferred] = {}
        self._session_detachments: dict[str, Deferred] = {}
        self._session_cleanup_required: set[str] = set()
        self._session_reconciliations: dict[str, Deferred] = {}
        self._connection_state_listeners: set[Callable[[ConnectionStateChange], None]] = set()
        self._request_sequence = 0
        self._disposed = False
        self._dispose_deferred: Deferred | None = None

    @property
    def disposed(self) -> bool:
        return self._disposed

    @property
    def connection_state(self) -> ConnectionState:
        return self._connection.state

    @property
    def connected(self) -> bool:
        return self._connection.state == "connected"

    @property
    def snapshot(self) -> ServerSnapshot | None:
        return self._state.snapshot

    @classmethod
    async def open(cls, options: PiClientOptions) -> PiClient:
        client = cls(options)
        try:
            await client.connect()
            return client
        except BaseException:
            await client.dispose()
            raise

    def connect(self) -> Awaitable[ServerSnapshot]:
        if self._disposed:
            return rejected(PiClientDisposedError())
        if self._connection.state == "disconnected":
            self._state.reset()
        return self._connection.connect()

    def reconnect(self) -> Awaitable[ServerSnapshot]:
        return self.connect()

    def disconnect(self, reason: str = "Client disconnected") -> None:
        self._connection.disconnect(reason)

    def subscribe(self, listener: Callable[[ServerSnapshot], None]) -> Unsubscribe:
        self._assert_not_disposed()
        return self._state.subscribe(listener)

    def on_event(self, listener: Callable[[ServerEvent], None]) -> Unsubscribe:
        self._assert_not_disposed()
        return self._state.on_event(listener)

    def on_connection_state_change(self, listener: Callable[[ConnectionStateChange], None]) -> Unsubscribe:
        self._assert_not_disposed()
        self._connection_state_listeners.add(listener)
        return lambda: self._connection_state_listeners.discard(listener)

    def list_sessions(self) -> Awaitable[list[SessionMetadata]]:
        result = self._request({"command": "list"})

        async def _finish() -> list[SessionMetadata]:
            return (await result)["sessions"]

        return _finish()

    def create_session(self, options: CreateSessionOptions | None = None) -> Awaitable[SessionHandle]:
        command: Command = {"command": "create"}
        if options is not None:
            if options.cwd is not None:
                command["cwd"] = options.cwd
            if options.name is not None:
                command["name"] = options.name
            if options.model is not None:
                command["model"] = options.model
            if options.thinking_level is not None:
                command["thinkingLevel"] = options.thinking_level
        result = self._request(command)

        async def _finish() -> SessionHandle:
            session = (await result)["session"]
            token = self._reserve_session_lease(session["id"], "exclusive")
            return self._create_session_lease(session["id"], token)

        return _finish()

    def attach_session(self, session_id: str) -> Awaitable[SessionHandle]:
        return self.acquire_session(session_id, AcquireSessionOptions(mode="shared"))

    def acquire_session(self, session_id: str, options: AcquireSessionOptions) -> Awaitable[SessionHandle]:
        try:
            self._assert_not_disposed()
            token = self._reserve_session_lease(session_id, options.mode)
        except Exception as error:
            return rejected(error)
        # pi's synchronous prologue reaches the attach request when nothing is
        # pending for the session; mirror it so the frame goes out at call time.
        attachment: Deferred | None = None
        if (
            self._session_detachments.get(session_id) is None
            and session_id not in self._session_cleanup_required
            and not self._state.is_session_attached(session_id)
        ):
            attachment = self._session_attachments.get(session_id)
            if attachment is None:
                attachment = self._start_attach_session(session_id)
                self._session_attachments[session_id] = attachment
        return self._finish_acquire_session(session_id, token, attachment)

    async def _finish_acquire_session(
        self, session_id: str, token: _SessionLeaseToken, attachment: Deferred | None
    ) -> SessionHandle:
        try:
            if attachment is not None:
                await self._await_attachment(session_id, attachment)
            else:
                detachment = self._session_detachments.get(session_id)
                if detachment is not None:
                    await detachment.wait_silenced()
                reconciled = (
                    await self._reconcile_session_cleanup(session_id)
                    if session_id in self._session_cleanup_required
                    else False
                )
                if reconciled or not self._state.is_session_attached(session_id):
                    pending = self._session_attachments.get(session_id)
                    if pending is None:
                        pending = self._start_attach_session(session_id)
                        self._session_attachments[session_id] = pending
                    await self._await_attachment(session_id, pending)
            return self._create_session_lease(session_id, token)
        except Exception:
            self._release_session_lease(session_id, token)
            raise

    async def _await_attachment(self, session_id: str, attachment: Deferred) -> None:
        try:
            await attachment.wait()
        finally:
            if self._session_attachments.get(session_id) is attachment:
                del self._session_attachments[session_id]

    def _start_attach_session(self, session_id: str) -> Deferred:
        # pi's `#attachSession`: forget the stale snapshot and issue the attach
        # request synchronously; restore the snapshot if the request fails.
        attachment = Deferred()
        previous = self._state.forget_session_snapshot(session_id)
        result = self._request({"command": "attach", "sessionId": session_id})

        async def _drive() -> None:
            try:
                await result
            except Exception as error:
                if previous is not None:
                    self._state.restore_session_snapshot(previous)
                attachment.reject(error)
                return
            attachment.resolve(None)

        tonio.spawn.without_tracking(_drive())
        return attachment

    def _request(self, command: Command) -> Awaitable[CommandResult]:
        if self._disposed:
            return rejected(PiClientDisposedError())
        if not self.connected:
            return rejected(PiDisconnectedError())
        self._request_sequence += 1
        request_id = f"request-{self._request_sequence}"
        deferred = Deferred()
        self._pending_requests[request_id] = _PendingRequest(command=command, deferred=deferred)
        try:
            frame = encode_client_message(
                {"type": "request", "id": request_id, "request": command},
                FrameDecoderOptions(max_frame_length=self._connection.max_frame_length),
            )
        except Exception as error:
            pending = self._take_pending_request(request_id)
            if pending is not None:
                pending.deferred.reject(error)
            return deferred.wait()
        self._connection.send(frame)
        return deferred.wait()

    def _create_session_lease(self, session_id: str, token: _SessionLeaseToken) -> SessionHandle:
        generation = self._session_lease_generations.get(session_id, 0)
        self._session_lease_generations[session_id] = generation
        state = "active"
        release_deferred: Deferred | None = None

        def refresh_state() -> None:
            nonlocal state
            if state in ("active", "releasing") and self._session_lease_generations.get(session_id) != generation:
                state = "invalidated"

        def is_active() -> bool:
            refresh_state()
            return state == "active" and self._state.is_session_attached(session_id)

        def assert_active() -> None:
            self._assert_not_disposed()
            if not self.connected:
                raise PiDisconnectedError()
            if not is_active():
                raise PiSessionDetachedError(session_id)

        def release(relinquish_on_failure: bool) -> Awaitable[None]:
            nonlocal state, release_deferred
            refresh_state()
            if state in ("released", "invalidated"):
                return resolved(None)
            if release_deferred is not None:
                return release_deferred.wait()
            try:
                assert_active()
            except Exception as error:
                return rejected(error)
            state = "releasing"
            deferred = Deferred()
            release_deferred = deferred
            count = self._session_lease_counts.get(session_id, 0)
            if count > 1:
                # pi's non-final path is fully synchronous.
                self._release_session_lease(session_id, token)
                state = "released"
                deferred.resolve(None)
                return deferred.wait()
            detach_result = self._request({"command": "detach", "sessionId": session_id})
            detachment = Deferred()
            self._session_detachments[session_id] = detachment

            async def _finish() -> None:
                try:
                    try:
                        await detach_result
                    except Exception as error:
                        detachment.reject(error)
                        raise
                    detachment.resolve(None)
                    self._release_session_lease(session_id, token)
                finally:
                    if self._session_detachments.get(session_id) is detachment:
                        del self._session_detachments[session_id]

            async def _drive() -> None:
                nonlocal state, release_deferred
                try:
                    await _finish()
                except Exception as error:
                    refresh_state()
                    if state == "invalidated":
                        deferred.resolve(None)
                        return
                    if relinquish_on_failure:
                        self._release_session_lease(session_id, token)
                        self._session_cleanup_required.add(session_id)
                        state = "released"
                    else:
                        state = "active"
                        release_deferred = None
                    deferred.reject(error)
                    return
                state = "released"
                deferred.resolve(None)

            tonio.spawn.without_tracking(_drive())
            return deferred.wait()

        def subscribe(listener: Callable[..., None]) -> Unsubscribe:
            assert_active()
            return self._state.subscribe_session(
                session_id, lambda snapshot: listener(snapshot) if is_active() else None
            )

        def on_event(listener: Callable[..., None]) -> Unsubscribe:
            assert_active()
            return self._state.on_session_event(
                session_id,
                lambda event: listener(event) if is_active() or event["type"] == "session_removed" else None,
            )

        def request(command: Command) -> Awaitable[CommandResult]:
            assert_active()
            return self._request(command)

        callbacks = SessionHandleCallbacks(
            is_attached=is_active,
            get_snapshot=lambda: self._state.get_session_snapshot(session_id) if is_active() else None,
            subscribe=subscribe,
            on_event=on_event,
            detach=lambda: release(False),
            dispose=lambda: release(True),
            request=request,
        )
        return SessionHandle(session_id, callbacks)

    def _handle_message(self, message: ServerMessage) -> None:
        if message["type"] == "event":
            event = message["event"]
            if event["type"] == "session_removed":
                self._invalidate_session_leases(event["sessionId"])
            self._state.apply_event(event)
            return
        pending = self._take_pending_request(message["id"])
        if pending is None:
            self._connection.fail(ProtocolValidationError("Response has no matching request"))
            return
        if not message["ok"]:
            pending.deferred.reject(PiServerError(message["error"]))
            return
        result = message["result"]
        if result["command"] != pending.command["command"]:
            error = ProtocolValidationError(
                f"Response command {result['command']} does not match {pending.command['command']}"
            )
            pending.deferred.reject(error)
            self._connection.fail(error)
            return
        self._state.apply_result(result)
        pending.deferred.resolve(result)

    def _handle_connection_state_change(self, change: ConnectionStateChange) -> None:
        if change.state == "disconnected":
            self._state.clear_attachments()
            self._invalidate_all_session_leases()
            self._reject_pending_requests(change.error if change.error is not None else PiDisconnectedError())
        self._notify_connection_state_listeners(change)

    def _take_pending_request(self, request_id: str) -> _PendingRequest | None:
        return self._pending_requests.pop(request_id, None)

    def _reject_pending_requests(self, error: Exception) -> None:
        requests = list(self._pending_requests.values())
        self._pending_requests.clear()
        for request in requests:
            request.deferred.reject(error)

    def dispose(self) -> Awaitable[None]:
        if self._dispose_deferred is not None:
            return self._dispose_deferred.wait()
        self._disposed = True
        deferred = Deferred()
        deferred.resolve(None)
        self._dispose_deferred = deferred
        error = PiClientDisposedError()
        self._reject_pending_requests(error)
        self._connection.disconnect(error)
        self._state.dispose()
        self._invalidate_all_session_leases()
        self._connection_state_listeners.clear()
        return deferred.wait()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.dispose()

    def _assert_not_disposed(self) -> None:
        if self._disposed:
            raise PiClientDisposedError()

    async def _reconcile_session_cleanup(self, session_id: str) -> bool:
        if session_id not in self._session_cleanup_required:
            return False
        reconciliation = self._session_reconciliations.get(session_id)
        if reconciliation is None:
            reconciliation = Deferred()
            self._session_reconciliations[session_id] = reconciliation
            result = self._request({"command": "detach", "sessionId": session_id})

            async def _drive() -> None:
                try:
                    await result
                    self._session_cleanup_required.discard(session_id)
                    reconciliation.resolve(None)
                except Exception as error:
                    reconciliation.reject(error)
                finally:
                    self._session_reconciliations.pop(session_id, None)

            tonio.spawn.without_tracking(_drive())
        await reconciliation.wait()
        return True

    def _reserve_session_lease(self, session_id: str, mode: SessionLeaseMode) -> _SessionLeaseToken:
        count = self._session_lease_counts.get(session_id, 0)
        if mode == "exclusive" and count > 0:
            raise PiSessionOwnershipError(session_id, f"Session {session_id} already has an active lease")
        if mode == "shared" and session_id in self._exclusive_session_leases:
            raise PiSessionOwnershipError(session_id, f"Session {session_id} has an exclusive lease")
        token = _SessionLeaseToken(mode=mode)
        self._session_lease_counts[session_id] = count + 1
        if mode == "exclusive":
            self._exclusive_session_leases[session_id] = token
        return token

    def _release_session_lease(self, session_id: str, token: _SessionLeaseToken) -> None:
        count = self._session_lease_counts.get(session_id, 0)
        if count <= 1:
            self._session_lease_counts.pop(session_id, None)
        else:
            self._session_lease_counts[session_id] = count - 1
        if self._exclusive_session_leases.get(session_id) is token:
            del self._exclusive_session_leases[session_id]

    def _invalidate_session_leases(self, session_id: str) -> None:
        self._session_lease_counts.pop(session_id, None)
        self._exclusive_session_leases.pop(session_id, None)
        self._session_cleanup_required.discard(session_id)
        self._session_lease_generations[session_id] = self._session_lease_generations.get(session_id, 0) + 1

    def _invalidate_all_session_leases(self) -> None:
        for session_id in list(self._session_lease_counts.keys()):
            self._invalidate_session_leases(session_id)
        self._session_cleanup_required.clear()

    def _notify_connection_state_listeners(self, change: ConnectionStateChange) -> None:
        for listener in list(self._connection_state_listeners):
            try:
                listener(change)
            except Exception as error:
                self._report_listener_error(error)

    def _report_listener_error(self, error: Exception) -> None:
        if self._options.on_listener_error is None:
            return
        try:
            self._options.on_listener_error(error)
        except Exception:
            # Diagnostics cannot affect protocol or transport state.
            pass
