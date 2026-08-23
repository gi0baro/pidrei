"""Protocol server core (port of pi server `server.ts`).

pi's observable semantics live in the synchronous prefix of its async
methods, so the state-entangled ones are sync prologues returning awaitables:
`start`/`close` memoize and flip flags at call time, `_send_message` queues
the frame on the connection before its first await (wire ordering),
`_fail_protocol` enters "closing" immediately (the receive loop checks it
between messages of one chunk), and `_disconnect` marks the connection
synchronously. `MAX_TIMER_DELAY_MS` keeps Node's timer-delay bound for
byte-identical validation errors.
"""

import uuid
from collections.abc import Awaitable, Coroutine
from typing import Any, Self

import tonio.colored as tonio

from pidrei_protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    PROTOCOL_VERSION,
    ClientMessage,
    ClientMessageDecoder,
    FrameDecoderOptions,
    ProtocolError,
    ProtocolValidationError,
    ServerMessage,
    encode_server_message,
    is_supported_protocol_version,
)

from .connection import (
    ByteConnection,
    ByteConnectionHandler,
    ConnectionState,
    is_terminal_connection,
)
from .errors import INTERNAL_SERVER_ERROR_MESSAGE, NOT_IMPLEMENTED_MESSAGE, InternalServerError, PiServerError
from .listener import PiServerListener
from .promise import Deferred, all_settled, driven, gather, rejected, resolved
from .sessions import LiveSessionManager, LiveSessionManagerOptions
from .snapshots import ServerSnapshotPublisher, ServerSnapshotPublisherOptions
from .timers import Timer
from .types import PiServerOptions, PiServerService


DEFAULT_HANDSHAKE_TIMEOUT_MS = 5_000
MAX_UINT32 = 0xFFFF_FFFF
MAX_TIMER_DELAY_MS = 2_147_483_647


class PiServer:
    def __init__(self, service: PiServerService, options: PiServerOptions) -> None:
        max_frame_length, handshake_timeout_ms = _resolve_options(options)
        self._listeners = options.listeners
        self.id = options.server_id if options.server_id is not None else str(uuid.uuid4())
        self._max_frame_length = max_frame_length
        self._handshake_timeout_ms = handshake_timeout_ms
        self._on_error = options.on_error
        self._connections: set[ConnectionState] = set()
        self._closing = False
        self._close_deferred: Deferred | None = None
        self._start_deferred: Deferred | None = None
        self._started = False
        self._sessions = LiveSessionManager(
            LiveSessionManagerOptions(
                service=service,
                is_closing=lambda: self._closing,
                send_message=self._send_message,
                close_connection=self._close_connection,
                disconnect=self._disconnect,
                broadcast_server_snapshot=lambda: self._snapshots.broadcast(),
                report_error=self._report_error,
            )
        )
        self._snapshots = ServerSnapshotPublisher(
            ServerSnapshotPublisherOptions(
                server_id=self.id,
                service=service,
                connections=self._connections,
                is_closing=lambda: self._closing,
                list_sessions=self._sessions.list_metadata,
                send_message=self._send_message,
                report_error=self._report_error,
            )
        )

    @property
    def addresses(self) -> list[str]:
        return [listener.address for listener in self._listeners if listener.address is not None]

    def start(self) -> Awaitable[Self]:
        if self._started:
            return rejected(Exception("PiServer is already started"))
        if self._start_deferred is not None:
            return rejected(Exception("PiServer is already starting"))
        if self._closing:
            return rejected(Exception("PiServer is closing or closed"))
        self._start_deferred = driven(self._start_internal())
        return self._start_deferred

    async def _start_internal(self) -> Self:
        started: list[PiServerListener] = []
        try:
            for listener in self._listeners:
                await listener.start(self.accept)
                started.append(listener)
            self._started = True
            return self
        except Exception:
            self._closing = True
            await all_settled([listener.close() for listener in started])
            await self._close_server_state()
            raise
        finally:
            self._start_deferred = None

    def accept(self, connection: ByteConnection) -> ByteConnectionHandler:
        if self._closing:
            tonio.spawn.without_tracking(self._close_connection(connection))
            return ByteConnectionHandler(
                on_data=lambda chunk: None,
                on_close=lambda: None,
                on_error=self._report_error,
            )

        # The late-bound `state` is safe: the timer task cannot fire before
        # this synchronous method binds it below.
        handshake_timeout = Timer(
            self._handshake_timeout_ms,
            lambda: self._fail_protocol(state, {"code": "invalid_request", "message": "Handshake timeout"}),
        )
        state = ConnectionState(
            id=str(uuid.uuid4()),
            connection=connection,
            decoder=ClientMessageDecoder(FrameDecoderOptions(max_frame_length=self._max_frame_length)),
            handshake_timeout=handshake_timeout,
        )
        self._connections.add(state)

        return ByteConnectionHandler(
            on_data=lambda chunk: self._receive(state, chunk),
            on_close=lambda: self._transport_closed(state),
            on_error=lambda error: self._handle_connection_error(state, error),
        )

    def close(self) -> Awaitable[None]:
        if self._close_deferred is not None:
            return self._close_deferred
        self._closing = True
        self._close_deferred = driven(self._close_internal())
        return self._close_deferred

    async def _close_internal(self) -> None:
        starting = self._start_deferred
        if starting is not None:
            await starting.wait_silenced()
        try:
            await gather([listener.close() for listener in self._listeners])
        finally:
            await self._close_server_state()
            self._started = False

    def _receive(self, state: ConnectionState, chunk: bytes) -> None:
        if is_terminal_connection(state):
            return
        try:
            messages = state.decoder.push(chunk)
        except Exception as error:
            self._fail_protocol(state, self._to_protocol_error(error))
            return
        for message in messages:
            if is_terminal_connection(state):
                return
            self._dispatch_message(state, message)

    def _dispatch_message(self, state: ConnectionState, message: ClientMessage) -> None:
        if state.stage == "awaitingHello":
            if message["type"] != "hello":
                self._fail_protocol(
                    state, {"code": "invalid_request", "message": "The first client message must be hello"}
                )
                return
            state.stage = "handshaking"
            handshake = Deferred()
            state.handshake = handshake
            finishing = self._finish_handshake(state, message)

            async def _drive_handshake() -> None:
                try:
                    await finishing
                except Exception as error:
                    await self._fail_protocol(state, self._to_protocol_error(error))
                handshake.resolve(None)

            tonio.spawn.without_tracking(_drive_handshake())
            return

        if message["type"] == "hello":
            self._fail_protocol(
                state, {"code": "invalid_request", "message": "hello may only be sent as the first message"}
            )
            return

        if state.stage == "ready":
            tonio.spawn.without_tracking(self._handle_request(state, message))
            return
        if state.stage != "handshaking":
            return
        handshake = state.handshake
        if handshake is None:
            return

        async def _deliver_after_handshake() -> None:
            await handshake.wait_silenced()
            if state.stage == "ready" and not state.disconnected:
                await self._handle_request(state, message)

        tonio.spawn.without_tracking(_deliver_after_handshake())

    def _finish_handshake(self, state: ConnectionState, hello: ClientMessage) -> Awaitable[None]:
        if not is_supported_protocol_version(hello["version"]):
            return self._fail_protocol(
                state,
                {
                    "code": "version",
                    "message": f"Unsupported protocol version {hello['version']}; expected {PROTOCOL_VERSION}",
                },
            )
        return driven(self._complete_handshake(state))

    async def _complete_handshake(self, state: ConnectionState) -> None:
        snapshot = await self._snapshots.get()
        if self._closing or state.disconnected or state.stage != "handshaking" or state.connection.closed:
            return
        sent = await self._send_message(
            state,
            {"type": "hello", "version": PROTOCOL_VERSION, "connectionId": state.id, "snapshot": snapshot},
        )
        if sent and not state.disconnected and state.stage == "handshaking":
            state.handshake_complete = True
            state.stage = "ready"
            state.handshake_timeout.cancel()
            if snapshot["revision"] != self._snapshots.current_revision:
                current = await self._snapshots.get()
                await self._send_message(
                    state, {"type": "event", "event": {"type": "server_snapshot", "snapshot": current}}
                )

    async def _handle_request(self, state: ConnectionState, envelope: ClientMessage) -> None:
        try:
            result = await self._sessions.execute_command(state, envelope["request"])
        except Exception as error:
            await self._send_message(
                state, {"type": "response", "id": envelope["id"], "ok": False, "error": self._to_protocol_error(error)}
            )
            return
        await self._send_message(state, {"type": "response", "id": envelope["id"], "ok": True, "result": result})

    def _transport_closed(self, connection: ConnectionState) -> None:
        if not connection.disconnected and connection.stage != "closing":
            try:
                connection.decoder.end()
            except Exception as error:
                self._report_error(error)
        self._disconnect(connection)

    def _handle_connection_error(self, state: ConnectionState, error: Exception) -> None:
        self._report_error(error)

        async def _finish() -> None:
            await self._close_connection(state.connection)
            await self._disconnect(state)

        tonio.spawn.without_tracking(_finish())

    def _disconnect(self, connection: ConnectionState) -> Awaitable[None]:
        if connection.disconnected:
            return resolved(None)
        handshake_complete = connection.handshake_complete
        connection.disconnected = True
        connection.stage = "closed"
        connection.handshake_timeout.cancel()
        self._connections.discard(connection)

        async def _finish() -> None:
            await self._sessions.disconnect(connection)
            if not self._closing and handshake_complete:
                self._snapshots.broadcast()

        return driven(_finish())

    def _send_message(self, connection: ConnectionState, message: ServerMessage) -> Coroutine[Any, Any, bool]:
        """Queue the frame now (wire ordering is decided here, synchronously)
        and return the coroutine that waits for it to go out. Callers that
        void the result (pi's `void this.sendMessage(...)`) spawn it; the
        connection's writer task already holds the frame, so nothing is lost
        if the coroutine runs later."""

        async def _not_sent() -> bool:
            return False

        if connection.disconnected or connection.connection.closed:
            return _not_sent()

        async def _fail() -> bool:
            await self._close_connection(connection.connection)
            await self._disconnect(connection)
            return False

        try:
            frame = encode_server_message(message, FrameDecoderOptions(max_frame_length=self._max_frame_length))
        except Exception as error:
            self._report_error(error)
            return _fail()
        try:
            sending = connection.connection.send(frame)
        except Exception as error:
            self._report_error(error)
            return _fail()

        async def _finish() -> bool:
            try:
                await sending
            except Exception as error:
                self._report_error(error)
                return await _fail()
            return True

        return _finish()

    def _fail_protocol(self, connection: ConnectionState, error: ProtocolError) -> Awaitable[None]:
        if connection.disconnected or connection.stage == "closing" or connection.stage == "closed":
            return resolved(None)
        connection.stage = "closing"
        connection.handshake_timeout.cancel()
        message: ServerMessage = {"type": "hello_error", "error": error}
        final_frame: bytes | None = None
        try:
            final_frame = encode_server_message(message, FrameDecoderOptions(max_frame_length=self._max_frame_length))
        except Exception as encode_error:
            self._report_error(encode_error)

        async def _finish() -> None:
            await self._close_connection(connection.connection, final_frame)
            await self._disconnect(connection)

        return driven(_finish())

    async def _close_server_state(self) -> None:
        connections = list(self._connections)
        for connection in connections:
            connection.stage = "closing"
            connection.handshake_timeout.cancel()
        await gather([self._close_connection(connection.connection) for connection in connections])
        await gather([self._disconnect(connection) for connection in connections])

        await self._sessions.close()
        self._connections.clear()

    async def _close_connection(self, connection: ByteConnection, final_chunk: bytes | None = None) -> None:
        try:
            await connection.close(final_chunk)
        except Exception as error:
            self._report_error(error)

    def _to_protocol_error(self, error: BaseException) -> ProtocolError:
        if isinstance(error, InternalServerError):
            self._report_error(error.cause)
            return {"code": "internal_error", "message": INTERNAL_SERVER_ERROR_MESSAGE}
        if isinstance(error, PiServerError):
            if error.code == "not_implemented":
                return {"code": "not_implemented", "message": NOT_IMPLEMENTED_MESSAGE}
            if error.details is None:
                return {"code": error.code, "message": str(error)}
            return {"code": error.code, "message": str(error), "details": error.details}
        if isinstance(error, ProtocolValidationError):
            return {"code": "invalid_request", "message": str(error)}
        self._report_error(error)
        return {"code": "internal_error", "message": INTERNAL_SERVER_ERROR_MESSAGE}

    def _report_error(self, error: object) -> None:
        if self._on_error is None:
            return
        try:
            self._on_error(error if isinstance(error, Exception) else Exception(str(error)))
        except Exception:
            # Error observers cannot affect server state.
            pass


def _resolve_options(options: PiServerOptions) -> tuple[int, int]:
    if not isinstance(options.listeners, list):
        raise TypeError("PiServer listeners must be an array")
    if options.server_id == "":
        raise TypeError("PiServer serverId must not be empty")
    max_frame_length = options.max_frame_length if options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
    if (
        isinstance(max_frame_length, bool)
        or not isinstance(max_frame_length, int)
        or max_frame_length <= 0
        or max_frame_length > MAX_UINT32
    ):
        raise TypeError(f"PiServer maxFrameLength must be an integer between 1 and {MAX_UINT32}")
    handshake_timeout_ms = (
        options.handshake_timeout_ms if options.handshake_timeout_ms is not None else DEFAULT_HANDSHAKE_TIMEOUT_MS
    )
    if (
        isinstance(handshake_timeout_ms, bool)
        or not isinstance(handshake_timeout_ms, int)
        or handshake_timeout_ms <= 0
        or handshake_timeout_ms > MAX_TIMER_DELAY_MS
    ):
        raise TypeError(f"PiServer handshakeTimeoutMs must be an integer between 1 and {MAX_TIMER_DELAY_MS}")
    return max_frame_length, handshake_timeout_ms
