"""Generation-guarded connection state machine (port of pi client `connection.ts`).

pi's `connect()` is a plain method: it transitions to "connecting"
synchronously and returns the handshake promise, with the transport opened in
the background. The port keeps that shape — `connect()` is a sync prologue
returning an awaitable — because listeners synchronously reconnecting from a
disconnection callback depend on it. Lifecycle objects are compared by
identity to detect reentrant transitions made by callbacks mid-dispatch,
exactly like pi's tagged-union objects.
"""

import dataclasses
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import tonio.colored as tonio

from pidrei_protocol import (
    DEFAULT_MAX_FRAME_LENGTH,
    PROTOCOL_VERSION,
    FrameDecoderOptions,
    ProtocolValidationError,
    ServerMessage,
    ServerMessageDecoder,
    ServerSnapshot,
    encode_client_message,
)

from .errors import PiDisconnectedError, PiServerError, to_disconnected_error
from .promise import Deferred, rejected
from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers
from .types import ConnectionState, ConnectionStateChange


MAX_UINT32 = 0xFFFF_FFFF


@dataclass(slots=True)
class _Lifecycle:
    state: ConnectionState
    id: int = 0
    decoder: ServerMessageDecoder | None = None
    transport: ByteTransport | None = None
    handshake: Deferred | None = None


@dataclass(slots=True, frozen=True)
class ConnectionOptions:
    transport_factory: ByteTransportFactory
    on_handshake: Callable[[ServerSnapshot], None]
    on_message: Callable[[ServerMessage], None]
    on_state_change: Callable[[ConnectionStateChange], None]
    max_frame_length: int | None = None


class Connection:
    def __init__(self, options: ConnectionOptions) -> None:
        self._options = options
        max_frame_length = (
            options.max_frame_length if options.max_frame_length is not None else DEFAULT_MAX_FRAME_LENGTH
        )
        if (
            isinstance(max_frame_length, bool)
            or not isinstance(max_frame_length, int)
            or max_frame_length <= 0
            or max_frame_length > MAX_UINT32
        ):
            raise TypeError(f"PiClient maxFrameLength must be an integer between 1 and {MAX_UINT32}")
        self._max_frame_length = max_frame_length
        self._lifecycle = _Lifecycle(state="disconnected")
        self._sequence = 0

    @property
    def state(self) -> ConnectionState:
        return self._lifecycle.state

    @property
    def max_frame_length(self) -> int:
        return self._max_frame_length

    def connect(self) -> Awaitable[ServerSnapshot]:
        if self._lifecycle.state != "disconnected":
            return rejected(PiDisconnectedError(f"PiClient is already {self._lifecycle.state}"))
        self._sequence += 1
        connection_id = self._sequence
        handshake = Deferred()
        self._lifecycle = _Lifecycle(
            state="connecting",
            id=connection_id,
            decoder=ServerMessageDecoder(FrameDecoderOptions(max_frame_length=self._max_frame_length)),
            handshake=handshake,
        )
        self._options.on_state_change(ConnectionStateChange(state="connecting"))
        handlers = ByteTransportHandlers(
            on_data=lambda chunk: self._handle_data(connection_id, chunk),
            on_close=lambda: self._handle_close() if self._is_current(connection_id) else None,
            on_error=lambda error: (
                self._fail_and_close(to_disconnected_error(error)) if self._is_current(connection_id) else None
            ),
        )
        tonio.spawn.without_tracking(self._open_transport(connection_id, handlers))
        return handshake.wait()

    def disconnect(self, reason: str | Exception = "Client disconnected") -> None:
        if self._lifecycle.state == "disconnected":
            return
        self._fail_and_close(PiDisconnectedError(reason) if isinstance(reason, str) else reason)

    def fail(self, error: Exception) -> None:
        self._fail_and_close(error)

    def send(self, frame: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state != "connected":
            raise PiDisconnectedError()
        transport = lifecycle.transport
        assert transport is not None
        try:
            sending = transport.send(frame)
        except Exception as error:
            self._fail_and_close(to_disconnected_error(error))
            return
        tonio.spawn.without_tracking(self._watch_send(transport, sending))

    async def _watch_send(self, transport: ByteTransport, sending: Awaitable[None]) -> None:
        try:
            await sending
        except Exception as error:
            current = self._lifecycle
            if current.state != "disconnected" and current.transport is transport:
                self._fail_and_close(to_disconnected_error(error))

    async def _open_transport(self, connection_id: int, handlers: ByteTransportHandlers) -> None:
        try:
            transport = await self._options.transport_factory(handlers)
        except Exception as error:
            if self._is_current(connection_id):
                self._fail(to_disconnected_error(error))
            return
        lifecycle = self._lifecycle
        if lifecycle.state != "connecting" or lifecycle.id != connection_id:
            transport.close()
            return
        self._lifecycle = dataclasses.replace(lifecycle, transport=transport)
        try:
            await transport.send(
                encode_client_message(
                    {"type": "hello", "version": PROTOCOL_VERSION},
                    FrameDecoderOptions(max_frame_length=self._max_frame_length),
                )
            )
        except Exception as error:
            if self._is_current(connection_id):
                self._fail_and_close(to_disconnected_error(error))

    def _handle_data(self, connection_id: int, chunk: bytes) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected" or lifecycle.id != connection_id:
            return
        if lifecycle.state == "connecting" and lifecycle.transport is None:
            self._fail_and_close(ProtocolValidationError("Received server data before the client hello was sent"))
            return
        decoder = lifecycle.decoder
        assert decoder is not None
        try:
            messages = decoder.push(chunk)
        except Exception as error:
            self._fail_and_close(error)
            return
        for message in messages:
            if self._lifecycle.state == "disconnected":
                return
            self._handle_message(message)

    def _handle_message(self, message: ServerMessage) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "connecting":
            if message["type"] == "hello_error":
                self._fail_and_close(PiServerError(message["error"]))
                return
            if message["type"] != "hello":
                self._fail_and_close(ProtocolValidationError("Expected server hello as first message"))
                return
            if lifecycle.transport is None:
                self._fail_and_close(ProtocolValidationError("Received server hello before the client hello was sent"))
                return
            connected = _Lifecycle(
                state="connected",
                id=lifecycle.id,
                decoder=lifecycle.decoder,
                transport=lifecycle.transport,
                handshake=lifecycle.handshake,
            )
            self._lifecycle = connected
            snapshot = message["snapshot"]
            try:
                self._options.on_handshake(snapshot)
            except Exception as error:
                if self._lifecycle is connected:
                    self._fail_and_close(error)
                return
            if self._lifecycle is not connected:
                return
            self._options.on_state_change(ConnectionStateChange(state="connected"))
            if self._lifecycle is not connected:
                return
            self._lifecycle = dataclasses.replace(connected, handshake=None)
            handshake = lifecycle.handshake
            assert handshake is not None
            handshake.resolve(snapshot)
            return
        if lifecycle.state != "connected":
            return
        if message["type"] == "hello" or message["type"] == "hello_error":
            self._fail_and_close(ProtocolValidationError("Unexpected handshake message"))
            return
        self._options.on_message(message)

    def _handle_close(self) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected":
            return
        error: Exception = PiDisconnectedError("Byte transport closed")
        decoder = lifecycle.decoder
        assert decoder is not None
        try:
            decoder.end()
        except Exception as decoder_error:
            error = decoder_error
        self._fail(error)

    def _fail_and_close(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        transport = None if lifecycle.state == "disconnected" else lifecycle.transport
        self._fail(error)
        if transport is not None:
            transport.close()

    def _fail(self, error: Exception) -> None:
        lifecycle = self._lifecycle
        if lifecycle.state == "disconnected":
            return
        self._lifecycle = _Lifecycle(state="disconnected")
        if lifecycle.handshake is not None:
            lifecycle.handshake.reject(error)
        self._options.on_state_change(ConnectionStateChange(state="disconnected", error=error))

    def _is_current(self, connection_id: int) -> bool:
        return self._lifecycle.state != "disconnected" and self._lifecycle.id == connection_id
