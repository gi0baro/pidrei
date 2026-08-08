"""Port of pi client `test/support.ts`.

The in-memory byte server mirrors the JS transport: `send` accepts and
dispatches the chunk synchronously (run-to-first-await parity) and reports the
outcome through the returned awaitable. `flush()` stands in for JS microtask
turns: it suspends into the tonio reactor so spawned continuations (request
drivers, release chains) can run.
"""

from collections.abc import Callable

import tonio.colored as tonio

from pidrei_client import PiClient, PiClientOptions
from pidrei_client.promise import rejected, resolved
from pidrei_client.transport import ByteTransportHandlers
from pidrei_protocol import (
    PROTOCOL_VERSION,
    ClientMessage,
    ClientMessageDecoder,
    ServerMessage,
    ServerSnapshot,
    SessionSnapshot,
    encode_server_message,
)


async def flush(turns: int = 4) -> None:
    """Let spawned continuations settle (JS `await Promise.resolve()` turns)."""
    for _ in range(turns):
        await tonio.sleep(0)


class _MemoryTransport:
    def __init__(self, server: MemoryByteServer) -> None:
        self._server = server
        self._closed = False

    def send(self, chunk: bytes):
        try:
            if self._closed:
                raise Exception("Transport is closed")
            self._server.sent_by_client.append(bytes(chunk))
            for message in self._server.decoder.push(bytes(chunk)):
                for listener in list(self._server.message_listeners):
                    listener(message)
        except Exception as error:
            return rejected(error)
        return resolved(None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._server.client_close_count += 1


class MemoryByteServer:
    def __init__(self) -> None:
        self.handlers: ByteTransportHandlers | None = None
        self.decoder = ClientMessageDecoder()
        self.message_listeners: set[Callable[[ClientMessage], None]] = set()
        self.sent_by_client: list[bytes] = []
        self.client_close_count = 0

    def connect(self, handlers: ByteTransportHandlers) -> _MemoryTransport:
        self.handlers = handlers
        return _MemoryTransport(self)

    def on_message(self, listener: Callable[[ClientMessage], None]) -> Callable[[], None]:
        self.message_listeners.add(listener)
        return lambda: self.message_listeners.discard(listener)

    def send(self, message: ServerMessage, split_at: int | None = None) -> None:
        frame = encode_server_message(message)
        if split_at is None:
            self.send_raw(frame)
            return
        self.send_raw(frame[:split_at])
        self.send_raw(frame[split_at:])

    def send_together(self, messages: list[ServerMessage]) -> None:
        self.send_raw(b"".join(encode_server_message(message) for message in messages))

    def send_raw(self, chunk: bytes) -> None:
        if self.handlers is not None:
            self.handlers.on_data(chunk)

    def close(self) -> None:
        if self.handlers is not None:
            self.handlers.on_close()

    def error(self, error: Exception) -> None:
        if self.handlers is not None:
            self.handlers.on_error(error)


BASE_SERVER_SNAPSHOT: ServerSnapshot = {
    "serverId": "server-1",
    "protocolVersion": PROTOCOL_VERSION,
    "revision": 1,
    "sessions": [],
    "models": [],
}


def session_snapshot(session_id: str, **overrides: object) -> SessionSnapshot:
    return {
        "id": session_id,
        "cwd": "/workspace",
        "createdAt": 1,
        "updatedAt": 1,
        "phase": "idle",
        "model": {"provider": "faux", "id": "model"},
        "thinkingLevel": "off",
        "attached": True,
        "locked": True,
        "revision": 1,
        "transcript": [],
        "queuedSteer": [],
        "queuedSteerCount": 0,
        **overrides,
    }


def create_client(server: MemoryByteServer) -> PiClient:
    async def transport_factory(handlers: ByteTransportHandlers):
        return server.connect(handlers)

    return PiClient(PiClientOptions(transport_factory=transport_factory))


async def connect_client(server: MemoryByteServer) -> PiClient:
    client = create_client(server)

    def on_message(message: ClientMessage) -> None:
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": BASE_SERVER_SNAPSHOT,
                }
            )

    server.on_message(on_message)
    await client.connect()
    return client


def collect_requests(server: MemoryByteServer) -> list[ClientMessage]:
    requests: list[ClientMessage] = []

    def on_message(message: ClientMessage) -> None:
        if message["type"] == "request":
            requests.append(message)

    server.on_message(on_message)
    return requests


async def attach_session(client: PiClient, server: MemoryByteServer, snapshot: SessionSnapshot):
    requests = collect_requests(server)
    attaching = client.attach_session(snapshot["id"])
    request = next((candidate for candidate in requests if candidate["request"]["command"] == "attach"), None)
    if request is None:
        raise Exception("Missing attach request")
    server.send(
        {
            "type": "response",
            "id": request["id"],
            "ok": True,
            "result": {"command": "attach", "session": snapshot},
        }
    )
    return await attaching
