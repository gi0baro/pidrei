"""Port of pi coding-agent `test/client/support.ts`.

The in-memory byte server mirrors the JS transport: `send` dispatches the
chunk synchronously (run-to-first-await parity) and server frames are pushed
straight into the client's `on_data` handler. `flush()` stands in for JS
microtask turns. Named `remote_session_support` because the flat shared
`tests` namespace already holds the client package's `support` module.
"""

from collections.abc import Callable

import tonio.colored as tonio

from pidrei.client.remote_session import RemoteSession, RemoteSessionOptions, open_remote_session
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
    """Let spawned continuations settle (JS `await Promise.resolve()` turns).

    A small positive sleep, not `sleep(0)` — see the note on the client
    package's copy of this helper: a zero sleep is not a guaranteed
    reschedule in tonio.
    """
    for _ in range(turns):
        await tonio.sleep(0.005)


class _MemoryTransport:
    def __init__(self, server: MemoryServer) -> None:
        self._server = server
        self._closed = False

    def send(self, chunk: bytes):
        try:
            if self._closed:
                raise Exception("Transport is closed")
            for message in self._server.decoder.push(bytes(chunk)):
                for listener in list(self._server.message_listeners):
                    listener(message)
        except Exception as error:
            return rejected(error)
        return resolved(None)

    def close(self) -> None:
        self._closed = True


class MemoryServer:
    def __init__(self) -> None:
        self.handlers: ByteTransportHandlers | None = None
        self.decoder = ClientMessageDecoder()
        self.message_listeners: set[Callable[[ClientMessage], None]] = set()

    def connect(self, handlers: ByteTransportHandlers) -> _MemoryTransport:
        self.handlers = handlers
        return _MemoryTransport(self)

    def on_message(self, listener: Callable[[ClientMessage], None]) -> Callable[[], None]:
        self.message_listeners.add(listener)
        return lambda: self.message_listeners.discard(listener)

    def send(self, message: ServerMessage) -> None:
        if self.handlers is not None:
            self.handlers.on_data(encode_server_message(message))

    def close(self) -> None:
        if self.handlers is not None:
            self.handlers.on_close()


SERVER_SNAPSHOT: ServerSnapshot = {
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


async def connect_client(server: MemoryServer) -> PiClient:
    async def transport_factory(handlers: ByteTransportHandlers):
        return server.connect(handlers)

    client = PiClient(PiClientOptions(transport_factory=transport_factory))

    def on_message(message: ClientMessage) -> None:
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": SERVER_SNAPSHOT,
                }
            )

    server.on_message(on_message)
    await client.connect()
    return client


def collect_requests(server: MemoryServer) -> list[ClientMessage]:
    requests: list[ClientMessage] = []

    def on_message(message: ClientMessage) -> None:
        if message["type"] == "request":
            requests.append(message)

    server.on_message(on_message)
    return requests


async def open_test_remote_session(
    client: PiClient,
    server: MemoryServer,
    snapshot: SessionSnapshot,
    options: RemoteSessionOptions | None = None,
) -> RemoteSession:
    requests = collect_requests(server)
    opening = open_remote_session(client, snapshot["id"], options)
    request = next((candidate for candidate in requests if candidate["request"]["command"] == "attach"), None)
    if request is None:
        raise Exception("Missing attach request")
    server.send(
        {"type": "response", "id": request["id"], "ok": True, "result": {"command": "attach", "session": snapshot}}
    )
    return await opening
