"""Wire-level test client (port of pi server `testing/client.ts`).

`hello()` mirrors pi's shape — register the waiter, fire the send without
awaiting it, return the response awaitable — so a caller observes the same
ordering as the JS `void this.sendMessage(...)` prologue.

pi scans the backlog and registers a waiter in one synchronous step; here the
socket reader appends messages from another thread, so `_guard` makes
scan-then-register atomic against append-then-notify. Without it a message
landing in the gap is missed by both — the waiter never resolves and the
awaiting test wedges (seen as a macOS-CI job timeout in the busy-work
session test, whose bare `next()` follows an already-spawned request).
"""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import tonio.colored as tonio
from tonio.colored import net

from pidrei_protocol import (
    PROTOCOL_VERSION,
    ClientMessage,
    Command,
    ServerMessage,
    ServerMessageDecoder,
    encode_client_message,
)

from ..promise import Deferred, rejected, resolved


@dataclass(slots=True, frozen=True)
class WireChannel:
    send: Callable[[bytes], Awaitable[None]]
    send_fragmented: Callable[[bytes, int], Awaitable[None]]
    close: Callable[[], Awaitable[None]]


@dataclass(slots=True, eq=False)
class _MessageWaiter:
    predicate: Callable[[ServerMessage], bool]
    deferred: Deferred


class ProtocolTestClient:
    def __init__(self, channel: WireChannel) -> None:
        self.messages: list[ServerMessage] = []
        self._channel = channel
        self._decoder = ServerMessageDecoder()
        self._waiters: set[_MessageWaiter] = set()
        self._closed_deferred = Deferred()
        self._request_sequence = 0
        self._closed_value = False
        self._guard = threading.Lock()

    @property
    def closed(self) -> bool:
        return self._closed_value

    def hello(self, version: int = PROTOCOL_VERSION) -> Awaitable[ServerMessage]:
        response = self.next(lambda message: message["type"] in ("hello", "hello_error"))
        self._send_silenced({"type": "hello", "version": version})
        return response

    async def request(self, command: Command, request_id: str | None = None) -> ServerMessage:
        if request_id is None:
            self._request_sequence += 1
            request_id = f"request-{self._request_sequence}"
        response = self.next(lambda message: message["type"] == "response" and message["id"] == request_id)
        await self.send_message({"type": "request", "id": request_id, "request": command})
        return await response

    def send_message(self, message: ClientMessage) -> Awaitable[None]:
        return self._channel.send(encode_client_message(message))

    def send_bytes(self, chunk: bytes) -> Awaitable[None]:
        return self._channel.send(chunk)

    def send_fragmented_message(self, message: ClientMessage, split_at: int) -> Awaitable[None]:
        return self._channel.send_fragmented(encode_client_message(message), split_at)

    def next(self, predicate: Callable[[ServerMessage], bool]) -> Awaitable[ServerMessage]:
        return self.next_from(0, predicate)

    def next_from(self, index: int, predicate: Callable[[ServerMessage], bool]) -> Awaitable[ServerMessage]:
        with self._guard:
            for message in self.messages[index:]:
                if predicate(message):
                    return resolved(message)
            if self._closed_value:
                return rejected(Exception("Wire client is closed"))
            waiter = _MessageWaiter(predicate=predicate, deferred=Deferred())
            self._waiters.add(waiter)
        return waiter.deferred

    def wait_for_close(self) -> Awaitable[None]:
        return resolved(None) if self._closed_value else self._closed_deferred

    def close(self) -> Awaitable[None]:
        return self._channel.close()

    def receive(self, chunk: bytes) -> None:
        try:
            for message in self._decoder.push(chunk):
                with self._guard:
                    self.messages.append(message)
                    matched = [waiter for waiter in self._waiters if waiter.predicate(message)]
                    self._waiters.difference_update(matched)
                for waiter in matched:
                    waiter.deferred.resolve(message)
        except Exception as error:
            self.fail(error)

    def mark_closed(self) -> None:
        with self._guard:
            if self._closed_value:
                return
            self._closed_value = True
        self._closed_deferred.resolve(None)
        self.fail(Exception("Wire connection closed"))

    def fail(self, error: Exception) -> None:
        with self._guard:
            waiters = list(self._waiters)
            self._waiters.clear()
        for waiter in waiters:
            waiter.deferred.reject(error)

    def _send_silenced(self, message: ClientMessage) -> None:
        sending = self.send_message(message)

        async def _run() -> None:
            try:
                await sending
            except Exception:
                pass

        tonio.spawn.without_tracking(_run())


async def connect_unix_test_client(path: str) -> ProtocolTestClient:
    stream = await net.open_unix_socket(path)
    locally_closed = [False]

    async def _send(chunk: bytes) -> None:
        await stream.send_all(chunk)

    async def _send_fragmented(chunk: bytes, split_at: int) -> None:
        await stream.send_all(chunk[:split_at])
        await stream.send_all(chunk[split_at:])

    async def _close() -> None:
        # Unlike Node (allowHalfOpen: false auto-destroys on remote EOF), the
        # tonio stream must always be closed locally to release the socket.
        locally_closed[0] = True
        try:
            stream.close()
        except Exception:
            pass
        await client.wait_for_close()

    client = ProtocolTestClient(WireChannel(send=_send, send_fragmented=_send_fragmented, close=_close))

    async def _run_reader() -> None:
        try:
            while True:
                chunk = await stream.receive_some()
                if not chunk:
                    break
                client.receive(chunk)
        except Exception as error:
            if not locally_closed[0]:
                client.fail(error)
        # Node sockets (allowHalfOpen: false) destroy on remote EOF; close the
        # tonio stream likewise so the server observes the disconnect.
        try:
            stream.close()
        except Exception:
            pass
        client.mark_closed()

    tonio.spawn.without_tracking(_run_reader())
    return client
