"""Mirror of pi mcp src/transports/in-memory.ts.

A message sent goes onto the peer's delivery channel as a deep copy (pi's
`structuredClone`), so it is delivered later and in send order, as pi's
`queueMicrotask` delivers it.
"""

import copy
import threading

from ..protocol.jsonrpc import JsonRpcMessage, McpConnectionClosedError
from .transport import SendResult, TransportEvents


class InMemoryTransport(TransportEvents):
    def __init__(self) -> None:
        super().__init__()
        self._state_lock = threading.Lock()
        self._peer: InMemoryTransport | None = None
        self._started = False
        self._closed = False

    def connect_peer(self, peer: InMemoryTransport) -> None:
        with self._state_lock:
            if self._peer is not None:
                raise RuntimeError("In-memory MCP transport already has a peer")
            self._peer = peer

    async def start(self) -> None:
        with self._state_lock:
            if self._closed:
                raise McpConnectionClosedError()
            self._started = True

    def send(self, message: JsonRpcMessage) -> SendResult:
        with self._state_lock:
            if not self._started or self._closed:
                return SendResult.failed(McpConnectionClosedError())
            peer = self._peer
        if peer is None or not peer._deliver(copy.deepcopy(message)):
            return SendResult.failed(McpConnectionClosedError("In-memory MCP peer is not connected"))
        return SendResult.succeeded()

    async def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            peer = self._peer
        self._emit_close()
        if peer is not None:
            await peer.close()

    def emit_error(self, error: Exception) -> None:
        """Exposed so tests can simulate transport-level failures."""
        self._emit_error(error)

    def _deliver(self, message: JsonRpcMessage) -> bool:
        """Queue `message` for this side's listeners; False when this side is
        not started or already closed (pi's "peer is not connected")."""
        with self._state_lock:
            if not self._started or self._closed:
                return False
            self._emit_message(message)
            return True


def create_in_memory_transport_pair() -> tuple[InMemoryTransport, InMemoryTransport]:
    """`(client, server)`."""
    client = InMemoryTransport()
    server = InMemoryTransport()
    client.connect_peer(server)
    server.connect_peer(client)
    return client, server
