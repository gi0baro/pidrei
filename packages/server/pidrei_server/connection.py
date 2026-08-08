"""Per-connection wire state (port of pi server `connection.ts`).

`ByteConnection.send` follows the repo's transport contract: a plain method
that takes ownership of the chunk synchronously in invocation order and
returns an awaitable for the outcome — pi relies on `async send` queueing the
frame before its first await, which a lazy coroutine would not do. Node's
timeout handle becomes a `Timer`.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol

from pidrei_protocol import ClientMessageDecoder

from .promise import Deferred
from .timers import Timer


class ByteConnection(Protocol):
    """An established, authorized ordered byte connection."""

    @property
    def closed(self) -> bool: ...
    def send(self, chunk: bytes) -> Awaitable[None]: ...
    def close(self, final_chunk: bytes | None = None) -> Awaitable[None]: ...


@dataclass(slots=True, frozen=True)
class ByteConnectionHandler:
    on_data: Callable[[bytes], None]
    on_close: Callable[[], None]
    on_error: Callable[[Exception], None]


type ByteConnectionAcceptor = Callable[[ByteConnection], ByteConnectionHandler]

type ConnectionStage = Literal["awaitingHello", "handshaking", "ready", "closing", "closed"]


@dataclass(slots=True, eq=False)
class ConnectionState:
    id: str
    connection: ByteConnection
    decoder: ClientMessageDecoder
    handshake_timeout: Timer
    session_ids: set[str] = field(default_factory=set)
    stage: ConnectionStage = "awaitingHello"
    disconnected: bool = False
    handshake_complete: bool = False
    handshake: Deferred | None = None


def is_terminal_connection(state: ConnectionState) -> bool:
    return state.disconnected or state.stage == "closing" or state.stage == "closed"
