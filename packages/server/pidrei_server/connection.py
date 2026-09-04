"""Byte-connection contracts (the transport half of pi server `connection.ts`).

`ByteConnection.send` follows the repo's transport contract: a plain method
that takes ownership of the chunk synchronously in invocation order and
returns an awaitable for the outcome — pi relies on `async send` queueing the
frame before its first await, which a lazy coroutine would not do. The
per-connection wire state (handshake stages, the message decoder) went with
the protocol server — UPSTREAM_EXPERIMENTAL_RULING.md.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


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
