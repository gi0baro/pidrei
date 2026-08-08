"""Byte-transport contract (port of pi client `transport.ts`).

The JS contract leans on run-to-first-await semantics: `send()` is invoked
synchronously and its effects (ordering, delivery to an in-memory peer) happen
at call time, while the outcome arrives through the returned promise. Python
coroutines are lazy, so the port makes that split explicit: `send` is a plain
method that must take ownership of the chunk in invocation order before
returning an awaitable for the outcome.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


class ByteTransport(Protocol):
    def send(self, chunk: bytes) -> Awaitable[None]:
        """Sends one byte chunk. Calls must be delivered in invocation order.

        The chunk must be accepted (or refused) synchronously; the returned
        awaitable reports the outcome and never needs to be awaited for the
        send to make progress.
        """
        ...

    def close(self) -> None:
        """Closes the transport. Implementations must make repeated calls harmless."""
        ...


@dataclass(slots=True, frozen=True)
class ByteTransportHandlers:
    """Delivers inbound bytes and exactly one terminal close/error signal."""

    on_data: Callable[[bytes], None]
    on_close: Callable[[], None]
    on_error: Callable[[Exception], None]


# Creates a fresh connected, authenticated transport. Exactly one terminal handler is expected.
type ByteTransportFactory = Callable[[ByteTransportHandlers], Awaitable[ByteTransport]]
