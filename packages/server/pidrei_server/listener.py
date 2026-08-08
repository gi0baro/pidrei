"""Transport listener contract (port of pi server `listener.ts`)."""

from collections.abc import Awaitable
from typing import Protocol

from .connection import ByteConnectionAcceptor


class PiServerListener(Protocol):
    """Supplies established byte connections after any required transport authentication."""

    @property
    def address(self) -> str | None:
        """Human-readable bound address after startup, when the transport has one."""
        ...

    def start(self, accept: ByteConnectionAcceptor) -> Awaitable[None]:
        """Starts listening and passes authorized connections to accept."""
        ...

    def close(self) -> Awaitable[None]: ...
