"""Mirror of pi client src/index.ts.

The unix transport is a separate import (`pidrei_client.unix`), matching pi's
`@earendil-works/pi-client/unix` subpath export.
"""

from .client import PiClient
from .errors import (
    PiClientDisposedError,
    PiDisconnectedError,
    PiServerError,
    PiSessionDetachedError,
    PiSessionOwnershipError,
)
from .session_handle import AcquireSessionOptions, PiSessionHandle, SessionLease, SessionLeaseMode
from .transport import ByteTransport, ByteTransportFactory, ByteTransportHandlers
from .types import (
    ConnectionState,
    ConnectionStateChange,
    CreateSessionOptions,
    ListenerErrorHandler,
    PiClientOptions,
    Unsubscribe,
)


__all__ = [
    "AcquireSessionOptions",
    "ByteTransport",
    "ByteTransportFactory",
    "ByteTransportHandlers",
    "ConnectionState",
    "ConnectionStateChange",
    "CreateSessionOptions",
    "ListenerErrorHandler",
    "PiClient",
    "PiClientDisposedError",
    "PiClientOptions",
    "PiDisconnectedError",
    "PiServerError",
    "PiSessionDetachedError",
    "PiSessionHandle",
    "PiSessionOwnershipError",
    "SessionLease",
    "SessionLeaseMode",
    "Unsubscribe",
]
