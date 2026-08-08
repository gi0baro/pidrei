"""Mirror of pi server `testing/index.ts`.

`Deferred` is the shared promise primitive rather than a testing-local class
(await the deferred itself where upstream awaits `.promise`).
"""

from ..promise import Deferred
from .client import ProtocolTestClient, WireChannel, connect_unix_test_client
from .server import TestServer, TestServerOptions, create_test_server
from .service import TEST_MODEL, TestServerService, TestSessionRuntime


__all__ = [
    "TEST_MODEL",
    "Deferred",
    "ProtocolTestClient",
    "TestServer",
    "TestServerOptions",
    "TestServerService",
    "TestSessionRuntime",
    "WireChannel",
    "connect_unix_test_client",
    "create_test_server",
]
