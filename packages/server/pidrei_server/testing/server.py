"""Test server composition (port of pi server `testing/server.ts`)."""

from collections.abc import Callable
from dataclasses import dataclass

from ..listener import PiServerListener
from ..server import PiServer
from ..types import PiServerOptions, PiServerService
from .service import TestServerService


@dataclass(slots=True, frozen=True)
class TestServerOptions:
    __test__ = False  # not a pytest class, despite the upstream name

    listeners: list[PiServerListener]
    max_frame_length: int | None = None
    handshake_timeout_ms: int | None = None
    server_id: str | None = None
    on_error: Callable[[Exception], None] | None = None
    service: PiServerService | None = None


@dataclass(slots=True, frozen=True)
class TestServer:
    __test__ = False  # not a pytest class, despite the upstream name

    server: PiServer
    service: PiServerService


def create_test_server(options: TestServerOptions) -> TestServer:
    """Create an unstarted PiServer with deterministic defaults for transport conformance tests."""
    service = options.service if options.service is not None else TestServerService()
    return TestServer(
        server=PiServer(
            service,
            PiServerOptions(
                listeners=options.listeners,
                max_frame_length=options.max_frame_length,
                handshake_timeout_ms=options.handshake_timeout_ms,
                server_id=options.server_id,
                on_error=options.on_error,
            ),
        ),
        service=service,
    )
