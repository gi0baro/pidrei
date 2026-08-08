"""Unix server preset (port of pi server `transports/unix/preset.ts`)."""

from ...server import PiServer
from ...types import PiServerOptions, PiServerService
from .listener import create_unix_listener
from .types import UnixListenerOptions, UnixServerOptions


def create_unix_server(service: PiServerService, options: UnixServerOptions) -> PiServer:
    """Compose PiServer with one Unix-domain socket listener."""
    listener = create_unix_listener(
        UnixListenerOptions(
            path=options.path,
            mode=options.mode,
            max_frame_length=options.max_frame_length,
            max_pending_bytes=options.max_pending_bytes,
            graceful_close_timeout_ms=options.graceful_close_timeout_ms,
            on_error=options.on_error,
        )
    )
    return PiServer(
        service,
        PiServerOptions(
            listeners=[listener],
            max_frame_length=options.max_frame_length,
            handshake_timeout_ms=options.handshake_timeout_ms,
            server_id=options.server_id,
            on_error=options.on_error,
        ),
    )
