"""Unix listener option types (port of pi server `transports/unix/types.ts`).

pi's `UnixServerOptions extends Omit<PiServerOptions, "listeners"> &
UnixListenerOptions` becomes one flat dataclass; Python has no structural
option merging.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class UnixListenerOptions:
    path: str
    # Socket filesystem permissions. Defaults to owner read/write only (0o600).
    mode: int | None = None
    # Maximum framed bytes queued per connection before a slow peer is disconnected.
    max_pending_bytes: int | None = None
    graceful_close_timeout_ms: int | None = None
    # Used to derive and validate max_pending_bytes. Must match the server when customized.
    max_frame_length: int | None = None
    on_error: Callable[[Exception], None] | None = None


@dataclass(slots=True, frozen=True)
class UnixServerOptions:
    path: str
    mode: int | None = None
    max_pending_bytes: int | None = None
    graceful_close_timeout_ms: int | None = None
    max_frame_length: int | None = None
    on_error: Callable[[Exception], None] | None = None
    handshake_timeout_ms: int | None = None
    server_id: str | None = None
