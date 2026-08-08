"""Shared client option and listener types (port of pi client `types.ts`)."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pidrei_protocol import ModelRef, ThinkingLevel

from .transport import ByteTransportFactory


type ConnectionState = Literal["disconnected", "connecting", "connected"]


@dataclass(slots=True, frozen=True)
class ConnectionStateChange:
    state: ConnectionState
    error: Exception | None = None


type Unsubscribe = Callable[[], None]
type ListenerErrorHandler = Callable[[Exception], None]


@dataclass(slots=True, frozen=True)
class PiClientOptions:
    transport_factory: ByteTransportFactory
    max_frame_length: int | None = None
    # Reports subscriber failures without allowing them to corrupt client state.
    on_listener_error: ListenerErrorHandler | None = None


@dataclass(slots=True, frozen=True)
class CreateSessionOptions:
    cwd: str | None = None
    name: str | None = None
    model: ModelRef | None = None
    thinking_level: ThinkingLevel | None = None
