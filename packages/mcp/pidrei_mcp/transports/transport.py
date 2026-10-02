"""Mirror of pi mcp src/transports/transport.ts.

A transport owns framing and I/O and hands the client individual JSON-RPC
messages. What a server sends is delivered one event at a time, in arrival
order: every emitter (a stdout reader, an HTTP stream reader, the in-memory
peer) puts its event on one channel, and one consumer awaits the listeners
for each event in registration order before taking the next. The close is
closing the channel: the consumer drains what was queued before it, then
delivers the close; an event emitted after it is dropped.

Outgoing, `send()` takes the message's place in the transport's order when
it is called and returns a `SendResult` to await for the outcome.
"""

import threading
from collections.abc import Awaitable, Callable, Generator
from typing import Any, Protocol

import tonio.colored as tonio
from tonio.colored.sync import channel

from ..protocol.jsonrpc import JsonRpcMessage


DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024

type McpTransportMessageListener = Callable[[JsonRpcMessage], Awaitable[None]]
type McpTransportErrorListener = Callable[[Exception], Awaitable[None]]
type McpTransportCloseListener = Callable[[], Awaitable[None]]


class SendResult:
    """The outcome of one `send()`: settled once, awaited for success or the
    send's error. `done`/`error` let a caller that does not wait for it
    handle an outcome that is already known without a coroutine."""

    __slots__ = ("_error", "_event", "_lock")

    def __init__(self) -> None:
        self._event = tonio.Event()
        self._lock = threading.Lock()
        self._error: Exception | None = None

    @classmethod
    def succeeded(cls) -> SendResult:
        result = cls()
        result.settle()
        return result

    @classmethod
    def failed(cls, error: Exception) -> SendResult:
        result = cls()
        result.settle(error)
        return result

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def error(self) -> Exception | None:
        return self._error

    def settle(self, error: Exception | None = None) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._error = error
            self._event.set()

    async def _wait(self) -> None:
        await self._event.wait()
        if self._error is not None:
            raise self._error

    def __await__(self) -> Generator[Any]:
        return self._wait().__await__()


class McpTransport(Protocol):
    async def start(self) -> None: ...
    def send(self, message: JsonRpcMessage) -> SendResult: ...
    async def close(self) -> None: ...
    def on_message(self, listener: McpTransportMessageListener) -> Callable[[], None]: ...
    def on_error(self, listener: McpTransportErrorListener) -> Callable[[], None]: ...
    def on_close(self, listener: McpTransportCloseListener) -> Callable[[], None]: ...
    def set_protocol_version(self, version: str) -> None: ...


_MESSAGE = "message"
_ERROR = "error"


class TransportEvents:
    """Listener bookkeeping and in-order delivery shared by transports."""

    def __init__(self) -> None:
        self._events_lock = threading.Lock()
        # Insertion-ordered sets: a listener registered twice is called once.
        self._message_listeners: dict[McpTransportMessageListener, None] = {}
        self._error_listeners: dict[McpTransportErrorListener, None] = {}
        self._close_listeners: dict[McpTransportCloseListener, None] = {}
        self._delivering = False
        self._events_sender, self._events_receiver = channel.unbounded()

    def on_message(self, listener: McpTransportMessageListener) -> Callable[[], None]:
        return self._subscribe(self._message_listeners, listener)

    def on_error(self, listener: McpTransportErrorListener) -> Callable[[], None]:
        return self._subscribe(self._error_listeners, listener)

    def on_close(self, listener: McpTransportCloseListener) -> Callable[[], None]:
        return self._subscribe(self._close_listeners, listener)

    def set_protocol_version(self, version: str) -> None:
        """Only the HTTP transport sends the negotiated version."""

    def _subscribe(self, listeners: dict, listener: Any) -> Callable[[], None]:
        with self._events_lock:
            listeners[listener] = None

        def unsubscribe() -> None:
            with self._events_lock:
                listeners.pop(listener, None)

        return unsubscribe

    def _emit_message(self, message: JsonRpcMessage) -> None:
        self._enqueue((_MESSAGE, message))

    def _emit_error(self, error: Exception) -> None:
        self._enqueue((_ERROR, error))

    def _emit_close(self) -> None:
        """Close the channel (idempotent): the consumer delivers what is
        queued, then the close."""
        self._events_sender.close()
        self._start_delivery()

    def _enqueue(self, event: tuple[str, Any]) -> None:
        try:
            self._events_sender.send(event)
        except BrokenPipeError:
            # Emitted after the close: dropped.
            return
        self._start_delivery()

    def _start_delivery(self) -> None:
        with self._events_lock:
            if self._delivering:
                return
            self._delivering = True
        tonio.spawn.without_tracking(self._delivery_loop())

    async def _delivery_loop(self) -> None:
        while True:
            try:
                kind, payload = await self._events_receiver.receive()
            except BrokenPipeError:
                # The channel is closed and drained: the close, delivered once.
                with self._events_lock:
                    listeners = list(self._close_listeners)
                for listener in listeners:
                    try:
                        await listener()
                    except Exception as error:
                        await self._report_listener_error(error)
                return
            with self._events_lock:
                listeners = list(self._message_listeners if kind == _MESSAGE else self._error_listeners)
            for listener in listeners:
                try:
                    await listener(payload)
                except Exception as error:
                    # The delivery never stops for a failing listener. A
                    # message listener's failure is reported to the error
                    # listeners; an error listener's own is dropped.
                    if kind == _MESSAGE:
                        await self._report_listener_error(error)

    async def _report_listener_error(self, error: Exception) -> None:
        with self._events_lock:
            listeners = list(self._error_listeners)
        for listener in listeners:
            try:
                await listener(error)
            except Exception:
                pass
