"""A client WebSocket: httpunk's HTTP/1 upgrade plus a sans-io frame codec.

pi calls the runtime's global `WebSocket` (packages/ai/src/api/
openai-codex-responses.ts). There is no such thing here, and no tonio WebSocket
client either, so this module is it — the transport the Codex responses adapter
drives. It owns nothing protocol-shaped: httpunk performs the handshake through
`utils/http.py` (`Response.is_upgrade` -> `H1Upgraded`, which drains the bytes
already buffered past the response head), and `websockets`' sans-io `Protocol`
does the framing, masking, automatic pongs and close bookkeeping. What is left
is this file: connect, a read task, a write task. See PLAN.md §5c for why that
codec and not `wsproto`, `picows` or a monkey-patched `websockets` client.

**Divergence from pi's surface, deliberate.** pi's `WebSocketLike` is
DOM-shaped: `addEventListener("message" | "error" | "close", ...)`. JavaScript
can register those listeners *after* connecting because nothing dispatches
until the current task yields. Here the read task is genuinely concurrent, so a
frame can arrive before a listener exists — a lost message. Events are
therefore queued from the moment the socket opens and consumed with
`receive_event()`; nothing can slip in between. `open` disappears with the
listeners: `connect()` returns only once the socket *is* open.

Scope is exactly what pi's Codex transport asks for and no more: text frames
out (an incoming binary frame is still surfaced, as pi's `decodeWebSocketData`
does), no subprotocols, no extensions (`permessage-deflate` appears nowhere in
pi), no proxy — pi's own proxy support here is a bun-specific workaround for a
bun bug, and Node, its other runtime, does not proxy WebSockets either.
"""

import base64
import os
import ssl
import threading
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

import tonio.colored as tonio
from tonio.colored.net import open_tcp_stream, tls
from tonio.colored.sync import channel
from websockets.frames import OP_BINARY, OP_CLOSE, OP_TEXT
from websockets.protocol import CLIENT, OPEN, Protocol as FrameProtocol
from websockets.utils import accept_key

from pidrei_ai.utils import http
from pidrei_ai.utils.cancel import CancelToken


WEBSOCKET_GUID_ACCEPT_HEADER = "sec-websocket-accept"
# readyState values pi checks a cached socket against (DOM `WebSocket`).
READY_STATE_OPEN = 1
READY_STATE_CLOSED = 3
_DEFAULT_CLOSE_CODE = 1000


@dataclass(slots=True)
class MessageEvent:
    """A received data frame (`data` is text, or bytes for a binary frame)."""

    data: str | bytes
    type: str = "message"


@dataclass(slots=True)
class CloseEvent:
    code: int | None = None
    reason: str | None = None
    was_clean: bool | None = None
    type: str = "close"


@dataclass(slots=True)
class ErrorEvent:
    message: str | None = None
    error: BaseException | None = None
    type: str = "error"


type WebSocketEvent = MessageEvent | CloseEvent | ErrorEvent


class WebSocketLike(Protocol):
    """The surface the Codex adapter drives (pi's `WebSocketLike`, dequeued)."""

    @property
    def ready_state(self) -> int: ...

    def send(self, data: str) -> None: ...

    def close(self, code: int | None = None, reason: str | None = None) -> None: ...

    async def receive_event(self) -> WebSocketEvent: ...


class WebSocketConnection:
    """An open client WebSocket over an upgraded HTTP/1 byte stream.

    `send`/`close` are synchronous, like the DOM methods pi calls: they encode a
    frame and hand the bytes to the write task. Every write goes through that
    one task so the codec's output ordering survives concurrent senders and the
    read task's automatic pongs; the codec itself is guarded by a lock, since
    tonio may run the two tasks on different threads.
    """

    def __init__(self, upgraded: Any, protocol: FrameProtocol) -> None:
        self._upgraded = upgraded
        self._protocol = protocol
        self._lock = threading.Lock()
        self._closed = False
        self._events_sender, self._events_receiver = channel.unbounded()
        self._out_sender, self._out_receiver = channel.unbounded()
        tonio.spawn.without_tracking(self._read_loop())
        tonio.spawn.without_tracking(self._write_loop())

    @property
    def ready_state(self) -> int:
        return READY_STATE_CLOSED if self._closed else READY_STATE_OPEN

    def send(self, data: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._protocol.send_text(data.encode())
            self._flush_locked()

    def close(self, code: int | None = None, reason: str | None = None) -> None:
        """Start the closing handshake (idempotent, like `WebSocket.close`)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._protocol.send_close(code if code is not None else _DEFAULT_CLOSE_CODE, reason or "")
                self._flush_locked()
            except Exception:
                pass
            self._out_sender.send(None)

    def _flush_locked(self) -> None:
        """Queue whatever the codec wants on the wire (caller holds the lock)."""
        for data in self._protocol.data_to_send():
            self._out_sender.send(data)

    async def _write_loop(self) -> None:
        while True:
            data = await self._out_receiver.receive()
            # `None` ends the loop; the codec's `b""` end-of-stream marker means
            # our close frame is on the wire, so the transport goes with it —
            # nothing in the Codex transport waits out the server's close echo.
            if data is None or data == b"":
                break
            try:
                await self._upgraded.send_all(data)
            except Exception:
                break
        await self._teardown()

    async def _read_loop(self) -> None:
        try:
            while True:
                chunk = await self._upgraded.receive_some()
                with self._lock:
                    events = self._receive_locked(chunk)
                for event in events:
                    self._events_sender.send(event)
                if not chunk:
                    break
        except Exception as error:
            self._events_sender.send(ErrorEvent(message=str(error) or type(error).__name__, error=error))
        self._finish()

    def _receive_locked(self, chunk: bytes) -> list[WebSocketEvent]:
        """Feed the codec and translate its frames (caller holds the lock)."""
        try:
            if chunk:
                self._protocol.receive_data(chunk)
            else:
                self._protocol.receive_eof()
        except Exception as error:
            return [ErrorEvent(message=str(error) or type(error).__name__, error=error)]
        events: list[WebSocketEvent] = []
        for frame in self._protocol.events_received():
            if frame.opcode is OP_TEXT:
                events.append(MessageEvent(data=frame.data.decode("utf-8", "replace")))
            elif frame.opcode is OP_BINARY:
                events.append(MessageEvent(data=bytes(frame.data)))
            elif frame.opcode is OP_CLOSE:
                close = self._protocol.close_rcvd
                # A DOM socket is already CLOSED when its close event fires, and
                # pi reads exactly that to decide whether a cached socket can be
                # reused - so the state flips before the event is queued, not
                # when the transport later reaches EOF.
                self._closed = True
                events.append(
                    CloseEvent(
                        code=close.code if close is not None else None,
                        reason=close.reason if close is not None and close.reason else None,
                        was_clean=self._protocol.close_sent is not None,
                    )
                )
        self._flush_locked()
        return events

    def _finish(self) -> None:
        """Mark the socket closed and release a consumer waiting on an event."""
        with self._lock:
            already_closed = self._closed
            self._closed = True
        if not already_closed:
            self._events_sender.send(CloseEvent(code=self._protocol.close_code, was_clean=False))
        self._out_sender.send(None)

    async def _teardown(self) -> None:
        try:
            await self._upgraded.aclose()
        except Exception:
            pass

    async def receive_event(self) -> WebSocketEvent:
        return await self._events_receiver.receive()


def _resolve_target(url: str) -> tuple[str, int, bool, str, str]:
    """Split a ws/wss URL into (host, port, tls, request target, Host header)."""
    parts = urlsplit(url)
    secure = parts.scheme in ("wss", "https")
    host = parts.hostname
    if not host:
        raise ValueError(f"Invalid WebSocket URL: {url}")
    port = parts.port if parts.port is not None else (443 if secure else 80)
    target = parts.path or "/"
    if parts.query:
        target = f"{target}?{parts.query}"
    authority = host if parts.port is None else f"{host}:{parts.port}"
    return host, port, secure, target, authority


async def connect(url: str, headers: dict[str, str], *, cancel: CancelToken | None = None) -> WebSocketConnection:
    """Open a WebSocket to `url`, returning once the handshake succeeded.

    `headers` are sent verbatim alongside the handshake's own fields, which the
    caller must not set. Cancellation and connect deadlines belong to the
    caller (the Codex adapter races this against both).
    """
    host, port, secure, target, authority = _resolve_target(url)
    key = base64.b64encode(os.urandom(16)).decode()
    handshake_headers = {
        "host": authority,
        "upgrade": "websocket",
        "connection": "Upgrade",
        "sec-websocket-key": key,
        "sec-websocket-version": "13",
    }
    for name, value in headers.items():
        if name.lower() not in handshake_headers:
            handshake_headers[name] = value

    if secure:
        context = ssl.create_default_context()
        transport = await tls.open_tls_over_tcp_stream(host, port, ssl_context=context)
    else:
        transport = await open_tcp_stream(host, port)

    try:
        status, response_headers, upgraded = await http.h1_client_upgrade(transport, target, handshake_headers)
    except Exception:
        await _close_transport(transport)
        raise

    if status != 101 or upgraded is None:
        await _close_transport(transport)
        raise RuntimeError(f"WebSocket handshake failed with status {status}")
    if response_headers.get(WEBSOCKET_GUID_ACCEPT_HEADER) != accept_key(key):
        await upgraded.aclose()
        raise RuntimeError("WebSocket handshake failed: invalid Sec-WebSocket-Accept")

    if cancel is not None and cancel.cancelled:
        await upgraded.aclose()
        raise RuntimeError("Request was aborted")

    return WebSocketConnection(upgraded, FrameProtocol(CLIENT, state=OPEN))


async def _close_transport(transport: Any) -> None:
    try:
        result = transport.close()
        if result is not None:
            await result
    except Exception:
        pass
