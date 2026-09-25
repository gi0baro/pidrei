"""pidrei-only: `utils/websocket.connect` over the real HTTP stack, plus its
cancel-release contract.

The Codex transport tests stub `websocket.connect`, so the loopback cases here
are the only exercise of the seam's `h1_client_upgrade` (httpunk `H1Connection`
→ `Response.upgraded` → `H1Upgraded`) end to end: the 101 handshake, draining
bytes the server sent right behind the response head, a non-101 rejection
(which closes the response), and a clean close. Added with the httpunk 0.3.0
bump (0.85.1.1), whose exception refactor made the gap visible.

The cancel case: the handshake runs inside a scope-owned producer; a cancel
arrives as a `CancelledError` at the handshake's await, and a close awaited
from that handler would never run (tonio serves no suspension of a cancelled
chain).
"""

import re

import pytest
import tonio.colored as tonio
from tonio.colored import net
from tonio.exceptions import CancelledError
from websockets.frames import OP_CLOSE, OP_TEXT
from websockets.protocol import OPEN, SERVER, Protocol as FrameProtocol
from websockets.utils import accept_key

from pidrei_ai.utils import websocket
from pidrei_ai.utils.websocket import MessageEvent


async def _read_head(stream) -> bytes:
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = await stream.receive_some()
        if not chunk:
            break
        head += chunk
    return head


async def _serve_echo(stream, *, greet: bytes, closed: list, close_seen: tonio.Event) -> None:
    """A WebSocket peer: 101 + a greeting frame in the same write (so the
    client must drain the bytes past the head), then echo text frames and
    record the client's close frame."""
    head = await _read_head(stream)
    key = re.search(rb"sec-websocket-key:\s*(\S+)", head, re.IGNORECASE).group(1).decode()
    protocol = FrameProtocol(SERVER, state=OPEN)
    protocol.send_text(greet)
    greeting = b"".join(protocol.data_to_send())
    await stream.send_all(
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
        b"Sec-WebSocket-Accept: " + accept_key(key).encode() + b"\r\n\r\n" + greeting
    )
    while True:
        chunk = await stream.receive_some()
        if not chunk:
            break
        protocol.receive_data(chunk)
        for frame in protocol.events_received():
            if frame.opcode is OP_TEXT:
                protocol.send_text(frame.data)
            elif frame.opcode is OP_CLOSE:
                closed.append(protocol.close_rcvd)  # the protocol queued the echo itself
                close_seen.set()
        try:
            await stream.send_all(b"".join(protocol.data_to_send()))
        except Exception:
            break  # the client drops its transport right behind its close frame
        if protocol.close_sent is not None:
            break
    stream.close()


async def _serve_reject(stream) -> None:
    await _read_head(stream)
    await stream.send_all(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    stream.close()


async def _listen(serve):
    listener = (await net.open_tcp_listeners(0, host="127.0.0.1"))[0]

    async def accept_loop() -> None:
        while True:
            try:
                stream = await listener.accept()
            except Exception:
                return
            tonio.spawn.without_tracking(serve(stream))

    tonio.spawn.without_tracking(accept_loop())
    return listener, listener.socket.getsockname()[1]


@pytest.mark.tonio
async def test_upgrades_over_a_real_socket_and_drains_bytes_sent_behind_the_head(monkeypatch):
    closed: list = []
    close_seen = tonio.Event()
    # `_finish` is the read loop's last step: once it ran, every event the
    # connection will ever emit is already queued.
    read_loop_done = tonio.Event()
    finish = websocket.WebSocketConnection._finish

    def finish_and_signal(self) -> None:
        finish(self)
        read_loop_done.set()

    monkeypatch.setattr(websocket.WebSocketConnection, "_finish", finish_and_signal)
    listener, port = await _listen(
        lambda stream: _serve_echo(stream, greet=b"hello", closed=closed, close_seen=close_seen)
    )
    try:
        connection = await websocket.connect(f"ws://127.0.0.1:{port}/v1/responses", {"x-probe": "1"})
        assert connection.ready_state == websocket.READY_STATE_OPEN
        assert await connection.receive_event() == MessageEvent(data="hello")
        connection.send("ping")
        assert await connection.receive_event() == MessageEvent(data="ping")

        # A local close puts the close frame on the wire and drops the
        # transport without waiting for the peer's echo (the Codex transport
        # never does); the consumer sees no error for its own teardown.
        connection.close()
        await close_seen.wait(5.0)
        assert close_seen.is_set(), "peer never received the close frame"
        assert [c.code for c in closed] == [1000]
        assert connection.ready_state == websocket.READY_STATE_CLOSED
        # Instead of a 0.2 s quiet window: wait for the read loop to end, then
        # queue a marker behind whatever it emitted; the marker must come first.
        await read_loop_done.wait(5)
        assert read_loop_done.is_set(), "read loop never ended after a local close"
        marker = MessageEvent(data="end-of-events marker")
        connection._events_sender.send(marker)
        settled = await connection.receive_event()
        assert settled is marker, f"unexpected event after a local close: {settled!r}"
    finally:
        listener.close()


@pytest.mark.tonio
async def test_drops_the_peers_close_echo_when_it_lands_before_the_local_teardown(monkeypatch):
    # pidrei-only: the write loop tears the transport down right behind the local
    # close frame, so the peer's echo normally never arrives. Holding the teardown
    # until the read loop has received the echo forces the other order, which a
    # descheduled write task produces in production.
    closed: list = []
    close_seen = tonio.Event()
    echo_received = tonio.Event()
    read_loop_done = tonio.Event()
    receive_locked = websocket.WebSocketConnection._receive_locked
    teardown = websocket.WebSocketConnection._teardown
    finish = websocket.WebSocketConnection._finish

    def receive_and_signal(self, chunk):
        events = receive_locked(self, chunk)
        if self._protocol.close_rcvd is not None:
            echo_received.set()
        return events

    async def teardown_after_echo(self) -> None:
        await echo_received.wait(5)
        await teardown(self)

    def finish_and_signal(self) -> None:
        finish(self)
        read_loop_done.set()

    monkeypatch.setattr(websocket.WebSocketConnection, "_receive_locked", receive_and_signal)
    monkeypatch.setattr(websocket.WebSocketConnection, "_teardown", teardown_after_echo)
    monkeypatch.setattr(websocket.WebSocketConnection, "_finish", finish_and_signal)
    listener, port = await _listen(
        lambda stream: _serve_echo(stream, greet=b"hello", closed=closed, close_seen=close_seen)
    )
    try:
        connection = await websocket.connect(f"ws://127.0.0.1:{port}/v1/responses", {})
        assert await connection.receive_event() == MessageEvent(data="hello")

        connection.close()
        await echo_received.wait(5)
        assert echo_received.is_set(), "the peer's close echo never reached the read loop"
        await read_loop_done.wait(5)
        assert read_loop_done.is_set()
        marker = MessageEvent(data="end-of-events marker")
        connection._events_sender.send(marker)
        settled = await connection.receive_event()
        assert settled is marker, f"unexpected event after a local close: {settled!r}"
    finally:
        listener.close()


@pytest.mark.tonio
async def test_rejects_a_non_101_handshake_response():
    listener, port = await _listen(_serve_reject)
    try:
        with pytest.raises(RuntimeError, match="WebSocket handshake failed with status 403"):
            await websocket.connect(f"ws://127.0.0.1:{port}/v1/responses", {})
    finally:
        listener.close()


@pytest.mark.tonio
async def test_connect_closes_the_transport_when_the_handshake_is_cancelled():
    closed = tonio.Event()

    class Transport:
        def close(self) -> None:
            closed.set()

    async def open_stream(_host, _port):
        return Transport()

    async def upgrade(_transport, _target, _headers):
        raise CancelledError()

    saved = (websocket.open_tcp_stream, websocket.http.h1_client_upgrade)
    websocket.open_tcp_stream = open_stream
    websocket.http.h1_client_upgrade = upgrade
    try:
        with pytest.raises(CancelledError):
            await websocket.connect("ws://example.test/v1/responses", {})
        await closed.wait(1)
    finally:
        websocket.open_tcp_stream, websocket.http.h1_client_upgrade = saved

    assert closed.is_set()
