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
async def test_upgrades_over_a_real_socket_and_drains_bytes_sent_behind_the_head():
    closed: list = []
    close_seen = tonio.Event()
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
        settled, completed = await tonio.time.timeout(connection.receive_event(), 0.2)
        assert not completed, f"unexpected event after a local close: {settled!r}"
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
