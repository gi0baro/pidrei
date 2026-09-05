"""pidrei-only: `utils/websocket.connect` releases its transport on a cancel.

The handshake runs inside a scope-owned producer; a cancel arrives as a
`CancelledError` at the handshake's await, and a close awaited from that
handler would never run (tonio serves no suspension of a cancelled chain).
"""

import pytest
import tonio.colored as tonio
from tonio.exceptions import CancelledError

from pidrei_ai.utils import websocket


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
