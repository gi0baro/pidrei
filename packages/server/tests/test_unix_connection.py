"""Port of pi server `test/unix-connection.test.ts`.

Node's `ControlledSocket` (write callback held open) becomes a controlled
stream whose `send_all` parks on an event; `socket.end(finalChunk)` maps to
the recorded final `send_all` + `send_eof`.
"""

import pytest
import tonio.colored as tonio

from pidrei_protocol import FrameDecoder, decode_cbor, encode_cbor, encode_frame
from pidrei_server.transports.unix.listener import UnixByteConnection
from tests.server_support import flush


class ControlledStream:
    def __init__(self):
        self.sent: list[bytes] = []
        self.eof_sent = False
        self.closed = False
        self.release = tonio.Event()

    async def send_all(self, data):
        await self.release.wait()
        self.sent.append(bytes(data))

    def send_eof(self):
        self.eof_sent = True

    def close(self):
        self.closed = True


@pytest.mark.tonio
async def test_queues_a_final_protocol_error_behind_pending_output_before_closing():
    stream = ControlledStream()
    connection = UnixByteConnection(stream, 1_000, 64 * 1024)
    pending_write = connection.send(bytes([1, 2, 3]))
    await flush()
    final_message = {"type": "hello_error", "error": {"code": "invalid_request", "message": "Protocol violation"}}
    closing = connection.close(encode_frame(encode_cbor(final_message)))
    await flush()

    assert stream.eof_sent is False
    assert stream.closed is False
    assert stream.sent == []

    stream.release.set()
    await pending_write
    await flush()
    assert stream.eof_sent is True
    assert [decode_cbor(frame) for frame in FrameDecoder().push(stream.sent[-1])] == [final_message]

    connection.mark_closed()
    await closing
