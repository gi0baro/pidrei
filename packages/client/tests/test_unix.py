"""Port of pi client `test/unix.test.ts` (POSIX-only; the win32 guard test is dropped).

Driven against the transport directly (the protocol client is not ported —
UPSTREAM_EXPERIMENTAL_RULING.md): the framed exchange feeds a `FrameDecoder`
from the transport handlers instead of going through the client's handshake,
and the truncated-final-frame case is gone with the client that rejected it.
"""

import pytest
import tonio.colored as tonio
from tonio.colored import net

from pidrei_client.transport import ByteTransportHandlers
from pidrei_client.unix import UnixTransportOptions, create_unix_transport_factory
from pidrei_protocol import FrameDecoder, decode_cbor, encode_cbor, encode_frame


def test_rejects_invalid_unix_transport_options():
    with pytest.raises(TypeError, match="must not be empty"):
        create_unix_transport_factory(UnixTransportOptions(path=""))
    with pytest.raises(TypeError, match="positive"):
        create_unix_transport_factory(UnixTransportOptions(path="/tmp/pi.sock", max_pending_bytes=0))


@pytest.mark.tonio
async def test_exchanges_fragmented_framed_messages_over_a_real_unix_socket(sock_dir):
    socket_path = str(sock_dir / "pi.sock")
    listener = await net.open_unix_listener(socket_path)
    request = {"type": "request", "id": "1", "call": {"serviceId": "sessions", "member": "list", "args": []}}
    replies = [{"type": "hello", "version": 1}, {"type": "response", "id": "1", "ok": True, "result": []}]

    async def serve():
        stream = await listener.accept()
        decoder = FrameDecoder()
        try:
            while True:
                chunk = await stream.receive_some()
                if not chunk:
                    return
                for frame in decoder.push(chunk):
                    assert decode_cbor(frame) == request
                    # First reply one byte at a time, second reply split in two.
                    for byte in encode_frame(encode_cbor(replies[0])):
                        await stream.send_all(bytes([byte]))
                    response = encode_frame(encode_cbor(replies[1]))
                    split = len(response) // 2
                    await stream.send_all(response[:split])
                    await stream.send_all(response[split:])
                    return
        finally:
            stream.close()

    server_task = tonio.spawn(serve())
    decoder = FrameDecoder()
    received: list[object] = []
    errors: list[Exception] = []
    done = tonio.Event()

    def on_data(chunk: bytes) -> None:
        for frame in decoder.push(chunk):
            received.append(decode_cbor(frame))
            if len(received) == len(replies):
                done.set()

    factory = create_unix_transport_factory(UnixTransportOptions(path=socket_path))
    transport = await factory(ByteTransportHandlers(on_data=on_data, on_close=done.set, on_error=errors.append))
    try:
        await transport.send(encode_frame(encode_cbor(request)))
        await done.wait(5)
        assert done.is_set(), "timed out waiting for the framed replies"
        assert received == replies
        assert errors == []
    finally:
        transport.close()
        await server_task
        listener.close()


@pytest.mark.tonio
async def test_bounds_pending_writes_preserves_order_and_reports_remote_end_once(sock_dir):
    socket_path = str(sock_dir / "pi.sock")
    first = b"\x01" * (2 * 1024 * 1024)
    second = b"\x02" * (2 * 1024 * 1024)
    expected_length = len(first) + len(second)
    received_length = [0]
    invalid_order = [False]
    resume_server = tonio.Event()
    listener = await net.open_unix_listener(socket_path)

    async def serve():
        stream = await listener.accept()
        # Mirror of `socket.pause()`: hold off reading until resumed.
        await resume_server.wait()
        while received_length[0] < expected_length:
            chunk = await stream.receive_some()
            if not chunk:
                return
            for index, byte in enumerate(chunk):
                expected = 1 if received_length[0] + index < len(first) else 2
                if byte != expected:
                    invalid_order[0] = True
            received_length[0] += len(chunk)
        await stream.send_all(b"\x09")
        stream.send_eof()

    server_task = tonio.spawn(serve())
    inbound: list[int] = []
    errors: list[Exception] = []
    close_count = [0]
    closed = tonio.Event()

    def on_close():
        close_count[0] += 1
        closed.set()

    factory = create_unix_transport_factory(UnixTransportOptions(path=socket_path, max_pending_bytes=expected_length))
    transport = await factory(ByteTransportHandlers(on_data=inbound.extend, on_close=on_close, on_error=errors.append))

    try:
        first_write = transport.send(first)
        second_write = transport.send(second)
        with pytest.raises(Exception, match="pending byte limit"):
            await transport.send(b"\x03")
        resume_server.set()
        await first_write
        await second_write
        await closed.wait()
        assert received_length[0] == expected_length
        assert invalid_order[0] is False
        assert inbound == [9]
        assert errors == []
        await tonio.sleep(0)
        assert close_count[0] == 1
    finally:
        transport.close()
        await server_task
        listener.close()


@pytest.mark.tonio
async def test_rejects_connection_errors(sock_dir):
    missing_path = str(sock_dir / "missing.sock")
    factory = create_unix_transport_factory(UnixTransportOptions(path=missing_path))
    with pytest.raises(OSError):
        await factory(
            ByteTransportHandlers(on_data=lambda chunk: None, on_close=lambda: None, on_error=lambda error: None)
        )
