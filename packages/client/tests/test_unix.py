"""Port of pi client `test/unix.test.ts` (POSIX-only; the win32 guard test is dropped)."""

import pytest
import tonio.colored as tonio
from tonio.colored import net

from pidrei_client import PiClient, PiClientOptions
from pidrei_client.transport import ByteTransportHandlers
from pidrei_client.unix import UnixTransportOptions, create_unix_transport_factory
from pidrei_protocol import (
    PROTOCOL_VERSION,
    ClientMessageDecoder,
    ProtocolValidationError,
    ServerSnapshot,
    encode_server_message,
)


SERVER_SNAPSHOT: ServerSnapshot = {
    "serverId": "unix-server",
    "protocolVersion": PROTOCOL_VERSION,
    "revision": 4,
    "sessions": [],
    "models": [],
}


def test_rejects_invalid_unix_transport_options():
    with pytest.raises(TypeError, match="must not be empty"):
        create_unix_transport_factory(UnixTransportOptions(path=""))
    with pytest.raises(TypeError, match="too long"):
        create_unix_transport_factory(UnixTransportOptions(path=f"/tmp/{'x' * 512}"))
    with pytest.raises(TypeError, match="positive"):
        create_unix_transport_factory(UnixTransportOptions(path="/tmp/pi.sock", max_pending_bytes=0))


@pytest.mark.tonio
async def test_pi_client_exchanges_fragmented_framed_messages_over_a_real_unix_socket(tmp_path):
    socket_path = str(tmp_path / "pi.sock")
    listener = await net.open_unix_listener(socket_path)

    async def serve():
        stream = await listener.accept()
        decoder = ClientMessageDecoder()
        try:
            while True:
                chunk = await stream.receive_some()
                if not chunk:
                    return
                for message in decoder.push(chunk):
                    if message["type"] == "hello":
                        hello = encode_server_message(
                            {
                                "type": "hello",
                                "version": PROTOCOL_VERSION,
                                "connectionId": "unix-connection",
                                "snapshot": SERVER_SNAPSHOT,
                            }
                        )
                        for byte in hello:
                            await stream.send_all(bytes([byte]))
                    else:
                        response = encode_server_message(
                            {
                                "type": "response",
                                "id": message["id"],
                                "ok": True,
                                "result": {"command": "list", "sessions": []},
                            }
                        )
                        split = len(response) // 2
                        await stream.send_all(response[:split])
                        await stream.send_all(response[split:])
        finally:
            stream.close()

    server_task = tonio.spawn(serve())
    client = PiClient(
        PiClientOptions(transport_factory=create_unix_transport_factory(UnixTransportOptions(path=socket_path)))
    )

    try:
        assert await client.connect() == SERVER_SNAPSHOT
        first = client.list_sessions()
        second = client.list_sessions()
        assert [await first, await second] == [[], []]
    finally:
        client.disconnect()
        await server_task
        listener.close()


@pytest.mark.tonio
async def test_bounds_pending_writes_preserves_order_and_reports_remote_end_once(tmp_path):
    socket_path = str(tmp_path / "pi.sock")
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
async def test_pi_client_rejects_a_truncated_final_frame_from_a_real_unix_socket(tmp_path):
    socket_path = str(tmp_path / "pi.sock")
    listener = await net.open_unix_listener(socket_path)

    async def serve():
        stream = await listener.accept()
        decoder = ClientMessageDecoder()
        while True:
            chunk = await stream.receive_some()
            if not chunk:
                return
            for message in decoder.push(chunk):
                if message["type"] == "hello":
                    await stream.send_all(
                        encode_server_message(
                            {
                                "type": "hello",
                                "version": PROTOCOL_VERSION,
                                "connectionId": "unix-truncated",
                                "snapshot": SERVER_SNAPSHOT,
                            }
                        )
                    )
                else:
                    await stream.send_all(bytes([0, 0, 0, 2, 1]))
                    stream.send_eof()
                    return

    server_task = tonio.spawn(serve())
    client = PiClient(
        PiClientOptions(transport_factory=create_unix_transport_factory(UnixTransportOptions(path=socket_path)))
    )

    try:
        await client.connect()
        with pytest.raises(ProtocolValidationError):
            await client.list_sessions()
        assert client.connection_state == "disconnected"
    finally:
        client.disconnect()
        await server_task
        listener.close()


@pytest.mark.tonio
async def test_rejects_connection_errors(tmp_path):
    missing_path = str(tmp_path / "missing.sock")
    factory = create_unix_transport_factory(UnixTransportOptions(path=missing_path))
    with pytest.raises(OSError):
        await factory(
            ByteTransportHandlers(on_data=lambda chunk: None, on_close=lambda: None, on_error=lambda error: None)
        )
