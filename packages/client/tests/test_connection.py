"""Port of pi client `test/connection.test.ts`."""

import pytest

from pidrei_client import (
    PiClient,
    PiClientOptions,
    PiDisconnectedError,
    PiServerError,
)
from pidrei_client.promise import resolved
from pidrei_protocol import (
    PROTOCOL_VERSION,
    ProtocolValidationError,
    encode_cbor,
    encode_frame,
    encode_server_message,
)
from tests.support import (
    BASE_SERVER_SNAPSHOT,
    MemoryByteServer,
    attach_session,
    connect_client,
    create_client,
    session_snapshot,
)


@pytest.mark.tonio
async def test_sends_a_framed_version_before_accepting_a_fragmented_server_hello():
    server = MemoryByteServer()
    received = []

    def on_message(message):
        received.append(message)
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": BASE_SERVER_SNAPSHOT,
                },
                3,
            )

    server.on_message(on_message)
    client = create_client(server)

    assert await client.connect() == BASE_SERVER_SNAPSHOT
    assert received[0] == {"type": "hello", "version": PROTOCOL_VERSION}
    assert client.connection_state == "connected"


@pytest.mark.tonio
async def test_rejects_server_data_delivered_before_sending_the_client_hello():
    close_count = 0
    send_count = 0

    class _Transport:
        def send(self, chunk):
            nonlocal send_count
            send_count += 1
            return resolved(None)

        def close(self):
            nonlocal close_count
            close_count += 1

    async def transport_factory(handlers):
        handlers.on_data(
            encode_server_message(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": BASE_SERVER_SNAPSHOT,
                }
            )
        )
        return _Transport()

    client = PiClient(PiClientOptions(transport_factory=transport_factory))

    with pytest.raises(ProtocolValidationError, match="Received server data before the client hello was sent"):
        await client.connect()
    assert client.connection_state == "disconnected"
    assert send_count == 0
    assert close_count == 1


@pytest.mark.tonio
async def test_isolates_subscriber_failures_from_handshake_and_transport_state():
    server = MemoryByteServer()

    def on_message(message):
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": BASE_SERVER_SNAPSHOT,
                }
            )

    server.on_message(on_message)
    client = create_client(server)

    def failing_listener(snapshot):
        raise Exception("consumer failure")

    client.subscribe(failing_listener)

    assert await client.connect() == BASE_SERVER_SNAPSHOT
    assert client.connection_state == "connected"


@pytest.mark.tonio
async def test_reports_subscriber_failures_without_changing_connection_state():
    server = MemoryByteServer()
    listener_errors = []

    def on_message(message):
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": BASE_SERVER_SNAPSHOT,
                }
            )

    server.on_message(on_message)

    async def transport_factory(handlers):
        return server.connect(handlers)

    client = PiClient(PiClientOptions(transport_factory=transport_factory, on_listener_error=listener_errors.append))

    def failing_listener(snapshot):
        raise Exception("consumer failure")

    client.subscribe(failing_listener)

    assert await client.connect() == BASE_SERVER_SNAPSHOT
    assert [str(error) for error in listener_errors] == ["consumer failure"]
    assert client.connection_state == "connected"


@pytest.mark.tonio
async def test_does_not_restore_a_connection_after_a_snapshot_listener_disconnects_during_handshake():
    server = MemoryByteServer()

    def on_message(message):
        if message["type"] != "hello":
            return
        server.send(
            {
                "type": "hello",
                "version": PROTOCOL_VERSION,
                "connectionId": "connection-1",
                "snapshot": BASE_SERVER_SNAPSHOT,
            }
        )

    server.on_message(on_message)
    client = create_client(server)
    client.subscribe(lambda snapshot: client.disconnect())

    with pytest.raises(PiDisconnectedError):
        await client.connect()
    assert client.connection_state == "disconnected"
    assert server.client_close_count == 1


@pytest.mark.tonio
async def test_does_not_restore_a_stale_connection_when_a_snapshot_listener_reconnects_during_handshake():
    first = MemoryByteServer()
    second = MemoryByteServer()
    connection = [0]
    for server in (first, second):

        def on_message(message, server=server):
            if message["type"] != "hello":
                return
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": f"connection-{connection[0]}",
                    "snapshot": {**BASE_SERVER_SNAPSHOT, "revision": connection[0]},
                }
            )

        server.on_message(on_message)

    async def transport_factory(handlers):
        index = connection[0]
        connection[0] += 1
        return (first if index == 0 else second).connect(handlers)

    client = PiClient(PiClientOptions(transport_factory=transport_factory))
    reconnect = [None]
    reconnect_requested = [False]

    def on_snapshot(snapshot):
        if reconnect_requested[0]:
            return
        reconnect_requested[0] = True
        client.disconnect()
        reconnect[0] = client.reconnect()

    client.subscribe(on_snapshot)

    with pytest.raises(PiDisconnectedError):
        await client.connect()
    assert reconnect[0] is not None
    snapshot = await reconnect[0]
    assert snapshot["revision"] == 2
    assert client.connection_state == "connected"
    assert first.client_close_count == 1


@pytest.mark.tonio
async def test_rejects_a_typed_handshake_version_error():
    server = MemoryByteServer()

    def on_message(message):
        server.send(
            {
                "type": "hello_error",
                "error": {"code": "version", "message": "Unsupported protocol version"},
            }
        )

    server.on_message(on_message)
    client = create_client(server)

    with pytest.raises(PiServerError, match="Unsupported protocol version") as excinfo:
        await client.connect()
    assert excinfo.value.code == "version"
    assert client.connection_state == "disconnected"
    assert server.client_close_count == 1


@pytest.mark.tonio
async def test_rejects_pending_requests_on_close_and_reconnects_through_a_fresh_factory_result():
    first = MemoryByteServer()
    second = MemoryByteServer()
    connection = [0]
    for server in (first, second):

        def on_message(message, server=server):
            if message["type"] == "hello":
                server.send(
                    {
                        "type": "hello",
                        "version": PROTOCOL_VERSION,
                        "connectionId": f"connection-{connection[0]}",
                        "snapshot": {**BASE_SERVER_SNAPSHOT, "revision": connection[0]},
                    }
                )

        server.on_message(on_message)

    async def transport_factory(handlers):
        index = connection[0]
        connection[0] += 1
        return (first if index == 0 else second).connect(handlers)

    client = PiClient(PiClientOptions(transport_factory=transport_factory))
    states = []
    client.on_connection_state_change(lambda change: states.append(change.state))
    await client.connect()
    pending = client.list_sessions()
    first.close()
    with pytest.raises(PiDisconnectedError):
        await pending
    assert client.connection_state == "disconnected"

    snapshot = await client.reconnect()
    assert snapshot["revision"] == 2
    assert client.connection_state == "connected"
    assert states == ["connecting", "connected", "disconnected", "connecting", "connected"]


@pytest.mark.tonio
async def test_supports_synchronous_reconnect_from_a_disconnection_listener():
    first = MemoryByteServer()
    second = MemoryByteServer()
    connection = [0]
    for server in (first, second):

        def on_message(message, server=server):
            if message["type"] != "hello":
                return
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": f"connection-{connection[0]}",
                    "snapshot": {**BASE_SERVER_SNAPSHOT, "revision": connection[0]},
                }
            )

        server.on_message(on_message)

    async def transport_factory(handlers):
        index = connection[0]
        connection[0] += 1
        return (first if index == 0 else second).connect(handlers)

    client = PiClient(PiClientOptions(transport_factory=transport_factory))
    await client.connect()
    reconnect = [None]

    def on_state_change(change):
        if change.state == "disconnected":
            reconnect[0] = client.reconnect()

    client.on_connection_state_change(on_state_change)

    first.close()
    assert reconnect[0] is not None
    snapshot = await reconnect[0]
    assert snapshot["revision"] == 2
    assert client.connection_state == "connected"


@pytest.mark.tonio
async def test_rejects_pending_requests_on_transport_errors():
    server = MemoryByteServer()
    client = await connect_client(server)
    pending = client.list_sessions()
    server.error(Exception("read failed"))

    with pytest.raises(PiDisconnectedError, match="read failed"):
        await pending
    assert client.connection_state == "disconnected"


@pytest.mark.tonio
async def test_enforces_the_configured_frame_limit_for_outbound_and_inbound_messages():
    server = MemoryByteServer()

    def on_message(message):
        if message["type"] == "hello":
            server.send(
                {
                    "type": "hello",
                    "version": PROTOCOL_VERSION,
                    "connectionId": "connection-1",
                    "snapshot": BASE_SERVER_SNAPSHOT,
                }
            )

    server.on_message(on_message)

    async def transport_factory(handlers):
        return server.connect(handlers)

    client = PiClient(PiClientOptions(max_frame_length=512, transport_factory=transport_factory))
    await client.connect()
    handle = await attach_session(client, server, session_snapshot("session-1"))
    sent_before = len(server.sent_by_client)
    with pytest.raises(ProtocolValidationError):
        await handle.prompt("x" * 1_000)
    assert len(server.sent_by_client) == sent_before

    server.send_raw(bytes([0, 0, 2, 1]))
    assert client.connection_state == "disconnected"


@pytest.mark.tonio
async def test_disconnects_on_invalid_protocol_data():
    server = MemoryByteServer()
    client = await connect_client(server)
    server.send_raw(encode_frame(encode_cbor({"type": "event", "event": {"type": "session_removed", "sessionId": 1}})))
    assert client.connection_state == "disconnected"


@pytest.mark.tonio
async def test_reports_truncated_framing_when_the_transport_closes():
    server = MemoryByteServer()
    client = await connect_client(server)
    pending = client.list_sessions()
    server.send_raw(bytes([0, 0, 0, 2, 1]))
    server.close()

    with pytest.raises(ProtocolValidationError, match="(?i)truncated"):
        await pending
    assert client.connection_state == "disconnected"


@pytest.mark.tonio
async def test_rejects_frame_limits_outside_the_unsigned_32_bit_range():
    server = MemoryByteServer()

    async def transport_factory(handlers):
        return server.connect(handlers)

    with pytest.raises(TypeError, match="maxFrameLength"):
        PiClient(PiClientOptions(max_frame_length=0x1_0000_0000, transport_factory=transport_factory))
