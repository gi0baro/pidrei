"""Port of pi client `test/disposal.test.ts`."""

import pytest

from pidrei_client import PiClient, PiClientDisposedError, PiClientOptions
from pidrei_protocol import PROTOCOL_VERSION
from tests.support import BASE_SERVER_SNAPSHOT, MemoryByteServer, attach_session, connect_client, session_snapshot


@pytest.mark.tonio
async def test_connects_through_its_ownership_factory():
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

    async def transport_factory(handlers):
        return server.connect(handlers)

    # pi: `PiClient.connect(options)` (static); Python cannot overload the
    # instance method of the same name.
    client = await PiClient.open(PiClientOptions(transport_factory=transport_factory))

    assert client.connected is True
    await client.dispose()


@pytest.mark.tonio
async def test_disconnects_invalidates_child_handles_and_rejects_pending_requests():
    server = MemoryByteServer()
    client = await connect_client(server)
    handle = await attach_session(client, server, session_snapshot("session-1"))
    pending = client.list_sessions()

    first_disposal = client.dispose()
    second_disposal = client.dispose()

    # pi asserts promise identity; here both awaitables share the memoized
    # disposal and complete idempotently.
    assert client.disposed is True
    assert client.connected is False
    assert handle.attached is False
    with pytest.raises(PiClientDisposedError):
        await pending
    with pytest.raises(PiClientDisposedError):
        await handle.prompt("after disposal")
    await first_disposal
    await second_disposal


@pytest.mark.tonio
async def test_supports_explicit_async_disposal():
    server = MemoryByteServer()
    client = await connect_client(server)

    # pi: `await client[Symbol.asyncDispose]()`
    async with client:
        pass

    assert client.disposed is True
    assert client.connection_state == "disconnected"
