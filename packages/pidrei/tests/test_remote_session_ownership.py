"""Mirror of pi coding-agent test/client/remote-session-ownership.test.ts."""

import pytest

from pidrei.client.remote_session import CreateRemoteSessionOptions, create_remote_session, open_remote_session
from pidrei_client import PiSessionOwnershipError
from pidrei_protocol import ClientMessage
from tests.remote_session_support import (
    MemoryServer,
    collect_requests,
    connect_client,
    flush,
    session_snapshot,
)


def respond_to_attach(server: MemoryServer, session_id: str) -> None:
    def on_message(message: ClientMessage) -> None:
        if message["type"] != "request" or message["request"]["command"] != "attach":
            return
        server.send(
            {
                "type": "response",
                "id": message["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot(session_id)},
            }
        )

    server.on_message(on_message)


def respond_to_detach(server: MemoryServer) -> None:
    def on_message(message: ClientMessage) -> None:
        if message["type"] != "request" or message["request"]["command"] != "detach":
            return
        server.send(
            {
                "type": "response",
                "id": message["id"],
                "ok": True,
                "result": {"command": "detach", "sessionId": message["request"]["sessionId"]},
            }
        )

    server.on_message(on_message)


@pytest.mark.tonio
async def test_factory_opens_a_session_and_disposal_awaits_detach_without_disconnecting_the_borrowed_client():
    server = MemoryServer()
    client = await connect_client(server)
    respond_to_attach(server, "session-1")
    remote_session = await open_remote_session(client, "session-1")
    requests = collect_requests(server)

    first_disposal = remote_session.dispose()
    second_disposal = remote_session.dispose()

    assert second_disposal is first_disposal
    await flush()
    assert first_disposal.settled is False
    detach_request = requests[-1] if requests else None
    assert detach_request is not None, "Missing detach request"
    assert detach_request["request"] == {"command": "detach", "sessionId": "session-1"}
    server.send(
        {
            "type": "response",
            "id": detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )
    await first_disposal
    assert remote_session.disposed is True
    assert client.connected is True


@pytest.mark.tonio
async def test_rejects_an_exclusive_coordinator_while_a_direct_shared_lease_is_active():
    server = MemoryServer()
    client = await connect_client(server)
    respond_to_attach(server, "session-1")
    respond_to_detach(server)
    direct_handle = await client.attach_session("session-1")
    requests = collect_requests(server)

    with pytest.raises(PiSessionOwnershipError):
        await open_remote_session(client, "session-1")

    assert requests == []
    assert direct_handle.active is True
    await direct_handle.detach()
    assert [request["request"]["command"] for request in requests] == ["detach"]


@pytest.mark.tonio
async def test_factory_creates_a_session():
    server = MemoryServer()
    client = await connect_client(server)

    def on_message(message: ClientMessage) -> None:
        if message["type"] != "request" or message["request"]["command"] != "create":
            return
        server.send(
            {
                "type": "response",
                "id": message["id"],
                "ok": True,
                "result": {"command": "create", "session": session_snapshot("session-1")},
            }
        )

    server.on_message(on_message)

    remote_session = await create_remote_session(client, CreateRemoteSessionOptions(cwd="/workspace"))

    assert remote_session.id == "session-1"
    assert remote_session.state.lifecycle.status == "ready"


@pytest.mark.tonio
async def test_disposal_reports_cleanup_failure_without_retaining_exclusive_ownership():
    server = MemoryServer()
    client = await connect_client(server)
    respond_to_attach(server, "session-1")
    remote_session = await open_remote_session(client, "session-1")
    detach_count = 0

    def on_message(message: ClientMessage) -> None:
        nonlocal detach_count
        if message["type"] != "request" or message["request"]["command"] != "detach":
            return
        detach_count += 1
        if detach_count == 1:
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": False,
                    "error": {"code": "invalid_request", "message": "no"},
                }
            )
        else:
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "detach", "sessionId": "session-1"},
                }
            )

    server.on_message(on_message)

    with pytest.raises(Exception, match="no"):
        await remote_session.dispose()
    assert remote_session.disposed is True
    assert client.connected is True

    replacement = await open_remote_session(client, "session-1")
    assert replacement.state.lifecycle.status == "ready"
    assert detach_count == 2


@pytest.mark.tonio
async def test_multiple_sessions_borrow_one_client_independently():
    server = MemoryServer()
    client = await connect_client(server)

    def on_message(message: ClientMessage) -> None:
        if message["type"] != "request":
            return
        if message["request"]["command"] == "attach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "attach", "session": session_snapshot(message["request"]["sessionId"])},
                }
            )
        if message["request"]["command"] == "detach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "detach", "sessionId": message["request"]["sessionId"]},
                }
            )

    server.on_message(on_message)

    first = await open_remote_session(client, "session-1")
    second = await open_remote_session(client, "session-2")
    await first.dispose()

    assert first.disposed is True
    assert second.disposed is False
    assert client.connected is True
    await second.dispose()


@pytest.mark.tonio
async def test_session_disposal_treats_a_client_first_disposal_as_released():
    server = MemoryServer()
    client = await connect_client(server)
    respond_to_attach(server, "session-1")
    remote_session = await open_remote_session(client, "session-1")
    requests = collect_requests(server)

    await client.dispose()
    # pi's `remoteSession[Symbol.asyncDispose]()`.
    await remote_session.__aexit__(None, None, None)

    assert remote_session.disposed is True
    assert requests == []
