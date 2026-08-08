"""Mirror of pi coding-agent test/client/remote-session-lifecycle.test.ts."""

import pytest

from pidrei.client.promise import Deferred
from pidrei.client.remote_session import CreateRemoteSessionOptions
from pidrei_protocol import ClientMessage
from tests.remote_session_support import (
    MemoryServer,
    collect_requests,
    connect_client,
    open_test_remote_session,
    session_snapshot,
)


def next_request(server: MemoryServer, command: str) -> Deferred:
    deferred = Deferred()

    def on_message(message: ClientMessage) -> None:
        if message["type"] != "request" or message["request"]["command"] != command:
            return
        unsubscribe()
        deferred.resolve(message)

    unsubscribe = server.on_message(on_message)
    return deferred


@pytest.mark.tonio
async def test_opens_a_replacement_before_detaching_the_current_session():
    server = MemoryServer()
    remote_session = await open_test_remote_session(await connect_client(server), server, session_snapshot("session-1"))
    requests = collect_requests(server)

    opening = remote_session.open("session-2")
    attach_request = requests[-1] if requests else None
    assert attach_request is not None, "Missing attach request"
    detach_request_deferred = next_request(server, "detach")
    server.send(
        {
            "type": "response",
            "id": attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-2")},
        }
    )
    detach_request = await detach_request_deferred
    assert detach_request["request"] == {"command": "detach", "sessionId": "session-1"}
    server.send(
        {
            "type": "response",
            "id": detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )
    await opening

    assert remote_session.id == "session-2"


@pytest.mark.tonio
async def test_rejects_another_mutation_while_replacement_attachment_is_pending():
    server = MemoryServer()
    remote_session = await open_test_remote_session(await connect_client(server), server, session_snapshot("session-1"))
    requests = collect_requests(server)

    opening = remote_session.open("session-2")
    with pytest.raises(Exception, match="Remote session is busy with open"):
        await remote_session.submit("race")
    with pytest.raises(Exception, match="Remote session is busy with open"):
        await remote_session.create(CreateRemoteSessionOptions(cwd="/other"))
    assert [request["request"] for request in requests] == [{"command": "attach", "sessionId": "session-2"}]

    attach_request = requests[0]
    detach_request_deferred = next_request(server, "detach")
    server.send(
        {
            "type": "response",
            "id": attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-2")},
        }
    )
    detach_request = await detach_request_deferred
    server.send(
        {
            "type": "response",
            "id": detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )
    await opening


@pytest.mark.tonio
async def test_rolls_back_a_replacement_when_the_current_server_session_becomes_active():
    server = MemoryServer()
    remote_session = await open_test_remote_session(await connect_client(server), server, session_snapshot("session-1"))
    requests = collect_requests(server)

    opening = remote_session.open("session-2")
    attach_request = requests[-1] if requests else None
    assert attach_request is not None, "Missing attach request"
    detach_request_deferred = next_request(server, "detach")
    server.send(
        {
            "type": "event",
            "event": {"type": "session_snapshot", "snapshot": session_snapshot("session-1", phase="turn", revision=2)},
        }
    )
    server.send(
        {
            "type": "response",
            "id": attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-2")},
        }
    )
    detach_request = await detach_request_deferred
    assert detach_request["request"] == {"command": "detach", "sessionId": "session-2"}
    server.send(
        {
            "type": "response",
            "id": detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-2"},
        }
    )

    with pytest.raises(Exception, match="Cannot open a session while session is turn"):
        await opening
    assert remote_session.id == "session-1"
    assert remote_session.state.lifecycle.status == "ready"


@pytest.mark.tonio
async def test_dispose_awaits_attachment_cleanup_started_by_reconnect():
    server = MemoryServer()
    client = await connect_client(server)
    remote_session = await open_test_remote_session(client, server, session_snapshot("session-1"))
    client.disconnect("test reconnect")
    attach_request_deferred = next_request(server, "attach")

    reconnecting = remote_session.reconnect()
    attach_request = await attach_request_deferred
    disposing = remote_session.dispose()
    with pytest.raises(Exception, match="Remote session is disposed"):
        await reconnecting
    detach_request_deferred = next_request(server, "detach")
    server.send(
        {
            "type": "response",
            "id": attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1", revision=2)},
        }
    )
    detach_request = await detach_request_deferred
    assert disposing.settled is False
    server.send(
        {
            "type": "response",
            "id": detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )

    await disposing
    assert disposing.settled is True
    assert client.connected is True


@pytest.mark.tonio
async def test_dispose_immediately_preempts_pending_work_and_awaits_attachment_cleanup():
    server = MemoryServer()
    client = await connect_client(server)
    remote_session = await open_test_remote_session(client, server, session_snapshot("session-1"))
    states: list[str] = []
    remote_session.subscribe(lambda state: states.append(state.lifecycle.status))
    requests = collect_requests(server)

    opening = remote_session.open("session-2")
    attach_request = next(
        (request for request in requests if request["request"]["command"] == "attach"),
        None,
    )
    assert attach_request is not None, "Missing attach request"
    disposing = remote_session.dispose()
    current_detach_request = next(
        (
            request
            for request in requests
            if request["request"]["command"] == "detach" and request["request"]["sessionId"] == "session-1"
        ),
        None,
    )
    assert current_detach_request is not None, "Missing current detach request"

    assert client.connected is True
    assert remote_session.state.lifecycle.status == "disposed"
    with pytest.raises(Exception):
        await opening
    replacement_detach_request_deferred = next_request(server, "detach")
    server.send(
        {
            "type": "response",
            "id": attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-2")},
        }
    )
    server.send(
        {
            "type": "response",
            "id": current_detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )
    replacement_detach_request = await replacement_detach_request_deferred
    assert replacement_detach_request["request"] == {"command": "detach", "sessionId": "session-2"}
    server.send(
        {
            "type": "response",
            "id": replacement_detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-2"},
        }
    )
    await disposing
    assert "disposed" in states
    with pytest.raises(Exception, match="Remote session is disposed"):
        remote_session.subscribe(lambda _state: None)
