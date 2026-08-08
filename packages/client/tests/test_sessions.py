"""Port of pi client `test/sessions.test.ts`."""

import pytest
import tonio.colored as tonio

from pidrei_client import AcquireSessionOptions, PiSessionDetachedError, PiSessionOwnershipError
from tests.support import MemoryByteServer, connect_client, flush, session_snapshot


def _auto_respond(server: MemoryByteServer, session_id: str | None = None) -> None:
    def on_message(message):
        if message["type"] != "request":
            return
        request = message["request"]
        if request["command"] == "attach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "attach", "session": session_snapshot(session_id or request["sessionId"])},
                }
            )
        if request["command"] == "detach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "detach", "sessionId": session_id or request["sessionId"]},
                }
            )

    server.on_message(on_message)


@pytest.mark.tonio
async def test_keeps_multiple_session_handles_independent_and_enforces_detach():
    server = MemoryByteServer()
    client = await connect_client(server)
    _auto_respond(server)

    first = await client.attach_session("session-1")
    second = await client.attach_session("session-2")
    assert first.attached is True
    assert second.attached is True
    await first.detach()
    assert first.attached is False
    assert second.attached is True
    with pytest.raises(PiSessionDetachedError):
        await first.abort()


@pytest.mark.tonio
async def test_detaches_a_shared_session_only_after_its_final_lease_is_released():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests: list[str] = []

    def on_message(message):
        if message["type"] == "request":
            requests.append(message["request"]["command"])

    server.on_message(on_message)
    _auto_respond(server, "session-1")

    first = await client.attach_session("session-1")
    second = await client.attach_session("session-1")
    assert second is not first
    assert requests == ["attach"]

    await first.detach()
    assert first.attached is False
    assert second.attached is True
    assert requests == ["attach"]

    await second.detach()
    assert second.attached is False
    assert requests == ["attach", "detach"]


@pytest.mark.tonio
async def test_enforces_exclusive_and_shared_lease_modes():
    server = MemoryByteServer()
    client = await connect_client(server)
    _auto_respond(server, "session-1")

    shared = await client.acquire_session("session-1", AcquireSessionOptions(mode="shared"))
    with pytest.raises(PiSessionOwnershipError):
        await client.acquire_session("session-1", AcquireSessionOptions(mode="exclusive"))
    await shared.dispose()

    exclusive = await client.acquire_session("session-1", AcquireSessionOptions(mode="exclusive"))
    with pytest.raises(PiSessionOwnershipError):
        await client.acquire_session("session-1", AcquireSessionOptions(mode="shared"))
    # pi: `await exclusive[Symbol.asyncDispose]()`
    async with exclusive:
        pass
    assert exclusive.active is False


@pytest.mark.tonio
async def test_invalidated_leases_dispose_without_protocol_cleanup():
    server = MemoryByteServer()
    client = await connect_client(server)

    def on_message(message):
        if message["type"] != "request" or message["request"]["command"] != "attach":
            return
        server.send(
            {
                "type": "response",
                "id": message["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-1")},
            }
        )

    server.on_message(on_message)
    lease = await client.acquire_session("session-1", AcquireSessionOptions(mode="exclusive"))

    client.disconnect()

    assert await lease.dispose() is None
    assert lease.active is False


@pytest.mark.tonio
async def test_rejects_commands_while_releasing_and_restores_an_explicit_detach_after_failure():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests: list[dict] = []

    def on_message(message):
        if message["type"] == "request":
            requests.append({"id": message["id"], "command": message["request"]["command"]})

    server.on_message(on_message)
    acquiring = client.acquire_session("session-1", AcquireSessionOptions(mode="exclusive"))
    attach_request = requests[-1] if requests else None
    assert attach_request is not None, "Missing attach request"
    server.send(
        {
            "type": "response",
            "id": attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1")},
        }
    )
    lease = await acquiring

    first_detach = lease.detach()
    failed_detach_request = requests[-1]
    with pytest.raises(PiSessionDetachedError):
        await lease.abort()
    assert failed_detach_request["command"] == "detach", "Missing detach request"
    server.send(
        {
            "type": "response",
            "id": failed_detach_request["id"],
            "ok": False,
            "error": {"code": "invalid_request", "message": "retry"},
        }
    )
    with pytest.raises(Exception, match="retry"):
        await first_detach
    assert lease.active is True

    second_detach = lease.detach()
    successful_detach_request = requests[-1]
    assert successful_detach_request["command"] == "detach", "Missing retry detach request"
    server.send(
        {
            "type": "response",
            "id": successful_detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )
    await second_detach
    assert lease.active is False


@pytest.mark.tonio
async def test_serializes_reacquisition_behind_final_lease_detachment():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests: list[dict] = []

    def on_message(message):
        if message["type"] == "request":
            requests.append({"id": message["id"], "command": message["request"]["command"]})

    server.on_message(on_message)

    first_attachment = client.attach_session("session-1")
    first_attach_request = requests[-1] if requests else None
    assert first_attach_request is not None, "Missing first attach request"
    server.send(
        {
            "type": "response",
            "id": first_attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1")},
        }
    )
    first = await first_attachment
    detaching = first.detach()
    detach_request = requests[-1]
    assert detach_request["command"] == "detach", "Missing detach request"
    reacquiring = tonio.spawn(client.attach_session("session-1"))
    await flush()
    assert [request["command"] for request in requests] == ["attach", "detach"]

    server.send(
        {
            "type": "response",
            "id": detach_request["id"],
            "ok": True,
            "result": {"command": "detach", "sessionId": "session-1"},
        }
    )
    await detaching
    await flush()
    second_attach_request = requests[-1]
    assert second_attach_request["command"] == "attach", "Missing second attach request"
    server.send(
        {
            "type": "response",
            "id": second_attach_request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1", revision=2)},
        }
    )

    reacquired = await reacquiring
    assert reacquired.attached is True


@pytest.mark.tonio
async def test_accepts_a_lower_revision_after_detaching_and_reacquiring_the_same_session():
    server = MemoryByteServer()
    client = await connect_client(server)
    attach_count = 0

    def on_message(message):
        nonlocal attach_count
        if message["type"] != "request":
            return
        request = message["request"]
        if request["command"] == "attach":
            attach_count += 1
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {
                        "command": "attach",
                        "session": session_snapshot("session-1", revision=10 if attach_count == 1 else 0),
                    },
                }
            )
        if request["command"] == "detach":
            server.send(
                {
                    "type": "response",
                    "id": message["id"],
                    "ok": True,
                    "result": {"command": "detach", "sessionId": "session-1"},
                }
            )

    server.on_message(on_message)

    first = await client.attach_session("session-1")
    assert first.snapshot is not None and first.snapshot["revision"] == 10
    await first.detach()
    reopened = await client.attach_session("session-1")
    assert reopened is not first
    assert reopened.snapshot is not None and reopened.snapshot["revision"] == 0
