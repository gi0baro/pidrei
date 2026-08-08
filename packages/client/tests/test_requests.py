"""Port of pi client `test/requests.test.ts`."""

import pytest

from pidrei_client import PiServerError
from pidrei_protocol import ProtocolValidationError
from tests.support import MemoryByteServer, collect_requests, connect_client, session_snapshot


@pytest.mark.tonio
async def test_correlates_coalesced_out_of_order_responses():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = collect_requests(server)
    listed = client.list_sessions()
    attached = client.attach_session("session-1")
    assert len(requests) == 2

    attach_request = next(request for request in requests if request["request"]["command"] == "attach")
    list_request = next(request for request in requests if request["request"]["command"] == "list")
    server.send_together(
        [
            {
                "type": "response",
                "id": attach_request["id"],
                "ok": True,
                "result": {"command": "attach", "session": session_snapshot("session-1")},
            },
            {
                "type": "response",
                "id": list_request["id"],
                "ok": True,
                "result": {"command": "list", "sessions": []},
            },
        ]
    )

    assert await listed == []
    handle = await attached
    assert handle.id == "session-1"
    assert handle.attached is True


@pytest.mark.tonio
async def test_rejects_a_mismatched_response_instead_of_leaving_its_request_pending():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = collect_requests(server)
    listed = client.list_sessions()
    assert len(requests) == 1
    assert requests[0]["request"]["command"] == "list"
    server.send(
        {
            "type": "response",
            "id": requests[0]["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1")},
        }
    )

    with pytest.raises(ProtocolValidationError, match="Response command attach does not match list"):
        await listed
    assert client.connection_state == "disconnected"


@pytest.mark.tonio
async def test_surfaces_typed_request_errors():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = collect_requests(server)
    attaching = client.attach_session("locked")
    server.send(
        {
            "type": "response",
            "id": requests[0]["id"] if requests else "missing",
            "ok": False,
            "error": {"code": "session_locked", "message": "Already attached"},
        }
    )
    with pytest.raises(PiServerError) as excinfo:
        await attaching
    assert excinfo.value.code == "session_locked"
