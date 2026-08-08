"""Port of pi client `test/state.test.ts`."""

import pytest

from tests.support import (
    BASE_SERVER_SNAPSHOT,
    MemoryByteServer,
    attach_session,
    collect_requests,
    connect_client,
    session_snapshot,
)


@pytest.mark.tonio
async def test_reduces_only_authoritative_snapshots_and_supports_unsubscribe():
    server = MemoryByteServer()
    client = await connect_client(server)
    requests = collect_requests(server)
    initial = session_snapshot("session-1", revision=1, phase="idle")
    handle = await attach_session(client, server, initial)
    observed: list[int] = []
    progress_types: list[str] = []
    unsubscribe = handle.subscribe(lambda snapshot: observed.append(snapshot["revision"]))
    unsubscribe_events = handle.on_event(lambda event: progress_types.append(event["type"]))
    server.send(
        {
            "type": "event",
            "event": {
                "type": "session_progress",
                "sessionId": "session-1",
                "progress": {
                    "type": "assistant_delta",
                    "messageId": "assistant-1",
                    "contentIndex": 0,
                    "kind": "text",
                    "delta": "hi",
                },
            },
        }
    )
    assert progress_types == ["session_progress"]
    assert handle.snapshot == initial

    prompting = handle.prompt("hello")
    assert handle.snapshot == initial
    prompt_request = next((request for request in requests if request["request"]["command"] == "prompt"), None)
    assert prompt_request is not None, "Missing prompt request"
    updated = session_snapshot("session-1", revision=2, phase="turn")
    server.send(
        {
            "type": "response",
            "id": prompt_request["id"],
            "ok": True,
            "result": {"command": "prompt", "session": updated},
        }
    )
    assert await prompting == updated
    assert handle.snapshot == updated
    assert observed == [2]

    unsubscribe()
    unsubscribe_events()
    server.send(
        {
            "type": "event",
            "event": {"type": "session_snapshot", "snapshot": session_snapshot("session-1", revision=3)},
        }
    )
    assert observed == [2]


@pytest.mark.tonio
async def test_keeps_session_leases_attached_across_server_metadata_snapshots():
    server = MemoryByteServer()
    client = await connect_client(server)
    handle = await attach_session(client, server, session_snapshot("session-1"))

    server.send(
        {
            "type": "event",
            "event": {
                "type": "server_snapshot",
                "snapshot": {
                    **BASE_SERVER_SNAPSHOT,
                    "revision": 2,
                    "sessions": [{"id": "session-1", "createdAt": 1, "sessionName": "Named session"}],
                },
            },
        }
    )

    assert handle.attached is True


@pytest.mark.tonio
async def test_does_not_let_a_delayed_command_response_replace_a_newer_event_snapshot():
    server = MemoryByteServer()
    client = await connect_client(server)
    initial = session_snapshot("session-1", revision=1, thinkingLevel="off")
    handle = await attach_session(client, server, initial)
    requests = collect_requests(server)
    changing = handle.set_thinking("high")
    request = next((candidate for candidate in requests if candidate["request"]["command"] == "set_thinking"), None)
    assert request is not None, "Missing set_thinking request"
    server.send(
        {
            "type": "event",
            "event": {
                "type": "session_snapshot",
                "snapshot": session_snapshot("session-1", revision=3, thinkingLevel="high"),
            },
        }
    )
    server.send(
        {
            "type": "response",
            "id": request["id"],
            "ok": True,
            "result": {
                "command": "set_thinking",
                "session": session_snapshot("session-1", revision=2, thinkingLevel="medium"),
            },
        }
    )

    await changing
    snapshot = handle.snapshot
    assert snapshot is not None
    assert snapshot["revision"] == 3
    assert snapshot["thinkingLevel"] == "high"


@pytest.mark.tonio
async def test_does_not_let_an_attach_response_replace_a_newer_snapshot_from_the_reacquired_runtime():
    server = MemoryByteServer()
    client = await connect_client(server)
    server.send(
        {
            "type": "event",
            "event": {
                "type": "session_snapshot",
                "snapshot": session_snapshot("session-1", revision=10, attached=False),
            },
        }
    )

    def on_message(message):
        if message["type"] != "request" or message["request"]["command"] != "attach":
            return
        server.send(
            {
                "type": "event",
                "event": {
                    "type": "session_snapshot",
                    "snapshot": session_snapshot("session-1", revision=3, thinkingLevel="high"),
                },
            }
        )
        server.send(
            {
                "type": "response",
                "id": message["id"],
                "ok": True,
                "result": {
                    "command": "attach",
                    "session": session_snapshot("session-1", revision=2, thinkingLevel="medium"),
                },
            }
        )

    server.on_message(on_message)

    handle = await client.attach_session("session-1")
    snapshot = handle.snapshot
    assert snapshot is not None
    assert snapshot["revision"] == 3
    assert snapshot["thinkingLevel"] == "high"
