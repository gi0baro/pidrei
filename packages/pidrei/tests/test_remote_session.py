"""Mirror of pi coding-agent test/client/remote-session.test.ts."""

import pytest

from pidrei.client.remote_session import RemoteSessionOptions
from tests.remote_session_support import (
    MemoryServer,
    collect_requests,
    connect_client,
    open_test_remote_session,
    session_snapshot,
)


@pytest.mark.tonio
async def test_projects_progress_for_subscribers_without_changing_the_authoritative_snapshot():
    server = MemoryServer()
    remote_session = await open_test_remote_session(
        await connect_client(server),
        server,
        session_snapshot(
            "session-1",
            phase="turn",
            transcript=[
                {
                    "id": "assistant-1",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "status": "streaming",
                    "model": {"provider": "faux", "id": "model"},
                    "timestamp": 1,
                }
            ],
        ),
    )
    views: list[str] = []

    def on_state(state) -> None:
        if state.transcript:
            item = state.transcript[0]
            if item["role"] == "assistant" and item["content"][0]["type"] == "text":
                views.append(item["content"][0]["text"])

    remote_session.subscribe(on_state)

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
                    "delta": " world",
                },
            },
        }
    )

    assert views == ["hello", "hello world"]
    assert remote_session.snapshot["transcript"][0]["content"] == [{"type": "text", "text": "hello"}]


@pytest.mark.tonio
async def test_becomes_unbound_and_can_reopen_after_its_session_is_removed():
    server = MemoryServer()
    client = await connect_client(server)
    remote_session = await open_test_remote_session(client, server, session_snapshot("session-1"))

    server.send({"type": "event", "event": {"type": "session_removed", "sessionId": "session-1"}})

    assert remote_session.id is None
    assert remote_session.snapshot is None
    assert remote_session.state.transcript == []
    assert remote_session.state.lifecycle.status == "unbound"

    requests = collect_requests(server)
    reopening = remote_session.open("session-1")
    request = requests[-1] if requests else None
    assert request is not None, "Missing attach request"
    assert request["request"] == {"command": "attach", "sessionId": "session-1"}
    server.send(
        {
            "type": "response",
            "id": request["id"],
            "ok": True,
            "result": {"command": "attach", "session": session_snapshot("session-1", revision=2)},
        }
    )
    await reopening
    assert remote_session.state.lifecycle.status == "ready"


@pytest.mark.tonio
async def test_exposes_the_active_operation_while_prompting():
    server = MemoryServer()
    remote_session = await open_test_remote_session(await connect_client(server), server, session_snapshot("session-1"))
    requests = collect_requests(server)
    lifecycles: list[str] = []

    def on_state(state) -> None:
        lifecycle = state.lifecycle
        lifecycles.append(
            f"{lifecycle.status}:{lifecycle.operation}" if lifecycle.status == "busy" else lifecycle.status
        )

    remote_session.subscribe(on_state)

    prompting = remote_session.submit("  first prompt  ")
    request = requests[-1] if requests else None
    assert request is not None, "Missing prompt request"
    assert request["request"] == {"command": "prompt", "sessionId": "session-1", "text": "first prompt"}
    assert remote_session.operation == "submit"
    server.send(
        {
            "type": "response",
            "id": request["id"],
            "ok": True,
            "result": {"command": "prompt", "session": session_snapshot("session-1", revision=2, phase="turn")},
        }
    )
    await prompting

    assert lifecycles == ["ready", "busy:submit", "busy:submit", "ready"]
    assert remote_session.state.lifecycle.status == "ready"


@pytest.mark.tonio
async def test_steers_when_the_server_session_is_in_a_turn():
    server = MemoryServer()
    remote_session = await open_test_remote_session(
        await connect_client(server), server, session_snapshot("session-1", phase="turn")
    )
    requests = collect_requests(server)

    steering = remote_session.submit("adjust")
    request = requests[-1] if requests else None
    assert request is not None, "Missing steer request"
    assert request["request"] == {"command": "steer", "sessionId": "session-1", "text": "adjust"}
    server.send(
        {
            "type": "response",
            "id": request["id"],
            "ok": True,
            "result": {"command": "steer", "session": session_snapshot("session-1", revision=2, phase="turn")},
        }
    )
    await steering


@pytest.mark.tonio
async def test_aborts_while_a_prompt_response_is_pending():
    server = MemoryServer()
    remote_session = await open_test_remote_session(await connect_client(server), server, session_snapshot("session-1"))
    requests = collect_requests(server)

    prompting = remote_session.submit("hello")
    prompt_request = requests[-1] if requests else None
    assert prompt_request is not None, "Missing prompt request"
    server.send(
        {
            "type": "event",
            "event": {"type": "session_snapshot", "snapshot": session_snapshot("session-1", revision=2, phase="turn")},
        }
    )

    aborting = remote_session.abort()
    abort_request = requests[-1] if requests else None
    assert abort_request is not None, "Missing abort request"
    assert abort_request["request"] == {"command": "abort", "sessionId": "session-1"}
    assert remote_session.operation == "abort"
    server.send(
        {
            "type": "response",
            "id": prompt_request["id"],
            "ok": True,
            "result": {"command": "prompt", "session": session_snapshot("session-1", revision=3, phase="turn")},
        }
    )
    await prompting
    assert remote_session.operation == "abort"
    server.send(
        {
            "type": "response",
            "id": abort_request["id"],
            "ok": True,
            "result": {"command": "abort", "session": session_snapshot("session-1", revision=4)},
        }
    )
    await aborting
    assert remote_session.state.lifecycle.status == "ready"


@pytest.mark.tonio
async def test_rejects_conflicting_operations_while_locally_busy():
    server = MemoryServer()
    remote_session = await open_test_remote_session(await connect_client(server), server, session_snapshot("session-1"))

    requests = collect_requests(server)
    prompting = remote_session.submit("hello")
    with pytest.raises(Exception, match="Remote session is busy with submit"):
        await remote_session.set_thinking("high")
    with pytest.raises(Exception, match="Remote session is busy with submit"):
        await remote_session.open("session-2")
    request = requests[-1] if requests else None
    assert request is not None, "Missing prompt request"
    server.send(
        {
            "type": "response",
            "id": request["id"],
            "ok": True,
            "result": {"command": "prompt", "session": session_snapshot("session-1", revision=2, phase="turn")},
        }
    )
    await prompting


@pytest.mark.tonio
async def test_reports_subscriber_failures_without_interrupting_other_subscribers():
    server = MemoryServer()
    listener_errors: list[Exception] = []
    remote_session = await open_test_remote_session(
        await connect_client(server),
        server,
        session_snapshot("session-1"),
        RemoteSessionOptions(on_listener_error=listener_errors.append),
    )

    def failing_listener(_state) -> None:
        raise Exception("render failed")

    remote_session.subscribe(failing_listener)
    notified = False

    def noting_listener(_state) -> None:
        nonlocal notified
        notified = True

    remote_session.subscribe(noting_listener)

    assert [str(error) for error in listener_errors] == ["render failed"]
    assert notified is True
