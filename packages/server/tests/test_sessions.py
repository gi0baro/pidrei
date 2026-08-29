"""Port of pi server `test/sessions.test.ts`."""

from dataclasses import replace

import pytest
import tonio.colored as tonio

from pidrei_server.testing import TEST_MODEL, Deferred, TestServerService
from tests.server_support import Harness, flush


async def settled(deferred: Deferred, what: str, timeout: float = 5.0):
    """Await `deferred` with a bound: a wedged choreography step fails with a
    name instead of hanging the whole CI job (seen on linux 3.14t)."""
    value, completed = await tonio.time.timeout(deferred.wait(), timeout)
    assert completed, f"timed out waiting for {what}"
    return value


MODEL = TEST_MODEL


async def attach(client, session_id):
    response = await client.request({"command": "attach", "sessionId": session_id})
    assert response["ok"] is True
    assert response["result"]["command"] == "attach"
    return response["result"]["session"]


@pytest.mark.tonio
async def test_serializes_server_snapshot_revisions(sock_dir):
    class OrderedSnapshotService(TestServerService):
        def __init__(self):
            super().__init__()
            self.first_started = Deferred()
            self.second_started = Deferred()
            self.first_release = Deferred()
            self.second_release = Deferred()
            self.controlled = False
            self.started_count = 0

        async def list_models(self):
            if not self.controlled:
                return await super().list_models()
            self.started_count += 1
            if self.started_count == 1:
                self.first_started.resolve(None)
                await self.first_release
            elif self.started_count == 2:
                self.second_started.resolve(None)
                await self.second_release
            return [MODEL]

    harness = Harness(sock_dir)
    try:
        service = OrderedSnapshotService()
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        await client.hello()
        service.controlled = True
        message_index = len(client.messages)

        first_create = tonio.spawn(client.request({"command": "create", "name": "first"}))
        await settled(service.first_started, "the first broadcast to reach list_models")
        second_create = tonio.spawn(client.request({"command": "create", "name": "second"}))
        # `sleep(0)` is not a guaranteed reschedule (see `flush`); give the
        # second broadcast real turns to (wrongly) reach list_models.
        await flush()
        assert service.started_count == 1

        service.first_release.resolve(None)
        await settled(service.second_started, "the second broadcast to reach list_models")
        service.second_release.resolve(None)
        await first_create
        await second_create
        await settled(
            client.next_from(
                message_index,
                lambda message: (
                    message["type"] == "event"
                    and message["event"]["type"] == "server_snapshot"
                    and message["event"]["snapshot"]["revision"] == 2
                ),
            ),
            "the revision-2 server snapshot",
        )

        revisions = [
            message["event"]["snapshot"]["revision"]
            for message in client.messages[message_index:]
            if message["type"] == "event" and message["event"]["type"] == "server_snapshot"
        ]
        assert revisions == [1, 2]
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_creates_server_assigned_durable_ids_and_supports_list_attach_and_detach(sock_dir):
    harness = Harness(sock_dir)
    try:
        server, service = await harness.start_server()
        client = await harness.connect(server)
        await client.hello()
        created = await client.request({"command": "create", "cwd": "/work", "name": "Created"})
        assert created["ok"] is True
        assert created["result"]["command"] == "create"
        created_id = created["result"]["session"]["id"]
        assert created_id == service.last_created_id
        assert created["result"]["session"]["cwd"] == "/work"
        assert created["result"]["session"]["name"] == "Created"
        assert created["result"]["session"]["attached"] is True
        assert created["result"]["session"]["locked"] is True

        listed = await client.request({"command": "list"})
        assert listed["ok"] is True
        assert listed["result"]["command"] == "list"
        assert listed["result"]["sessions"] == [
            {
                "id": service.last_created_id,
                "createdAt": 1,
                "updatedAt": 1,
                "sessionName": "Created",
                "cwd": "/work",
            }
        ]
        detached = await client.request({"command": "detach", "sessionId": created_id})
        assert detached["ok"] is True
        assert detached["result"] == {"command": "detach", "sessionId": created_id}
        assert service.latest_runtime(created_id).dispose_count == 1
        detached_again = await client.request({"command": "detach", "sessionId": created_id})
        assert detached_again["ok"] is True
        assert detached_again["result"] == {"command": "detach", "sessionId": created_id}

        attached = await attach(client, created_id)
        assert attached["id"] == service.last_created_id
        assert len(service.runtimes[created_id]) == 2
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_preserves_backend_metadata_while_refreshing_live_session_metadata(sock_dir):
    class ExtendedMetadataService(TestServerService):
        async def list_sessions(self):
            return [
                {**metadata, "parentSessionId": "parent-1", "sessionName": "stale name"}
                for metadata in await super().list_sessions()
            ]

    harness = Harness(sock_dir)
    try:
        service = ExtendedMetadataService()
        service.seed("session-1", "Live name")
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        await client.hello()
        await attach(client, "session-1")

        listed = await client.request({"command": "list"})
        assert listed["ok"] is True
        assert listed["result"]["sessions"] == [
            {
                "id": "session-1",
                "createdAt": 1,
                "updatedAt": 1,
                "parentSessionId": "parent-1",
                "sessionName": "Live name",
                "cwd": "/tmp/pi-server-conformance",
            }
        ]
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_keeps_multiple_attachments_on_one_connection_independent(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed("first")
        service.seed("second")
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        await client.hello()
        await attach(client, "first")
        await attach(client, "second")

        await client.request({"command": "detach", "sessionId": "first"})
        assert service.latest_runtime("first").dispose_count == 1
        assert service.latest_runtime("second").dispose_count == 0
        response = await client.request({"command": "set_thinking", "sessionId": "second", "thinkingLevel": "medium"})
        assert response["ok"] is True
        assert response["result"]["session"]["id"] == "second"
        assert response["result"]["session"]["thinkingLevel"] == "medium"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_broadcasts_full_snapshots_and_progress_only_to_clients_attached_to_that_session(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed()
        server, _ = await harness.start_server(service)
        attached_client = await harness.connect(server)
        unattached_client = await harness.connect(server)
        await attached_client.hello()
        await unattached_client.hello()
        await attach(attached_client, "session-1")
        runtime = service.latest_runtime("session-1")
        progress = {
            "type": "assistant_delta",
            "messageId": "assistant-1",
            "contentIndex": 0,
            "kind": "text",
            "delta": "hello",
        }
        runtime.emit_progress(progress)
        progress_message = await attached_client.next(
            lambda message: message["type"] == "event" and message["event"]["type"] == "session_progress"
        )
        assert progress_message == {
            "type": "event",
            "event": {"type": "session_progress", "sessionId": "session-1", "progress": progress},
        }
        assert not any(
            message["type"] == "event" and message["event"]["type"] == "session_progress"
            for message in unattached_client.messages
        )

        message_count = len(attached_client.messages)
        runtime.emit_snapshot()
        current_revision = (await runtime.snapshot())["revision"]
        snapshot_message = await attached_client.next_from(
            message_count,
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["revision"] == current_revision
            ),
        )
        assert snapshot_message["event"]["snapshot"]["id"] == "session-1"
        assert snapshot_message["event"]["snapshot"]["attached"] is True
        assert snapshot_message["event"]["snapshot"]["locked"] is True
        assert not any(
            message["type"] == "event" and message["event"]["type"] == "session_snapshot"
            for message in unattached_client.messages
        )
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_allows_every_attached_client_to_control_a_singleton_live_runtime(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed()
        server, _ = await harness.start_server(service)
        first = await harness.connect(server)
        second = await harness.connect(server)
        await first.hello()
        await second.hello()
        await attach(first, "session-1")
        second_list = await second.request({"command": "list"})
        assert second_list["ok"] is True
        assert second_list["result"]["sessions"] == [
            {
                "id": "session-1",
                "createdAt": 1,
                "updatedAt": 1,
                "sessionName": "Session session-1",
                "cwd": "/tmp/pi-server-conformance",
            }
        ]
        await attach(second, "session-1")
        assert len(service.runtimes["session-1"]) == 1

        model_response = await second.request(
            {"command": "set_model", "sessionId": "session-1", "model": {"provider": "test", "id": "large"}}
        )
        assert model_response["ok"] is True
        assert model_response["result"]["session"]["model"]["id"] == "large"
        await first.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["model"]["id"] == "large"
            )
        )
        thinking_response = await first.request(
            {"command": "set_thinking", "sessionId": "session-1", "thinkingLevel": "high"}
        )
        assert thinking_response["ok"] is True
        assert thinking_response["result"]["session"]["thinkingLevel"] == "high"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_does_not_queue_prompts_and_processes_steer_and_abort_while_a_prompt_response_is_pending(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed()
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        await client.hello()
        await attach(client, "session-1")

        prompt = tonio.spawn(client.request({"command": "prompt", "sessionId": "session-1", "text": "first"}))
        await client.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["phase"] == "turn"
            )
        )
        busy = await client.request({"command": "prompt", "sessionId": "session-1", "text": "second"})
        assert busy["ok"] is False
        assert busy["error"]["code"] == "busy"

        steer = await client.request({"command": "steer", "sessionId": "session-1", "text": "adjust"})
        assert steer["ok"] is True
        assert steer["result"]["command"] == "steer"
        assert [item.text for item in service.latest_runtime("session-1").steers] == ["adjust"]
        abort = await client.request({"command": "abort", "sessionId": "session-1"})
        assert abort["ok"] is True
        assert abort["result"]["command"] == "abort"
        prompt_response = await prompt
        assert prompt_response["ok"] is True
        assert prompt_response["result"]["command"] == "prompt"
        assert prompt_response["result"]["session"]["phase"] == "idle"
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_returns_operation_attachment_state_relative_to_the_requesting_connection(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed()
        server, _ = await harness.start_server(service)
        first = await harness.connect(server)
        second = await harness.connect(server)
        await first.hello()
        await second.hello()
        await attach(first, "session-1")
        await attach(second, "session-1")

        prompt = tonio.spawn(first.request({"command": "prompt", "sessionId": "session-1", "text": "hello"}))
        await first.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["phase"] == "turn"
            )
        )
        await first.request({"command": "detach", "sessionId": "session-1"})
        service.latest_runtime("session-1").finish_prompt()

        prompt_response = await prompt
        assert prompt_response["ok"] is True
        assert prompt_response["result"]["session"]["id"] == "session-1"
        assert prompt_response["result"]["session"]["attached"] is False
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_keeps_busy_work_alive_after_disconnect_and_disposes_when_it_next_becomes_idle(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed()
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        await client.hello()
        await attach(client, "session-1")
        prompt = tonio.spawn(client.request({"command": "prompt", "sessionId": "session-1", "text": "survive"}))
        await client.next(
            lambda message: (
                message["type"] == "event"
                and message["event"]["type"] == "session_snapshot"
                and message["event"]["snapshot"]["phase"] == "turn"
            )
        )
        runtime = service.latest_runtime("session-1")
        await client.close()
        with pytest.raises(Exception):
            await prompt
        assert runtime.dispose_count == 0
        runtime.finish_prompt()
        await runtime.disposed
        assert runtime.dispose_count == 1

        reconnect = await harness.connect(server)
        await reconnect.hello()
        snapshot = await attach(reconnect, "session-1")
        assert len(snapshot["transcript"]) == 2
        assert snapshot["transcript"][1]["role"] == "assistant"
        assert snapshot["transcript"][1]["content"] == [{"type": "text", "text": "reply:survive"}]
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_restores_persisted_sessions_lazily_after_a_server_restart(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed()
        first_server, _ = await harness.start_server(service)
        first_client = await harness.connect(first_server)
        await first_client.hello()
        await attach(first_client, "session-1")
        await first_client.request({"command": "set_thinking", "sessionId": "session-1", "thinkingLevel": "high"})
        await first_client.close()
        await first_server.close()

        second_server, _ = await harness.start_server(service)
        assert len(service.runtimes["session-1"]) == 1
        second_client = await harness.connect(second_server)
        await second_client.hello()
        restored = await attach(second_client, "session-1")
        assert restored["thinkingLevel"] == "high"
        assert len(service.runtimes["session-1"]) == 2
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_rejects_and_disposes_a_service_runtime_with_the_wrong_server_assigned_id(sock_dir):
    class WrongIdService(TestServerService):
        async def create_session(self, options):
            return await super().create_session(replace(options, id="wrong-id"))

    harness = Harness(sock_dir)
    try:
        service = WrongIdService()
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        await client.hello()
        response = await client.request({"command": "create"})
        assert response["ok"] is False
        assert response["error"]["code"] == "invalid_request"
        assert service.latest_runtime("wrong-id").dispose_count == 1
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_maps_service_lock_errors_and_rejects_control_from_unattached_clients(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed("locked")
        service.locked.add("locked")
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        await client.hello()
        locked = await client.request({"command": "attach", "sessionId": "locked"})
        assert locked["ok"] is False
        assert locked["error"]["code"] == "session_locked"
        unattached = await client.request({"command": "abort", "sessionId": "locked"})
        assert unattached["ok"] is False
        assert unattached["error"]["code"] == "invalid_request"
    finally:
        await harness.close()
