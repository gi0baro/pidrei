"""Port of pi server `test/conformance.test.ts`."""

import pytest
import tonio.colored as tonio
from tonio.colored import fs

from pidrei_protocol import PROTOCOL_VERSION, encode_client_message, encode_frame
from pidrei_server import InternalServerError, NotImplementedError, PiServerError
from pidrei_server.testing import Deferred, TestServerService
from tests.server_support import Harness, settled


@pytest.mark.tonio
async def test_accepts_a_transport_fragmented_framed_cbor_hello(sock_dir):
    harness = Harness(sock_dir)
    try:
        server, _ = await harness.start_server()
        client = await harness.connect(server)
        response = client.next(lambda message: message["type"] == "hello")
        await client.send_fragmented_message({"type": "hello", "version": PROTOCOL_VERSION}, 2)
        hello = await settled(response, "the hello response")
        assert hello["type"] == "hello"
        assert hello["version"] == PROTOCOL_VERSION
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_enforces_version_and_exactly_one_first_message_hello(sock_dir):
    harness = Harness(sock_dir)
    try:
        server, _ = await harness.start_server()

        bad_version = await harness.connect(server)
        response = await settled(bad_version.hello(PROTOCOL_VERSION + 1), "the version-mismatch hello response")
        assert response["type"] == "hello_error"
        assert response["error"]["code"] == "version"
        await settled(bad_version.wait_for_close(), "the bad-version connection close")

        request_first = await harness.connect(server)
        first_error = request_first.next(lambda message: message["type"] == "hello_error")
        await request_first.send_message({"type": "request", "id": "too-early", "request": {"command": "list"}})
        error = await settled(first_error, "the request-before-hello error")
        assert error["error"]["code"] == "invalid_request"
        await settled(request_first.wait_for_close(), "the request-first connection close")

        duplicate = await harness.connect(server)
        assert (await settled(duplicate.hello(), "the duplicate client's hello response"))["type"] == "hello"
        duplicate_error = duplicate.next(lambda message: message["type"] == "hello_error")
        await duplicate.send_message({"type": "hello", "version": PROTOCOL_VERSION})
        assert (await settled(duplicate_error, "the duplicate-hello error"))["error"]["code"] == "invalid_request"
        await settled(duplicate.wait_for_close(), "the duplicate connection close")
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_closes_connections_that_do_not_complete_hello_before_the_timeout(sock_dir):
    harness = Harness(sock_dir)
    try:
        server, _ = await harness.start_server(handshake_timeout_ms=20)
        client = await harness.connect(server)
        await settled(client.wait_for_close(), "the handshake-timeout close")
        assert any(
            message["type"] == "hello_error" and message["error"]["code"] == "invalid_request"
            for message in client.messages
        )
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_keeps_the_handshake_timeout_active_until_the_server_hello_is_sent(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        delay = service.delay_next_list()
        server, _ = await harness.start_server(service, handshake_timeout_ms=20)
        client = await harness.connect(server)
        await client.send_message({"type": "hello", "version": PROTOCOL_VERSION})
        await settled(delay.entered, "the delayed list_sessions entry")
        await settled(client.wait_for_close(), "the mid-handshake timeout close")
        delay.release.resolve(None)
        assert any(
            message["type"] == "hello_error" and message["error"]["code"] == "invalid_request"
            for message in client.messages
        )
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_bounds_and_closes_malformed_or_oversized_frames(sock_dir):
    harness = Harness(sock_dir)
    try:
        malformed_server, _ = await harness.start_server()
        malformed = await harness.connect(malformed_server)
        malformed_error = malformed.next(lambda message: message["type"] == "hello_error")
        await malformed.send_bytes(encode_frame(bytes([0xFF])))
        assert (await settled(malformed_error, "the malformed-frame error"))["error"]["code"] == "invalid_request"
        await settled(malformed.wait_for_close(), "the malformed connection close")

        bounded_server, _ = await harness.start_server(max_frame_length=128)
        oversized = await harness.connect(bounded_server)
        frame = bytearray(4 + 129)
        frame[3] = 129
        await oversized.send_bytes(bytes(frame))
        await settled(oversized.wait_for_close(), "the oversized connection close")
        assert not any(message["type"] == "hello" for message in oversized.messages)

        outbound_server, _ = await harness.start_server(max_frame_length=128)
        outbound = await harness.connect(outbound_server)
        await outbound.send_message({"type": "hello", "version": PROTOCOL_VERSION})
        await settled(outbound.wait_for_close(), "the outbound-bounded connection close")
        assert outbound.messages == []
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_catches_up_a_handshaking_client_after_a_concurrent_server_change(sock_dir):
    class RacingService(TestServerService):
        def __init__(self):
            super().__init__()
            self.entered = Deferred()
            self.release = Deferred()
            self.race = False

        async def list_sessions(self):
            sessions = await super().list_sessions()
            if not self.race:
                return sessions
            self.entered.resolve(None)
            await self.release
            return sessions

    harness = Harness(sock_dir)
    try:
        service = RacingService()
        service.seed("shared")
        server, _ = await harness.start_server(service)
        controller = await harness.connect(server)
        await settled(controller.hello(), "the controller's hello handshake")
        service.race = True
        joining = await harness.connect(server)
        hello = joining.hello()
        await settled(service.entered, "the raced list_sessions entry")
        await settled(controller.request({"command": "attach", "sessionId": "shared"}), "the attach response")
        service.release.resolve(None)
        handshake = await settled(hello, "the joining client's hello response")
        assert handshake["type"] == "hello"
        catchup = await settled(
            joining.next(
                lambda message: (
                    message["type"] == "event"
                    and message["event"]["type"] == "server_snapshot"
                    and message["event"]["snapshot"]["revision"] > handshake["snapshot"]["revision"]
                )
            ),
            "the catch-up server_snapshot event",
        )
        sessions = catchup["event"]["snapshot"]["sessions"]
        assert [(item["id"], item["sessionName"]) for item in sessions] == [("shared", "Session shared")]
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_shares_request_event_attachment_and_disconnect_behavior(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed("first")
        service.seed("second")
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        hello = await settled(client.hello(), "the hello response")
        assert hello["type"] == "hello"
        assert [item["id"] for item in hello["snapshot"]["sessions"]] == ["first", "second"]

        listed = await settled(client.request({"command": "list"}), "the list response")
        assert listed["ok"] is True
        assert listed["result"]["command"] == "list"
        assert [item["id"] for item in listed["result"]["sessions"]] == ["first", "second"]
        attached_first = await settled(
            client.request({"command": "attach", "sessionId": "first"}), "the first attach response"
        )
        assert attached_first["ok"] is True
        assert attached_first["result"]["session"]["id"] == "first"
        assert attached_first["result"]["session"]["attached"] is True
        attached_second = await settled(
            client.request({"command": "attach", "sessionId": "second"}), "the second attach response"
        )
        assert attached_second["ok"] is True
        assert attached_second["result"]["session"]["attached"] is True

        progress = {
            "type": "assistant_delta",
            "messageId": "assistant-1",
            "contentIndex": 0,
            "kind": "text",
            "delta": "hello",
        }
        progress_event = client.next(
            lambda message: message["type"] == "event" and message["event"]["type"] == "session_progress"
        )
        service.latest_runtime("first").emit_progress(progress)
        assert await settled(progress_event, "the session_progress event") == {
            "type": "event",
            "event": {"type": "session_progress", "sessionId": "first", "progress": progress},
        }

        detached = await settled(client.request({"command": "detach", "sessionId": "first"}), "the detach response")
        assert detached["ok"] is True
        assert detached["result"] == {"command": "detach", "sessionId": "first"}
        assert service.latest_runtime("first").dispose_count == 1
        thinking = await settled(
            client.request({"command": "set_thinking", "sessionId": "second", "thinkingLevel": "high"}),
            "the set_thinking response",
        )
        assert thinking["ok"] is True
        assert thinking["result"]["session"]["id"] == "second"
        assert thinking["result"]["session"]["thinkingLevel"] == "high"

        second_runtime = service.latest_runtime("second")
        await client.close()
        await settled(second_runtime.disposed, "the second runtime's disposal")
        assert second_runtime.dispose_count == 1
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_disconnects_attached_clients_when_a_runtime_reports_a_terminal_error(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed("terminal")
        errors = []
        server, _ = await harness.start_server(service, on_error=errors.append)
        client = await harness.connect(server)
        await settled(client.hello(), "the hello handshake")
        await settled(client.request({"command": "attach", "sessionId": "terminal"}), "the attach response")
        runtime = service.latest_runtime("terminal")

        runtime.set_phase("turn")
        runtime.emit_error(PiServerError("session_locked", "lock ownership lost"))
        await settled(client.wait_for_close(), "the terminal-error disconnect")
        await settled(runtime.disposed, "the terminal runtime's disposal")
        assert runtime.dispose_count == 1
        assert "terminal" not in service.locked
        assert any(getattr(error, "code", None) == "session_locked" for error in errors)

        next_client = await harness.connect(server)
        await settled(next_client.hello(), "the next client's hello handshake")
        reattached = await settled(
            next_client.request({"command": "attach", "sessionId": "terminal"}), "the reattach response"
        )
        assert reattached["ok"] is True
        assert reattached["result"]["session"]["id"] == "terminal"
        assert service.latest_runtime("terminal") is not runtime
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_does_not_expose_unexpected_service_errors_to_clients(sock_dir):
    class FailingService(TestServerService):
        def __init__(self):
            super().__init__()
            self._list_count = 0

        async def list_sessions(self):
            self._list_count += 1
            if self._list_count > 1:
                raise Exception("private service detail")
            return await super().list_sessions()

    harness = Harness(sock_dir)
    try:
        errors = []

        def on_error(error):
            errors.append(error)
            raise Exception("observer failure")

        server, _ = await harness.start_server(FailingService(), on_error=on_error)
        client = await harness.connect(server)
        await settled(client.hello(), "the hello handshake")
        response = await settled(client.request({"command": "list"}), "the list response")
        assert response["ok"] is False
        assert response["error"] == {"code": "internal_error", "message": "Internal server error"}
        assert any(str(error) == "private service detail" for error in errors)
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_keeps_not_implemented_stable(sock_dir):
    class IncompleteService(TestServerService):
        def __init__(self):
            super().__init__()
            self._list_count = 0

        async def list_sessions(self):
            self._list_count += 1
            if self._list_count > 1:
                raise NotImplementedError()
            return await super().list_sessions()

    harness = Harness(sock_dir)
    try:
        server, _ = await harness.start_server(IncompleteService())
        client = await harness.connect(server)
        await settled(client.hello(), "the hello handshake")
        response = await settled(client.request({"command": "list"}), "the list response")
        assert response["ok"] is False
        assert response["error"] == {"code": "not_implemented", "message": "Operation is not implemented"}
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_reports_wrapped_internal_causes_without_exposing_them(sock_dir):
    cause = Exception("private storage detail")
    cause.__cause__ = Exception("private root cause")

    class WrappedFailureService(TestServerService):
        def __init__(self):
            super().__init__()
            self._list_count = 0

        async def list_sessions(self):
            self._list_count += 1
            if self._list_count > 1:
                raise InternalServerError(cause)
            return await super().list_sessions()

    harness = Harness(sock_dir)
    try:
        errors = []
        server, _ = await harness.start_server(WrappedFailureService(), on_error=errors.append)
        client = await harness.connect(server)
        await settled(client.hello(), "the hello handshake")
        response = await settled(client.request({"command": "list"}), "the list response")
        assert response["ok"] is False
        assert response["error"] == {"code": "internal_error", "message": "Internal server error"}
        assert "private" not in str(response)
        assert cause in errors
        assert not any(isinstance(error, InternalServerError) for error in errors)
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_can_respond_out_of_request_order_after_the_handshake(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed("first")
        server, _ = await harness.start_server(service)
        client = await harness.connect(server)
        await settled(client.hello(), "the hello handshake")

        delay = service.delay_next_list()
        slow_task = tonio.spawn(client.request({"command": "list"}, "slow"))
        await settled(delay.entered, "the delayed list_sessions entry")
        fast = await settled(client.request({"command": "attach", "sessionId": "first"}, "fast"), "the fast response")
        assert fast["ok"] is True
        assert fast["id"] == "fast"
        assert fast["result"]["command"] == "attach"
        assert not any(message["type"] == "response" and message["id"] == "slow" for message in client.messages)

        delay.release.resolve(None)
        slow_response = await settled(slow_task, "the slow response")
        assert slow_response["ok"] is True
        assert slow_response["id"] == "slow"
        assert slow_response["result"]["command"] == "list"
        response_ids = [
            message["id"]
            for message in client.messages
            if message["type"] == "response" and message["id"] in ("slow", "fast")
        ]
        assert response_ids == ["fast", "slow"]
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_gracefully_closes_connections_sessions_and_listener_resources(sock_dir):
    harness = Harness(sock_dir)
    try:
        service = TestServerService()
        service.seed("first")
        server, _ = await harness.start_server(service)
        socket_path = server.addresses[0]
        client = await harness.connect(server)
        await settled(client.hello(), "the hello handshake")
        await settled(client.request({"command": "attach", "sessionId": "first"}), "the attach response")
        runtime = service.latest_runtime("first")
        client_closed = client.wait_for_close()

        await server.close()
        await settled(client_closed, "the client close after server shutdown")
        assert runtime.dispose_count == 1
        assert server.addresses == []
        with pytest.raises(FileNotFoundError):
            await fs.Path(socket_path).lstat()
        await server.close()
    finally:
        await harness.close()


@pytest.mark.tonio
async def test_unix_socket_decodes_multiple_framed_requests_from_one_raw_chunk(sock_dir):
    harness = Harness(sock_dir)
    try:
        server, _ = await harness.start_server()
        client = await harness.connect(server)
        await settled(client.hello(), "the hello handshake")
        first = encode_client_message({"type": "request", "id": "first", "request": {"command": "list"}})
        second = encode_client_message({"type": "request", "id": "second", "request": {"command": "list"}})
        combined = first + second
        first_response = client.next(lambda message: message["type"] == "response" and message["id"] == "first")
        second_response = client.next(lambda message: message["type"] == "response" and message["id"] == "second")
        await client.send_bytes(combined)
        assert (await settled(first_response, "the first response"))["ok"] is True
        assert (await settled(second_response, "the second response"))["ok"] is True
    finally:
        await harness.close()
