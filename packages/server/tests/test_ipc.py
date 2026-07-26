"""IPC server/client round-trips over a real unix socket."""

import json
import os

import pytest
import tonio.colored as tonio
from tonio.colored import net

from pidrei_server.config import ENV_SERVER_DIR, get_socket_path
from pidrei_server.ipc.client import send_ipc_request
from pidrei_server.ipc.protocol import encode_message
from pidrei_server.ipc.server import IpcRequestHandler, start_ipc_server

from .server_helpers import env_var


def _make_handler(responses=None, rpc_stream_factory=None):
    async def handle_request(request):
        if request.get("type") == "rpc_stream":
            return {"type": "rpc_ready", "ok": True, "instance": {"id": request["instanceId"], "status": "online"}}
        if request.get("type") == "boom":
            raise Exception("kaboom")
        return {"type": f"{request.get('type')}_result", "ok": True, "echo": request}

    def open_rpc_stream(instance_id, on_response, on_session_event, on_ui_request):
        if rpc_stream_factory is None:
            return None
        return rpc_stream_factory(instance_id, on_response, on_session_event, on_ui_request)

    return IpcRequestHandler(handle_request=handle_request, open_rpc_stream=open_rpc_stream)


async def _read_lines(stream, count, buffer):
    lines = []
    while len(lines) < count:
        newline_index = buffer[0].find("\n")
        if newline_index != -1:
            line = buffer[0][:newline_index].strip()
            buffer[0] = buffer[0][newline_index + 1 :]
            if line:
                lines.append(json.loads(line))
            continue
        chunk = await stream.receive_some()
        if not chunk:
            break
        buffer[0] += chunk.decode("utf-8")
    return lines


class TestIpcRoundTrip:
    @pytest.mark.tonio
    async def test_request_response(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            server = await start_ipc_server(_make_handler())
            try:
                response = await send_ipc_request({"type": "list"})
                assert response == {"type": "list_result", "ok": True, "echo": {"type": "list"}}
            finally:
                server.close()

    @pytest.mark.tonio
    async def test_handler_error_becomes_error_response(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            server = await start_ipc_server(_make_handler())
            try:
                response = await send_ipc_request({"type": "boom"})
                assert response == {"type": "error", "ok": False, "error": "kaboom"}
            finally:
                server.close()

    @pytest.mark.tonio
    async def test_second_server_refuses_to_start(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            server = await start_ipc_server(_make_handler())
            try:
                with pytest.raises(Exception, match="server is already running"):
                    await start_ipc_server(_make_handler())
            finally:
                server.close()

    @pytest.mark.tonio
    async def test_stale_socket_is_removed(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            # A bound-then-closed socket leaves a dead socket file behind.
            import socket as stdlib_socket

            socket_path = get_socket_path()
            os.makedirs(os.path.dirname(socket_path), exist_ok=True)
            dead = stdlib_socket.socket(stdlib_socket.AF_UNIX, stdlib_socket.SOCK_STREAM)
            dead.bind(socket_path)
            dead.close()
            assert os.path.exists(socket_path)

            server = await start_ipc_server(_make_handler())
            try:
                response = await send_ipc_request({"type": "list"})
                assert response["ok"] is True
            finally:
                server.close()


class TestRpcStream:
    @pytest.mark.tonio
    async def test_rpc_stream_flow(self, tmp_dir):
        closed = []

        class FakeStream:
            def __init__(self, instance_id, on_response, on_session_event, on_ui_request):
                self._on_response = on_response
                self._on_session_event = on_session_event
                self._on_ui_request = on_ui_request

            async def handle_request(self, request):
                if request.get("type") == "fail":
                    raise Exception("nope")
                self._on_session_event({"type": "custom_event", "value": 1})
                self._on_response({"type": "response", "command": request.get("type"), "success": True})

            def close(self):
                closed.append(True)

        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            server = await start_ipc_server(_make_handler(rpc_stream_factory=FakeStream))
            try:
                stream = await net.open_unix_socket(get_socket_path())
                buffer = [""]
                try:
                    await stream.send_all(encode_message({"type": "rpc_stream", "instanceId": "i-1"}).encode("utf-8"))
                    (ready,) = await _read_lines(stream, 1, buffer)
                    assert ready["type"] == "rpc_ready"
                    assert ready["instance"]["id"] == "i-1"

                    await stream.send_all(encode_message({"type": "get_state"}).encode("utf-8"))
                    event, response = await _read_lines(stream, 2, buffer)
                    assert event == {"type": "custom_event", "value": 1}
                    assert response["command"] == "get_state"

                    await stream.send_all(encode_message({"type": "fail"}).encode("utf-8"))
                    (error,) = await _read_lines(stream, 1, buffer)
                    assert error == {"type": "error", "ok": False, "error": "nope"}
                finally:
                    stream.close()

                # The server-side connection notices the close and disposes the stream.
                for _ in range(50):
                    if closed:
                        break
                    await tonio.time.sleep(0.02)
                assert closed
            finally:
                server.close()

    @pytest.mark.tonio
    async def test_rpc_stream_unknown_instance(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            server = await start_ipc_server(_make_handler(rpc_stream_factory=None))
            try:
                stream = await net.open_unix_socket(get_socket_path())
                buffer = [""]
                try:
                    await stream.send_all(encode_message({"type": "rpc_stream", "instanceId": "nope"}).encode("utf-8"))
                    (error,) = await _read_lines(stream, 1, buffer)
                    assert error == {"type": "error", "ok": False, "error": "Unknown instance: nope"}
                finally:
                    stream.close()
            finally:
                server.close()
