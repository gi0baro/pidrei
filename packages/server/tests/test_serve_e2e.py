"""End-to-end: the pidrei-server console script serving over the unix socket.

The spawn test drives the full stack: serve subprocess -> supervisor ->
`python -m pidrei --mode rpc` child (fake auth key, offline)."""

import json
import os
import signal
import subprocess
import sys

import pytest
import tonio.colored as tonio

from pidrei.config import ENV_AGENT_DIR
from pidrei_server.config import ENV_SERVER_DIR, get_socket_path
from pidrei_server.ipc.client import send_ipc_request

from .server_helpers import env_var


def _server_script() -> str:
    return os.path.join(os.path.dirname(sys.executable), "pidrei-server")


async def _read_until(stream, needle: str, collected: list[str], timeout_s: float = 30.0) -> bool:
    async def pump() -> bool:
        while True:
            chunk = await stream.receive_some()
            if not chunk:
                return False
            collected.append(chunk.decode("utf-8", "replace"))
            if needle in "".join(collected):
                return True

    result, completed = await tonio.time.timeout(pump(), timeout_s)
    return bool(completed and result)


async def _start_server(tmp_dir, extra_env=None):
    env = {
        **os.environ,
        ENV_SERVER_DIR: str(tmp_dir),
        "PIDREI_OFFLINE": "1",
        **(extra_env or {}),
    }
    process = await tonio.open_process(
        [_server_script(), "serve"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    out: list[str] = []
    ready = await _read_until(process.stdout, "server listening on", out)
    if not ready:
        stderr = b""
        try:
            stderr = await process.stderr.receive_some()
        except Exception:
            pass
        process.kill()
        await process.wait()
        raise Exception(f"server did not start. Stdout: {''.join(out)} Stderr: {stderr.decode('utf-8', 'replace')}")
    return process


async def _stop_server(process) -> int:
    process.send_signal(signal.SIGTERM)
    result, completed = await tonio.time.timeout(process.wait(), 30.0)
    if not completed:
        process.kill()
        await process.wait()
        raise Exception("server did not shut down on SIGTERM")
    return result


class TestServeEndToEnd:
    @pytest.mark.tonio
    async def test_serve_list_status_and_shutdown(self, tmp_dir):
        process = await _start_server(tmp_dir)
        try:
            with env_var(ENV_SERVER_DIR, str(tmp_dir)):
                response = await send_ipc_request({"type": "list"})
                assert response == {"type": "list_result", "ok": True, "instances": []}

                response = await send_ipc_request({"type": "status", "instanceId": "nope"})
                assert response == {"type": "error", "ok": False, "error": "Unknown instance: nope"}

                response = await send_ipc_request({"type": "stop", "instanceId": "nope"})
                assert response == {"type": "error", "ok": False, "error": "Unknown instance: nope"}

                socket_path = get_socket_path()
                assert os.path.exists(socket_path)
        finally:
            exit_code = await _stop_server(process)
        assert exit_code == 0
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            assert not os.path.exists(get_socket_path())

    @pytest.mark.tonio
    async def test_spawn_rpc_and_stop_real_instance(self, tmp_dir):
        temp_root = os.path.realpath(str(tmp_dir))
        agent_dir = os.path.join(temp_root, "agent")
        project_dir = os.path.join(temp_root, "project")
        server_dir = os.path.join(temp_root, "server")
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(agent_dir, "auth.json"), "w", encoding="utf-8") as handle:
            json.dump({"anthropic": {"type": "api_key", "key": "test-key"}}, handle)

        process = await _start_server(server_dir, extra_env={ENV_AGENT_DIR: agent_dir})
        try:
            with env_var(ENV_SERVER_DIR, server_dir):
                response = await send_ipc_request({"type": "spawn", "cwd": project_dir, "label": "e2e"})
                assert response["type"] == "spawn_result", response
                assert response["ok"] is True
                instance = response["instance"]
                assert instance["status"] == "online"
                assert instance["label"] == "e2e"
                assert instance["cwd"] == project_dir
                assert instance["sessionId"]

                response = await send_ipc_request({"type": "list"})
                assert [entry["id"] for entry in response["instances"]] == [instance["id"]]

                response = await send_ipc_request(
                    {"type": "rpc", "instanceId": instance["id"], "command": {"type": "get_state"}}
                )
                assert response["type"] == "rpc_result", response
                assert response["response"]["success"] is True
                assert response["response"]["data"]["sessionId"] == instance["sessionId"]

                response = await send_ipc_request({"type": "stop", "instanceId": instance["id"]})
                assert response == {"type": "stop_result", "ok": True, "instanceId": instance["id"]}

                response = await send_ipc_request({"type": "list"})
                assert response["instances"] == []
        finally:
            exit_code = await _stop_server(process)
        assert exit_code == 0


class TestCliBasics:
    @pytest.mark.tonio
    async def test_version_and_help_exit_cleanly(self, tmp_dir):
        for args, expect in ((["--version"], "0.1.0"), (["--help"], "Usage:")):
            process = await tonio.open_process(
                [_server_script(), *args],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out: list[bytes] = []
            while True:
                chunk = await process.stdout.receive_some()
                if not chunk:
                    break
                out.append(chunk)
            result, completed = await tonio.time.timeout(process.wait(), 15.0)
            if not completed:
                process.kill()
                await process.wait()
                raise Exception(f"pidrei-server {args} hung on exit")
            assert result == 0
            assert expect in b"".join(out).decode("utf-8")

    @pytest.mark.tonio
    async def test_list_command_prints_response(self, tmp_dir):
        process = await _start_server(tmp_dir)
        try:
            client = await tonio.open_process(
                [_server_script(), "list"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, ENV_SERVER_DIR: str(tmp_dir), "PIDREI_OFFLINE": "1"},
            )
            out: list[bytes] = []
            while True:
                chunk = await client.stdout.receive_some()
                if not chunk:
                    break
                out.append(chunk)
            assert await client.wait() == 0
            printed = json.loads(b"".join(out).decode("utf-8"))
            assert printed == {"type": "list_result", "ok": True, "instances": []}
        finally:
            await _stop_server(process)
