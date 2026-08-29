"""Mirrors pi coding-agent test/rpc-client-clone.test.ts and
test/rpc-client-process-exit.test.ts, plus one end-to-end smoke of the
RpcClient against the real spawned CLI (pi's rpc.test.ts covers that path
behind an API key; the auth store is stubbed with a fake key here and only
network-free commands are exercised).

pi's exit test spawns a Node child script via cliPath; the mirror spawns a
Python child script the same way.
"""

import json
import os

import pytest

from pidrei.config import ENV_AGENT_DIR
from pidrei.modes.rpc.rpc_client import RpcClient, RpcClientOptions


class TestRpcClientClone:
    @pytest.mark.tonio
    async def test_sends_the_clone_rpc_command(self):
        client = RpcClient()
        sent = []

        async def fake_send(command):
            sent.append(command)
            return {"type": "response", "command": "clone", "success": True, "data": {"cancelled": False}}

        client._send = fake_send

        result = await client.clone()

        assert sent == [{"type": "clone"}]
        assert result == {"cancelled": False}


class TestRpcClientChildProcessFailures:
    @pytest.mark.tonio
    async def test_rejects_an_in_flight_request_when_the_child_process_exits(self, tmp_path):
        child_path = os.path.join(str(tmp_path), "child.py")
        with open(child_path, "w", encoding="utf-8") as handle:
            handle.write("import sys\nsys.stdin.readline()\nsys.exit(43)\n")

        client = RpcClient(RpcClientOptions(cli_path=child_path))
        await client.start()
        try:
            with pytest.raises(Exception, match=r"Agent process exited \(code=43 signal=null\)"):
                await client.get_commands()
        finally:
            await client.stop()


class TestRpcClientEndToEnd:
    @pytest.mark.tonio
    async def test_drives_the_spawned_cli_in_rpc_mode(self, tmp_path):
        temp_root = os.path.realpath(str(tmp_path))
        agent_dir = os.path.join(temp_root, "agent")
        project_dir = os.path.join(temp_root, "project")
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(project_dir, exist_ok=True)
        with open(os.path.join(agent_dir, "auth.json"), "w", encoding="utf-8") as handle:
            json.dump({"anthropic": {"type": "api_key", "key": "test-key"}}, handle)

        client = RpcClient(
            RpcClientOptions(
                cwd=project_dir,
                env={ENV_AGENT_DIR: agent_dir, "PIDREI_OFFLINE": "1"},
                provider="anthropic",
                model="claude-sonnet-4-5",
            )
        )
        await client.start()
        try:
            state = await client.get_state()
            assert state["model"]["provider"] == "anthropic"
            assert state["model"]["id"] == "claude-sonnet-4-5"
            assert state["isStreaming"] is False
            assert state["messageCount"] == 0

            result = await client.bash("echo hello")
            assert result["output"].strip() == "hello"
            assert result["exitCode"] == 0
            assert result["cancelled"] is False

            levels = await client.get_available_thinking_levels()
            assert len(levels) > 0
            await client.set_thinking_level("high")
            state = await client.get_state()
            assert state["thinkingLevel"] == "high"
        finally:
            await client.stop()
