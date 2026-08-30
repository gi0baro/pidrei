"""Mirror of pi's rpc-client-clear-queue.test.ts."""

import pytest

from pidrei.modes.rpc.rpc_client import RpcClient


@pytest.mark.tonio
async def test_sends_the_clear_queue_rpc_command():
    client = RpcClient()
    sent = []

    async def fake_send(command):
        sent.append(command)
        return {
            "type": "response",
            "command": "clear_queue",
            "success": True,
            "data": {"steering": ["Change direction"], "followUp": ["Summarize when finished"]},
        }

    client._send = fake_send

    result = await client.clear_queue()

    assert sent == [{"type": "clear_queue"}]
    assert result == {"steering": ["Change direction"], "followUp": ["Summarize when finished"]}
