"""RpcProcessInstance against a fake JSONL child process."""

import pytest
import tonio.colored as tonio

from pidrei_server.rpc_process import RpcProcessOptions, create_rpc_process_instance

from .server_helpers import write_fake_rpc_child


async def _wait_for(condition, timeout_s=5.0):
    for _ in range(int(timeout_s / 0.02)):
        if condition():
            return True
        await tonio.time.sleep(0.02)
    return condition()


class TestRpcProcess:
    @pytest.mark.tonio
    async def test_send_receives_correlated_response(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        instance = await create_rpc_process_instance(RpcProcessOptions(cwd=str(tmp_dir), cli_path=child))
        try:
            response = await instance.send({"type": "get_state"})
            assert response["success"] is True
            assert response["data"]["sessionId"] == "sess-1"
            assert response["id"].startswith("server_1_")

            # An explicit id is preserved.
            response = await instance.send({"type": "get_state", "id": "my-id"})
            assert response["id"] == "my-id"
        finally:
            await instance.dispose()

    @pytest.mark.tonio
    async def test_events_and_ui_requests_dispatch(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        instance = await create_rpc_process_instance(RpcProcessOptions(cwd=str(tmp_dir), cli_path=child))
        events = []
        ui_requests = []
        try:
            unsubscribe = instance.on_event(events.append)
            instance.set_ui_request_handler(ui_requests.append)

            await instance.send({"type": "emit_event", "value": 42})
            await instance.send({"type": "emit_ui"})
            assert events == [{"type": "custom_event", "value": 42}]
            assert ui_requests == [
                {"type": "extension_ui_request", "id": "ui-1", "method": "confirm", "title": "sure?"}
            ]

            unsubscribe()
            await instance.send({"type": "emit_event", "value": 43})
            assert len(events) == 1
        finally:
            await instance.dispose()

    @pytest.mark.tonio
    async def test_exit_rejects_pending_and_notifies(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        instance = await create_rpc_process_instance(RpcProcessOptions(cwd=str(tmp_dir), cli_path=child))
        exits = []
        instance.on_exit(exits.append)

        with pytest.raises(Exception, match=r"RPC process exited \(code=7 signal=null\)"):
            await instance.send({"type": "exit"})

        assert await _wait_for(lambda: len(exits) == 1)
        assert "RPC process exited" in str(exits[0])

        with pytest.raises(Exception, match="RPC process is not running"):
            await instance.send({"type": "get_state"})

        await instance.dispose()  # no-op after exit

    @pytest.mark.tonio
    async def test_dispose_terminates_child(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        instance = await create_rpc_process_instance(RpcProcessOptions(cwd=str(tmp_dir), cli_path=child))
        await instance.send({"type": "get_state"})
        await instance.dispose()
        assert instance.process.poll() is not None
