"""ServerSupervisor lifecycle against the fake RPC child."""

import pytest
import tonio.colored as tonio

from pidrei_server.config import ENV_SERVER_DIR
from pidrei_server.storage import load_instances, save_instances
from pidrei_server.supervisor import ServerSupervisor

from .server_helpers import env_var, write_fake_rpc_child


async def _wait_for(condition, timeout_s=5.0):
    for _ in range(int(timeout_s / 0.02)):
        if condition():
            return True
        await tonio.time.sleep(0.02)
    return condition()


class TestSupervisorLifecycle:
    @pytest.mark.tonio
    async def test_spawn_syncs_session_and_goes_online(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            supervisor = ServerSupervisor()
            record = await supervisor.spawn_instance({"cwd": str(tmp_dir), "label": "one", "cli_path": child})
            try:
                assert record["status"] == "online"
                assert record["label"] == "one"
                assert record["sessionId"] == "sess-1"
                assert record["sessionFile"] == "/tmp/sess-1.jsonl"
                assert "radiusPiId" not in record  # pi's radius field; integration dropped

                stored = load_instances()
                assert [instance["id"] for instance in stored] == [record["id"]]
                assert stored[0]["status"] == "online"

                assert supervisor.get_instance(record["id"])["status"] == "online"
                assert supervisor.get_live_instance(record["id"])["id"] == record["id"]
                assert [instance["id"] for instance in supervisor.list_live_instances()] == [record["id"]]
            finally:
                await supervisor.shutdown()

    @pytest.mark.tonio
    async def test_stop_instance_removes_record(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            supervisor = ServerSupervisor()
            record = await supervisor.spawn_instance({"cwd": str(tmp_dir), "cli_path": child})
            stopped = await supervisor.stop_instance(record["id"])
            assert stopped["status"] == "stopped"
            assert load_instances() == []
            assert supervisor.get_live_instance(record["id"]) is None
            assert await supervisor.stop_instance("missing") is None

    @pytest.mark.tonio
    async def test_handle_rpc_and_unknown_instance(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            supervisor = ServerSupervisor()
            record = await supervisor.spawn_instance({"cwd": str(tmp_dir), "cli_path": child})
            try:
                response = await supervisor.handle_rpc(record["id"], {"type": "compact"})
                assert response["success"] is True
                assert await supervisor.handle_rpc("missing", {"type": "compact"}) is None
            finally:
                await supervisor.shutdown()

    @pytest.mark.tonio
    async def test_unexpected_exit_marks_error(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            supervisor = ServerSupervisor()
            record = await supervisor.spawn_instance({"cwd": str(tmp_dir), "cli_path": child})

            with pytest.raises(Exception, match="RPC process exited"):
                await supervisor.handle_rpc(record["id"], {"type": "exit"})

            assert await _wait_for(lambda: supervisor.get_live_instance(record["id"]) is None)
            stored = supervisor.get_instance(record["id"])
            assert stored["status"] == "error"

    @pytest.mark.tonio
    async def test_failed_spawn_marks_stopped_and_raises(self, tmp_dir):
        crashing = write_fake_rpc_child(tmp_dir, body="import sys\nsys.exit(3)\n", name="crashing_child.py")
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            supervisor = ServerSupervisor()
            with pytest.raises(Exception, match=r"RPC process exited \(code=3 signal=null\)"):
                await supervisor.spawn_instance({"cwd": str(tmp_dir), "cli_path": crashing})

            assert supervisor.list_live_instances() == []
            stored = load_instances()
            assert len(stored) == 1
            assert stored[0]["status"] == "stopped"

    @pytest.mark.tonio
    async def test_recover_after_restart(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            save_instances(
                [
                    {"id": "a", "status": "online", "cwd": "/tmp", "createdAt": "t"},
                    {"id": "b", "status": "starting", "cwd": "/tmp", "createdAt": "t"},
                    {"id": "c", "status": "error", "cwd": "/tmp", "createdAt": "t"},
                ]
            )
            supervisor = ServerSupervisor()
            await supervisor.recover_after_restart()
            statuses = {instance["id"]: instance["status"] for instance in load_instances()}
            assert statuses == {"a": "stopped", "b": "stopped", "c": "error"}


class TestSupervisorRpcStream:
    @pytest.mark.tonio
    async def test_open_rpc_stream_events_and_close(self, tmp_dir):
        child = write_fake_rpc_child(tmp_dir)
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            supervisor = ServerSupervisor()
            record = await supervisor.spawn_instance({"cwd": str(tmp_dir), "cli_path": child})
            events = []
            ui_requests = []
            try:
                handle = supervisor.open_rpc_stream(record["id"], events.append, ui_requests.append)
                assert handle is not None
                assert supervisor.open_rpc_stream("missing", events.append, ui_requests.append) is None

                response = await handle.handle_rpc({"type": "emit_event", "value": 5})
                assert response["success"] is True
                assert events == [{"type": "custom_event", "value": 5}]

                await handle.handle_rpc({"type": "emit_ui"})
                assert len(ui_requests) == 1
                assert ui_requests[0]["method"] == "confirm"

                handle.close()
                await handle2_noop_check(supervisor, record, events)
            finally:
                await supervisor.shutdown()


async def handle2_noop_check(supervisor, record, events):
    """After close, further child events no longer reach the subscriber."""
    live_count = len(events)
    response = await supervisor.handle_rpc(record["id"], {"type": "emit_event", "value": 6})
    assert response["success"] is True
    await tonio.time.sleep(0.1)
    assert len(events) == live_count
