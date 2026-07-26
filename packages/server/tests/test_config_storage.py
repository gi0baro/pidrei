"""Config paths, storage round-trips, protocol framing."""

import json
import os

from pidrei_server.config import (
    ENV_SERVER_DIR,
    get_instances_path,
    get_machine_path,
    get_server_dir,
    get_socket_path,
)
from pidrei_server.ipc.protocol import encode_message, parse_request_line, parse_response_line
from pidrei_server.storage import (
    delete_machine,
    get_instance,
    load_instances,
    load_machine,
    remove_instance,
    save_instances,
    save_machine,
    upsert_instance,
)

from .server_helpers import env_var


class TestConfig:
    def test_server_dir_env_override(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            assert get_server_dir() == str(tmp_dir)
            assert get_socket_path() == os.path.join(str(tmp_dir), "server.sock")
            assert get_machine_path() == os.path.join(str(tmp_dir), "machine.json")
            assert get_instances_path() == os.path.join(str(tmp_dir), "instances.json")

    def test_server_dir_under_config_dir(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, None), env_var("PIDREI_CONFIG_DIR", str(tmp_dir)):
            assert get_server_dir() == os.path.join(str(tmp_dir), "server")


class TestStorage:
    def test_machine_round_trip(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            assert load_machine() is None
            machine = {"id": "m-1", "createdAt": "2026-07-25T00:00:00.000Z"}
            save_machine(machine)
            assert load_machine() == machine
            delete_machine()
            assert load_machine() is None
            delete_machine()  # no-op on missing file

    def test_instances_upsert_get_remove(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            assert load_instances() == []
            first = {"id": "i-1", "status": "starting", "cwd": "/tmp", "createdAt": "t"}
            second = {"id": "i-2", "status": "online", "cwd": "/tmp", "createdAt": "t"}
            upsert_instance(first)
            upsert_instance(second)
            assert [instance["id"] for instance in load_instances()] == ["i-1", "i-2"]

            upsert_instance({**first, "status": "online"})
            assert get_instance("i-1")["status"] == "online"
            assert get_instance("missing") is None

            remove_instance("i-1")
            assert [instance["id"] for instance in load_instances()] == ["i-2"]

    def test_instances_persisted_as_pretty_json(self, tmp_dir):
        with env_var(ENV_SERVER_DIR, str(tmp_dir)):
            save_instances([{"id": "i-1"}])
            with open(get_instances_path(), encoding="utf-8") as handle:
                content = handle.read()
            assert content == json.dumps([{"id": "i-1"}], indent=2)


class TestProtocol:
    def test_encode_parse_round_trip(self):
        message = {"type": "spawn", "cwd": "/tmp/préjet", "label": "café"}
        encoded = encode_message(message)
        assert encoded.endswith("\n")
        assert "café" in encoded  # unicode unescaped like JSON.stringify
        assert parse_request_line(encoded.strip()) == message
        assert parse_response_line(encoded.strip()) == message
