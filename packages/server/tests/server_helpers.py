"""Shared helpers for pidrei_server tests."""

import contextlib
import os


@contextlib.contextmanager
def env_var(name, value):
    original = os.environ.get(name)
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = original


# A stand-in RPC child speaking the JSONL protocol: responds to get_state,
# emits events/ui requests on demand, and exits on request or SIGTERM.
FAKE_RPC_CHILD = """\
import json
import signal
import sys

signal.signal(signal.SIGTERM, lambda *args: sys.exit(0))


def reply(message):
    print(json.dumps(message), flush=True)


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    command = json.loads(line)
    command_type = command.get("type")
    if command_type == "get_state":
        reply(
            {
                "id": command.get("id"),
                "type": "response",
                "command": "get_state",
                "success": True,
                "data": {"sessionId": "sess-1", "sessionFile": "/tmp/sess-1.jsonl"},
            }
        )
    elif command_type == "emit_event":
        reply({"type": "custom_event", "value": command.get("value")})
        reply({"id": command.get("id"), "type": "response", "command": "emit_event", "success": True})
    elif command_type == "emit_ui":
        reply({"type": "extension_ui_request", "id": "ui-1", "method": "confirm", "title": "sure?"})
        reply({"id": command.get("id"), "type": "response", "command": "emit_ui", "success": True})
    elif command_type == "echo_ui_response":
        # The next line on stdin is an extension_ui_response; echo it back as an event.
        reply({"id": command.get("id"), "type": "response", "command": "echo_ui_response", "success": True})
    elif command_type == "exit":
        sys.exit(7)
    else:
        reply({"id": command.get("id"), "type": "response", "command": command_type, "success": True, "data": {}})
"""


def write_fake_rpc_child(tmp_dir, body=FAKE_RPC_CHILD, name="fake_rpc_child.py"):
    path = os.path.join(str(tmp_dir), name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(body)
    return path
