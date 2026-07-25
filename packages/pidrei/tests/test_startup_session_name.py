"""Mirrors pi coding-agent test/startup-session-name.test.ts."""

import json
import os
from datetime import UTC, datetime

from .cli_spawn_helpers import run_cli
from .coding_session_helpers import now_ms


def _create_session_file(project_dir: str, session_file: str) -> None:
    timestamp = datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    header = {"type": "session", "version": 3, "id": "existing-session", "timestamp": timestamp, "cwd": project_dir}
    message_entry = {
        "type": "message",
        "id": "assistant-1",
        "parentId": None,
        "timestamp": timestamp,
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "timestamp": now_ms(),
        },
    }
    with open(session_file, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n" + json.dumps(message_entry) + "\n")


def _read_session_info_names(session_file: str) -> list[str]:
    with open(session_file, encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle.read().strip().split("\n")]
    return [entry.get("name", "") for entry in entries if entry.get("type") == "session_info"]


def _setup(tmp_dir) -> dict[str, str]:
    temp_root = os.path.realpath(str(tmp_dir))
    dirs = {
        "agent_dir": os.path.join(temp_root, "agent"),
        "project_dir": os.path.join(temp_root, "project"),
        "session_file": os.path.join(temp_root, "session.jsonl"),
    }
    os.makedirs(dirs["agent_dir"], exist_ok=True)
    os.makedirs(dirs["project_dir"], exist_ok=True)
    _create_session_file(dirs["project_dir"], dirs["session_file"])
    return dirs


class TestStartupSessionName:
    def test_sets_name_on_the_selected_session_before_runtime_model_validation(self, tmp_dir):
        dirs = _setup(tmp_dir)
        result = run_cli(
            [
                "--session",
                dirs["session_file"],
                "--name",
                "  CLI Named Session  ",
                "--model",
                "missing-model",
                "-p",
                "hi",
            ],
            cwd=dirs["project_dir"],
            agent_dir=dirs["agent_dir"],
        )

        assert result.code == 1
        assert _read_session_info_names(dirs["session_file"]) == ["CLI Named Session"]

    def test_rejects_empty_name_values_without_appending_session_metadata(self, tmp_dir):
        dirs = _setup(tmp_dir)
        result = run_cli(
            ["--session", dirs["session_file"], "--name", "   ", "--model", "missing-model", "-p", "hi"],
            cwd=dirs["project_dir"],
            agent_dir=dirs["agent_dir"],
        )

        assert result.code == 1
        assert "--name requires a non-empty value" in result.stderr
        assert _read_session_info_names(dirs["session_file"]) == []
