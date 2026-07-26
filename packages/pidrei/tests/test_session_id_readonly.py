"""Mirrors pi coding-agent test/session-id-readonly.test.ts."""

import json
import os
from datetime import UTC, datetime

from .cli_spawn_helpers import run_cli


def _has_session_with_id(root: str, session_id: str) -> bool:
    if not os.path.exists(root):
        return False
    for entry in os.scandir(root):
        if entry.is_dir() and _has_session_with_id(entry.path, session_id):
            return True
        if not entry.is_file() or not entry.name.endswith(".jsonl"):
            continue
        try:
            with open(entry.path, encoding="utf-8") as handle:
                first_line = handle.readline()
            header = json.loads(first_line)
            if header.get("type") == "session" and header.get("id") == session_id:
                return True
        except Exception:  # noqa: S112
            # Ignore malformed session files.
            continue
    return False


def _write_session(session_dir: str, cwd: str, session_id: str) -> None:
    header = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "cwd": cwd,
    }
    with open(os.path.join(session_dir, f"{session_id}.jsonl"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")


class _Dirs:
    def __init__(self, tmp_dir):
        # realpath: session cwd filtering compares paths textually, so the
        # fixture must use physical paths (pi has the same macOS note).
        temp_root = os.path.realpath(str(tmp_dir))
        self.agent_dir = os.path.join(temp_root, "agent")
        self.project_dir = os.path.join(temp_root, "project")
        self.session_dir = os.path.join(temp_root, "sessions")
        os.makedirs(self.agent_dir, exist_ok=True)
        os.makedirs(self.project_dir, exist_ok=True)


class TestSessionIdReadOnlyCommands:
    def test_does_not_reserve_a_session_for_help(self, tmp_dir):
        dirs = _Dirs(tmp_dir)
        result = run_cli(["--session-id", "read-only-help", "--help"], cwd=dirs.project_dir, agent_dir=dirs.agent_dir)

        assert result.code == 0
        assert not _has_session_with_id(os.path.join(dirs.agent_dir, "sessions"), "read-only-help")

    def test_allows_no_session_with_session_id(self, tmp_dir):
        dirs = _Dirs(tmp_dir)
        result = run_cli(
            ["--no-session", "--session-id", "ephemeral-id", "--help"],
            cwd=dirs.project_dir,
            agent_dir=dirs.agent_dir,
        )

        assert result.code == 0
        assert not _has_session_with_id(os.path.join(dirs.agent_dir, "sessions"), "ephemeral-id")

    def test_does_not_reserve_a_session_for_list_models(self, tmp_dir):
        dirs = _Dirs(tmp_dir)
        result = run_cli(
            ["--session-id", "read-only-models", "--list-models"],
            cwd=dirs.project_dir,
            agent_dir=dirs.agent_dir,
        )

        assert result.code == 0
        assert not _has_session_with_id(os.path.join(dirs.agent_dir, "sessions"), "read-only-models")

    def test_warns_when_a_missing_session_id_creates_a_new_session(self, tmp_dir):
        dirs = _Dirs(tmp_dir)
        result = run_cli(
            [
                "--session-dir",
                dirs.session_dir,
                "--session-id",
                "missing-session-id",
                "--model",
                "missing-model",
                "-p",
                "hi",
            ],
            cwd=dirs.project_dir,
            agent_dir=dirs.agent_dir,
        )

        assert result.code == 1
        assert (
            "Warning: No project session found with id 'missing-session-id'; "
            "creating a new session with that id." in result.stderr
        )

    def test_does_not_warn_when_session_id_opens_an_existing_session(self, tmp_dir):
        dirs = _Dirs(tmp_dir)
        os.makedirs(dirs.session_dir, exist_ok=True)
        _write_session(dirs.session_dir, dirs.project_dir, "existing-session-id")
        result = run_cli(
            [
                "--session-dir",
                dirs.session_dir,
                "--session-id",
                "existing-session-id",
                "--model",
                "missing-model",
                "-p",
                "hi",
            ],
            cwd=dirs.project_dir,
            agent_dir=dirs.agent_dir,
        )

        assert result.code == 1
        assert "No project session found with id 'existing-session-id'" not in result.stderr

    def test_rejects_an_existing_fork_target_session_id(self, tmp_dir):
        dirs = _Dirs(tmp_dir)
        os.makedirs(dirs.session_dir, exist_ok=True)
        _write_session(dirs.session_dir, dirs.project_dir, "source-id")
        _write_session(dirs.session_dir, dirs.project_dir, "existing-id")
        result = run_cli(
            ["--session-dir", dirs.session_dir, "--fork", "source-id", "--session-id", "existing-id", "-p", "hi"],
            cwd=dirs.project_dir,
            agent_dir=dirs.agent_dir,
        )

        assert result.code == 1
        assert "Session already exists with id 'existing-id'" in result.stderr


class TestSessionIdValidation:
    def test_rejects_ids_invalid_under_session_manager_rules_without_stack_traces(self, tmp_dir):
        dirs = _Dirs(tmp_dir)
        for session_id in ["-bad", "bad id"]:
            result = run_cli(["--session-id", session_id, "-p", "hi"], cwd=dirs.project_dir, agent_dir=dirs.agent_dir)

            assert result.code == 1
            assert "Session id must be non-empty" in result.stderr
            assert "Traceback" not in result.stderr
