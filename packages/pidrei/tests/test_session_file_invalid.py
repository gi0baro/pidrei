"""Mirrors pi coding-agent test/session-file-invalid.test.ts."""

import os

from .cli_spawn_helpers import run_cli


class TestSessionInvalidFileHandling:
    def test_prints_a_friendly_error_and_preserves_non_session_file_content(self, tmp_path):
        temp_root = os.path.realpath(str(tmp_path))
        agent_dir = os.path.join(temp_root, "agent")
        project_dir = os.path.join(temp_root, "project")
        session_file = os.path.join(temp_root, "not-a-session.log")
        original_content = '{"type":"event","data":"not a session"}\n'
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(project_dir, exist_ok=True)
        with open(session_file, "w", encoding="utf-8") as handle:
            handle.write(original_content)

        result = run_cli(["--session", session_file, "-p", "hi"], cwd=project_dir, agent_dir=agent_dir)

        assert result.code == 1
        assert f"Error: Session file is not a valid pidrei session: {session_file}" in result.stderr
        assert "Traceback" not in result.stderr
        with open(session_file, encoding="utf-8") as handle:
            assert handle.read() == original_content
