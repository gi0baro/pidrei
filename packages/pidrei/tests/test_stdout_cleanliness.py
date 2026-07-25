"""Mirrors pi coding-agent test/stdout-cleanliness.test.ts.

pi's suite also asserts that trusted project npm-package install chatter
routes to stderr; project package installs (npm sources) land in Phase 5,
so those assertions are dropped and the stdout/stderr routing of
--version/--help across output modes is mirrored as-is.
"""

import os
import re

from .cli_spawn_helpers import run_cli


def _make_dirs(tmp_dir) -> tuple[str, str]:
    agent_dir = os.path.join(str(tmp_dir), "agent")
    project_dir = os.path.join(str(tmp_dir), "project")
    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    return agent_dir, project_dir


class TestStdoutCleanlinessInNonInteractiveModes:
    def test_prints_version_to_stdout_when_stdout_is_redirected(self, tmp_dir):
        agent_dir, project_dir = _make_dirs(tmp_dir)
        result = run_cli(["--version"], cwd=project_dir, agent_dir=agent_dir)

        assert result.code == 0
        assert re.match(r"^\d+\.\d+\.\d+", result.stdout.strip())
        assert result.stderr == ""

    def test_prints_plain_help_to_stdout_when_stdout_is_redirected(self, tmp_dir):
        agent_dir, project_dir = _make_dirs(tmp_dir)
        result = run_cli(["--help"], cwd=project_dir, agent_dir=agent_dir)

        assert result.code == 0
        assert "Usage:" in result.stdout
        assert "Usage:" not in result.stderr

    def test_keeps_stdout_empty_for_mode_json_help(self, tmp_dir):
        agent_dir, project_dir = _make_dirs(tmp_dir)
        result = run_cli(["--mode", "json", "--help", "--approve"], cwd=project_dir, agent_dir=agent_dir)

        assert result.code == 0
        assert result.stdout == ""
        assert "Usage:" in result.stderr

    def test_keeps_stdout_empty_for_p_help(self, tmp_dir):
        agent_dir, project_dir = _make_dirs(tmp_dir)
        result = run_cli(["-p", "--help", "--approve"], cwd=project_dir, agent_dir=agent_dir)

        assert result.code == 0
        assert result.stdout == ""
        assert "Usage:" in result.stderr

    def test_untrusted_project_help_still_works(self, tmp_dir):
        agent_dir, project_dir = _make_dirs(tmp_dir)
        result = run_cli(["-p", "--help"], cwd=project_dir, agent_dir=agent_dir)

        assert result.code == 0
        assert result.stdout == ""
        assert "Usage:" in result.stderr
