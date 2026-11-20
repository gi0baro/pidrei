"""Mirror of pi coding-agent test/model-scope-session-replacement.test.ts.

pi spawns `node src/cli.ts --mode rpc`; the mirror spawns `python -m pidrei --mode rpc` through
`RpcCli`, which waits for the command's response before ending stdin, as pi's test does.
"""

import json
import os
import re

from .cli_spawn_helpers import RpcCli


def _make_dirs(tmp_path) -> tuple[str, str]:
    agent_dir = os.path.join(str(tmp_path), "agent")
    project_dir = os.path.join(str(tmp_path), "project")
    os.makedirs(agent_dir, exist_ok=True)
    os.makedirs(project_dir, exist_ok=True)
    return agent_dir, project_dir


def _run_new_session_over_rpc(agent_dir: str, project_dir: str) -> str:
    """Start RPC mode, replace the session once, and return everything the process wrote to stderr."""
    with RpcCli(["--no-session"], cwd=project_dir, agent_dir=agent_dir) as child:
        child.request({"id": "1", "type": "new_session"})
        return child.close().stderr


class TestModelScopeWarningsOnSessionReplacement:
    # #11264: the runtime factory runs again on /new, /resume and /fork. It printed the model scope warnings
    # to the terminal, which drew over the interactive TUI. They are now startup diagnostics, reported once.
    def test_reports_unmatched_enabled_models_patterns_once_at_startup_not_again_on_new_session(self, tmp_path):
        agent_dir, project_dir = _make_dirs(tmp_path)
        with open(os.path.join(agent_dir, "models.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "providers": {
                        "fake": {
                            "baseUrl": "http://127.0.0.1:9/v1",
                            "api": "openai-completions",
                            "apiKey": "test",
                            "models": [{"id": "fake-model"}],
                        }
                    }
                },
                handle,
            )
        with open(os.path.join(agent_dir, "settings.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "defaultProvider": "fake",
                    "defaultModel": "fake-model",
                    "enabledModels": ["fake/fake-model", "github-copilot/missing-model"],
                },
                handle,
            )

        stderr = _run_new_session_over_rpc(agent_dir, project_dir)

        assert len(re.findall(r'No models match pattern "github-copilot/missing-model"', stderr)) == 1
