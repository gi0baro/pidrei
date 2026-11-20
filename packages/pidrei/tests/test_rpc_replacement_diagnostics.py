"""Mirror of pi coding-agent test/rpc-replacement-diagnostics.test.ts.

pi spawns `node src/cli.ts --mode rpc`; the mirror spawns `python -m pidrei --mode rpc` through
`RpcCli`, which waits for each command's response before the next step, as pi's test does.
"""

import json
import os
import re

from .cli_spawn_helpers import RpcCli


def _write_settings(agent_dir: str, enabled_models: list[str]) -> None:
    with open(os.path.join(agent_dir, "settings.json"), "w", encoding="utf-8") as handle:
        json.dump({"defaultProvider": "fake", "defaultModel": "fake-model", "enabledModels": enabled_models}, handle)


class TestRpcDiagnosticsAfterSessionReplacement:
    # The runtime factory runs again on every new_session, but only the startup diagnostics were reported, so a
    # problem that first appeared on replacement went nowhere. Found while fixing #11264.
    def test_writes_a_diagnostic_that_first_appears_on_new_session_to_stderr_once(self, tmp_path):
        agent_dir = os.path.join(str(tmp_path), "agent")
        project_dir = os.path.join(str(tmp_path), "project")
        os.makedirs(agent_dir, exist_ok=True)
        os.makedirs(project_dir, exist_ok=True)
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
        _write_settings(agent_dir, ["fake/fake-model"])

        with RpcCli(["--no-session"], cwd=project_dir, agent_dir=agent_dir) as child:
            child.request({"id": "1", "type": "new_session"})
            _write_settings(agent_dir, ["fake/fake-model", "github-copilot/missing-model"])
            child.request({"id": "2", "type": "new_session"})
            child.request({"id": "3", "type": "new_session"})
            result = child.close()

        assert len(re.findall(r'Warning: No models match pattern "github-copilot/missing-model"', result.stderr)) == 1
        assert "No models match pattern" not in result.stdout
