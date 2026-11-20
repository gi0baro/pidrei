"""Mirror of pi coding-agent test/model-fallback-diagnostic.test.ts.

pi spawns `node src/cli.ts --mode rpc`; the mirror spawns `python -m pidrei --mode rpc` through
`RpcCli`, which waits for the command's response before ending stdin, as pi's test does. The session
file is written with `SessionManager`, as in pi; the child is driven on the blocking pool.
"""

import json
import os

import pytest
import tonio.colored as tonio

from pidrei.core.session_manager import SessionManager
from pidrei_ai.types import AssistantMessage, TextContent, Usage, UsageCost, UserMessage
from pidrei_utils import clock

from .cli_spawn_helpers import CliResult, RpcCli


async def _create_session_with_missing_model(project_dir: str, session_dir: str) -> str:
    """A saved session whose last model, gone/missing-model, no longer exists."""
    session_manager = await SessionManager(project_dir, session_dir)
    await session_manager.append_message(UserMessage(content="hello", timestamp=clock.now_ms()))
    await session_manager.append_message(
        AssistantMessage(
            content=[TextContent(text="hi")],
            api="openai-completions",
            provider="gone",
            model="missing-model",
            usage=Usage(input=1, output=1, cache_read=0, cache_write=0, total_tokens=2, cost=UsageCost()),
            stop_reason="stop",
            timestamp=clock.now_ms(),
        )
    )
    session_file = session_manager.get_session_file()
    assert session_file is not None, "session was not persisted"
    return session_file


def _write_agent_dir_blocking(agent_dir: str, project_dir: str) -> None:
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
    with open(os.path.join(agent_dir, "settings.json"), "w", encoding="utf-8") as handle:
        json.dump({"defaultProvider": "fake", "defaultModel": "fake-model"}, handle)


def _switch_session_over_rpc_blocking(agent_dir: str, project_dir: str, session_file: str) -> CliResult:
    with RpcCli(["--no-session"], cwd=project_dir, agent_dir=agent_dir) as child:
        child.request({"id": "1", "type": "switch_session", "sessionPath": session_file})
        return child.close()


class TestModelFallbackMessage:
    # Found while fixing #11264: "Could not restore model" was shown only at interactive startup. A /resume into
    # a session whose model is gone fell back to another model silently, and other modes never showed it.
    @pytest.mark.tonio
    async def test_is_written_to_stderr_in_rpc_mode_when_switch_session_falls_back_to_another_model(self, tmp_path):
        agent_dir = os.path.join(str(tmp_path), "agent")
        project_dir = os.path.join(str(tmp_path), "project")
        session_dir = os.path.join(str(tmp_path), "sessions")
        await tonio.spawn_blocking(_write_agent_dir_blocking, agent_dir, project_dir)
        session_file = await _create_session_with_missing_model(project_dir, session_dir)

        result = await tonio.spawn_blocking(_switch_session_over_rpc_blocking, agent_dir, project_dir, session_file)

        assert "Warning: Could not restore model gone/missing-model. Using fake/fake-model" in result.stderr
        assert "Could not restore model" not in result.stdout
