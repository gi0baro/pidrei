"""Mirror of pi coding-agent test/project-trust-saved-decision.test.ts.

pi's probe extension calls `process.exit(0)` from `session_start`; here its
provider answers the print-mode prompt, so the run ends normally.
"""

import json
import os

import pytest
import tonio.colored as tonio

from pidrei.cli.project_trust import CreateProjectTrustContextOptions, create_project_trust_context
from pidrei.core.project_trust import ResolveProjectTrustedOptions, resolve_project_trusted
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.trust_manager import ProjectTrustStore

from .cli_spawn_helpers import run_cli


_EXTENSION = """
from tonio.colored import fs

from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.types import DoneEvent


def stream_simple(model, _context, _options=None):
    stream = AssistantMessageEventStream()
    stream.push(DoneEvent(reason="stop", message=faux_assistant_message("ok")))
    return stream


async def extension(pi):
    pi.register_provider(
        "trust-probe",
        {
            "baseUrl": "https://trust-probe.invalid/v1",
            "apiKey": "trust-probe-key",
            "api": "openai-completions",
            "streamSimple": stream_simple,
            "models": [
                {
                    "id": "trust-probe-model",
                    "name": "trust-probe-model",
                    "reasoning": False,
                    "input": ["text"],
                    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                    "contextWindow": 10000,
                    "maxTokens": 1000,
                }
            ],
        },
    )

    async def on_session_start(_event, ctx):
        await fs.Path(PROBE_FILE).write_text(json.dumps({"trusted": ctx.is_project_trusted()}))

    pi.on("session_start", on_session_start)
"""


def _setup(tmp_path) -> dict[str, str]:
    temp_root = os.path.realpath(str(tmp_path))
    dirs = {
        "agent_dir": os.path.join(temp_root, "agent"),
        "project_dir": os.path.join(temp_root, "project"),
        "probe_file": os.path.join(temp_root, "probe.json"),
        "extension_file": os.path.join(temp_root, "probe_extension.py"),
    }
    os.makedirs(dirs["agent_dir"])
    os.makedirs(dirs["project_dir"])
    # AGENTS.md alone is not a trust-requiring project resource.
    with open(os.path.join(dirs["project_dir"], "AGENTS.md"), "w", encoding="utf-8") as handle:
        handle.write("# untrusted\n")
    with open(dirs["extension_file"], "w", encoding="utf-8") as handle:
        handle.write(f"import json\n\nPROBE_FILE = {dirs['probe_file']!r}\n{_EXTENSION}")
    return dirs


def _probe_project_trust(dirs: dict[str, str]) -> bool:
    result = run_cli(
        [
            "--no-session",
            "-e",
            dirs["extension_file"],
            "--provider",
            "trust-probe",
            "--model",
            "trust-probe-model",
            "-p",
            "hi",
        ],
        cwd=dirs["project_dir"],
        agent_dir=dirs["agent_dir"],
        timeout=30,
    )
    assert result.code == 0, result.stderr
    with open(dirs["probe_file"], encoding="utf-8") as handle:
        return json.load(handle)["trusted"]


# Regression tests for https://github.com/earendil-works/pi/issues/10947


@pytest.mark.tonio
async def test_reports_a_saved_do_not_trust_decision_to_extensions(tmp_path):
    dirs = _setup(tmp_path)
    await ProjectTrustStore(dirs["agent_dir"]).set(dirs["project_dir"], False)

    assert await tonio.spawn_blocking(_probe_project_trust, dirs) is False


def test_auto_trusts_when_no_decision_is_saved(tmp_path):
    dirs = _setup(tmp_path)

    assert _probe_project_trust(dirs) is True


@pytest.mark.tonio
async def test_resolve_project_trusted_honors_the_saved_decision_before_auto_trust(tmp_path):
    dirs = _setup(tmp_path)
    trust_store = ProjectTrustStore(dirs["agent_dir"])

    async def resolve() -> bool:
        return await resolve_project_trusted(
            ResolveProjectTrustedOptions(
                cwd=dirs["project_dir"],
                trust_store=trust_store,
                project_trust_context=create_project_trust_context(
                    CreateProjectTrustContextOptions(
                        cwd=dirs["project_dir"],
                        mode="print",
                        settings_manager=SettingsManager.in_memory(),
                        has_ui=False,
                    )
                ),
            )
        )

    assert await resolve() is True
    await trust_store.set(dirs["project_dir"], False)
    assert await resolve() is False
    await trust_store.set(dirs["project_dir"], True)
    assert await resolve() is True
