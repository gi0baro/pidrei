"""Mirror of pi's suite/regressions/8261-subagent-project-trust.test.ts.

pi mocks the `pi-coding-agent` module so the example never sees a real user
agent dir; here `agentScope: "project"` already skips user discovery, so the
extension is loaded as-is.
"""

import os

import pytest

from pidrei.core.agent_session import ExtensionBindings
from pidrei.examples.extensions.subagent import extension as subagent_extension
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call

from .harness import create_harness, get_message_text


class _UIContext:
    """Stands in for pi's `vi.fn()` confirm spy."""

    def __init__(self, result: bool) -> None:
        self.result = result
        self.calls = 0

    async def confirm(self, _title, _message=None):
        self.calls += 1
        return self.result


async def _run_project_agent(*, trusted: bool, confirm_result: bool = False) -> tuple[int, str]:
    harness = await create_harness(extension_factories=[subagent_extension])
    ui = _UIContext(confirm_result)
    try:
        agents_dir = os.path.join(harness.temp_dir, ".pidrei", "agents")
        os.makedirs(agents_dir, exist_ok=True)
        with open(os.path.join(agents_dir, "project-agent.md"), "w", encoding="utf-8") as handle:
            handle.write(
                "---\nname: project-agent\ndescription: Project test agent\n---\n\nHandle the delegated task.\n"
            )
        harness.settings_manager.set_project_trusted(trusted)

        await harness.session.bind_extensions(ExtensionBindings(ui_context=ui, mode="tui"))

        harness.set_responses(
            [
                faux_assistant_message(
                    faux_tool_call(
                        "subagent",
                        {
                            "agent": "project-agent",
                            "task": "Test project trust",
                            "agentScope": "project",
                            "cwd": os.path.join(harness.temp_dir, "missing-cwd"),
                        },
                    ),
                    stop_reason="toolUse",
                ),
                faux_assistant_message("done"),
            ]
        )

        await harness.session.prompt("Delegate this task")

        tool_result = next(
            (m for m in harness.session.messages if getattr(m, "role", None) == "toolResult"),
            None,
        )
        return ui.calls, get_message_text(tool_result)
    finally:
        harness.cleanup()


@pytest.mark.tonio
async def test_skips_per_call_confirmation_for_trusted_projects():
    confirm_calls, tool_result = await _run_project_agent(trusted=True)

    assert confirm_calls == 0
    assert "Canceled:" not in tool_result


@pytest.mark.tonio
async def test_keeps_confirmation_for_untrusted_interactive_projects():
    confirm_calls, tool_result = await _run_project_agent(trusted=False, confirm_result=False)

    assert confirm_calls == 1
    assert "Canceled: project-local agents not approved." in tool_result
