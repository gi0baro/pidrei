"""Mirror of pi's regressions/6162-extension-active-tools-next-turn.test.ts.

pi's faux response callables take `(context)`; pidrei's take
`(context, options, state, model)`, so each one absorbs the extra arguments.
"""

import pytest

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent, TranscriptContext
from pidrei_ai.utils.transcript import get_current_system_prompt, get_current_tools

from .harness import create_harness


def get_provider_tool_names(context: TranscriptContext) -> list[str]:
    return sorted(tool.name for tool in get_current_tools(context.messages))


async def register_switch_tools(pi) -> None:
    """Register `switch_tools`, which swaps the active set to `after_switch` when executed."""

    async def switch(*_args):
        pi.set_active_tools(["after_switch"])
        return AgentToolResult(content=[TextContent(text="switched")], details={})

    async def after(*_args):
        return AgentToolResult(content=[TextContent(text="after")], details={})

    pi.register_tool(
        ToolDefinition(
            name="switch_tools",
            label="Switch Tools",
            description="Switch the active extension tool set",
            prompt_snippet="Switch to the next extension tool",
            parameters={"type": "object", "properties": {}},
            execute=switch,
        )
    )
    pi.register_tool(
        ToolDefinition(
            name="after_switch",
            label="After Switch",
            description="Tool that should be available after switching",
            prompt_snippet="Run after the active tool set changes",
            parameters={"type": "object", "properties": {}},
            execute=after,
        )
    )


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


# Regression #6162
@pytest.mark.tonio
async def test_applies_set_active_tools_before_the_next_provider_request_in_the_same_run(harnesses):
    harness = await create_harness(extension_factories=[register_switch_tools])
    harnesses.append(harness)
    harness.session.set_active_tools_by_name(["switch_tools"])

    provider_tool_names: list[list[str]] = []

    async def first(context, *_rest):
        provider_tool_names.append(get_provider_tool_names(context))
        return faux_assistant_message(faux_tool_call("switch_tools", {}), stop_reason="toolUse")

    async def second(context, *_rest):
        provider_tool_names.append(get_provider_tool_names(context))
        return faux_assistant_message("done")

    harness.set_responses([first, second])

    assert harness.session.get_active_tool_names() == ["switch_tools"]

    await harness.session.prompt("start")

    assert harness.session.get_active_tool_names() == ["after_switch"]
    assert provider_tool_names == [["switch_tools"], ["after_switch"]]


@pytest.mark.tonio
async def test_reports_the_refreshed_system_prompt_during_the_run(harnesses):
    harness = await create_harness(extension_factories=[register_switch_tools])
    harnesses.append(harness)
    harness.session.set_active_tools_by_name(["switch_tools"])

    provider_prompts: list[str] = []
    session_prompts: list[str] = []

    async def first(context, *_rest):
        provider_prompts.append(get_current_system_prompt(context.messages))
        session_prompts.append(harness.session.system_prompt)
        return faux_assistant_message(faux_tool_call("switch_tools", {}), stop_reason="toolUse")

    async def second(context, *_rest):
        provider_prompts.append(get_current_system_prompt(context.messages))
        session_prompts.append(harness.session.system_prompt)
        return faux_assistant_message("done")

    harness.set_responses([first, second])

    await harness.session.prompt("start")

    assert len(provider_prompts) == 2
    assert provider_prompts[0] != provider_prompts[1]
    assert session_prompts == provider_prompts


@pytest.mark.tonio
async def test_preserves_before_agent_start_system_prompt_overrides_when_tools_change_mid_run(harnesses):
    async def factory(pi) -> None:
        async def on_before_agent_start(event, _ctx):
            return {"systemPrompt": f"{event['systemPrompt']}\n\nkeep this run override"}

        pi.on("before_agent_start", on_before_agent_start)
        await register_switch_tools(pi)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.session.set_active_tools_by_name(["switch_tools"])

    provider_system_prompts: list[str] = []
    provider_tool_names: list[list[str]] = []

    def capture(context: TranscriptContext) -> None:
        provider_system_prompts.append(get_current_system_prompt(context.messages))
        provider_tool_names.append(get_provider_tool_names(context))

    async def first(context, *_rest):
        capture(context)
        return faux_assistant_message(faux_tool_call("switch_tools", {}), stop_reason="toolUse")

    async def second(context, *_rest):
        capture(context)
        return faux_assistant_message("done")

    harness.set_responses([first, second])

    await harness.session.prompt("start")

    assert provider_tool_names == [["switch_tools"], ["after_switch"]]
    assert len(provider_system_prompts) == 2
    assert "keep this run override" in provider_system_prompts[0]
    assert "keep this run override" in provider_system_prompts[1]
