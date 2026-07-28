"""Mirror of pi's regressions/6162-extension-active-tools-next-turn.test.ts.

pi's faux response callables take `(context)`; pidrei's take
`(context, options, state, model)`, so each one absorbs the extra arguments.
"""

import pytest

from pidrei.core.extensions import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness


def make_tool(name: str, label: str, description: str, execute, prompt_snippet: str | None = None):
    return ToolDefinition(
        name=name,
        label=label,
        description=description,
        prompt_snippet=prompt_snippet,
        parameters={"type": "object", "properties": {}},
        execute=execute,
    )


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


@pytest.mark.tonio
async def test_applies_set_active_tools_before_the_next_provider_request(harnesses):
    def factory(pi) -> None:
        async def switch(*_args):
            pi.set_active_tools(["after_switch"])
            return AgentToolResult(content=[TextContent(text="switched")], details={})

        async def after(*_args):
            return AgentToolResult(content=[TextContent(text="after")], details={})

        pi.register_tool(
            make_tool(
                "switch_tools",
                "Switch Tools",
                "Switch the active extension tool set",
                switch,
                "Switch to the next extension tool",
            )
        )
        pi.register_tool(
            make_tool(
                "after_switch",
                "After Switch",
                "Tool that should be available after switching",
                after,
                "Run after the active tool set changes",
            )
        )

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.session.set_active_tools_by_name(["switch_tools"])

    provider_tool_names: list[list[str]] = []

    async def first(context, *_rest):
        provider_tool_names.append(sorted(tool.name for tool in (context.tools or [])))
        return faux_assistant_message(faux_tool_call("switch_tools", {}), stop_reason="toolUse")

    async def second(context, *_rest):
        provider_tool_names.append(sorted(tool.name for tool in (context.tools or [])))
        return faux_assistant_message("done")

    harness.set_responses([first, second])

    assert harness.session.get_active_tool_names() == ["switch_tools"]

    await harness.session.prompt("start")

    assert harness.session.get_active_tool_names() == ["after_switch"]
    assert provider_tool_names == [["switch_tools"], ["after_switch"]]


@pytest.mark.tonio
async def test_records_additive_active_tool_changes_on_the_current_tool_result(harnesses):
    def factory(pi) -> None:
        async def load_more(*_args):
            pi.set_active_tools([*pi.get_active_tools(), "after_load"])
            return AgentToolResult(content=[TextContent(text="loaded")], details={})

        async def after(*_args):
            return AgentToolResult(content=[TextContent(text="after")], details={})

        pi.register_tool(make_tool("load_more_tools", "Load More Tools", "Load more tools", load_more))
        pi.register_tool(make_tool("after_load", "After Load", "Tool available after loading", after))

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.session.set_active_tools_by_name(["load_more_tools"])

    added_tool_names: list[list[str]] = []

    async def first(_context, *_rest):
        return faux_assistant_message(faux_tool_call("load_more_tools", {}), stop_reason="toolUse")

    async def second(context, *_rest):
        added_tool_names.append(
            [
                name
                for message in context.messages
                if getattr(message, "role", None) == "toolResult"
                for name in (getattr(message, "added_tool_names", None) or [])
            ]
        )
        return faux_assistant_message("done")

    harness.set_responses([first, second])

    await harness.session.prompt("start")

    assert harness.session.get_active_tool_names() == ["load_more_tools", "after_load"]
    assert added_tool_names == [["after_load"]]


@pytest.mark.tonio
async def test_preserves_before_agent_start_system_prompt_overrides_when_tools_change(harnesses):
    def factory(pi) -> None:
        async def on_before_agent_start(event, _ctx):
            return {"systemPrompt": f"{event['systemPrompt']}\n\nkeep this run override"}

        async def switch(*_args):
            pi.set_active_tools(["after_switch"])
            return AgentToolResult(content=[TextContent(text="switched")], details={})

        async def after(*_args):
            return AgentToolResult(content=[TextContent(text="after")], details={})

        pi.on("before_agent_start", on_before_agent_start)
        pi.register_tool(
            make_tool(
                "switch_tools",
                "Switch Tools",
                "Switch the active extension tool set",
                switch,
                "Switch to the next extension tool",
            )
        )
        pi.register_tool(
            make_tool(
                "after_switch",
                "After Switch",
                "Tool that should be available after switching",
                after,
                "Run after the active tool set changes",
            )
        )

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    harness.session.set_active_tools_by_name(["switch_tools"])

    provider_system_prompts: list[str] = []
    provider_tool_names: list[list[str]] = []

    def record(context) -> None:
        provider_system_prompts.append(context.system_prompt or "")
        provider_tool_names.append(sorted(tool.name for tool in (context.tools or [])))

    async def first(context, *_rest):
        record(context)
        return faux_assistant_message(faux_tool_call("switch_tools", {}), stop_reason="toolUse")

    async def second(context, *_rest):
        record(context)
        return faux_assistant_message("done")

    harness.set_responses([first, second])

    await harness.session.prompt("start")

    assert provider_tool_names == [["switch_tools"], ["after_switch"]]
    assert len(provider_system_prompts) == 2
    assert "keep this run override" in provider_system_prompts[0]
    assert "keep this run override" in provider_system_prompts[1]
