"""Mirror of pi's regressions/5109-exclude-tools.test.ts. pi binds the
extensions itself after creating the harness; here the harness binds them."""

import pytest

from pidrei.core.extensions import ToolDefinition

from .harness import create_harness


def tool_names(tools) -> list[str]:
    return sorted(tool.name for tool in tools)


async def _ok(*_args):
    return {"content": [{"type": "text", "text": "ok"}], "details": {}}


async def _tools_factory(pi) -> None:
    async def on_session_start(_event, _ctx):
        pi.register_tool(
            ToolDefinition(
                name="ask_question",
                label="Ask Question",
                description="Ask a question",
                prompt_snippet="Ask a question",
                parameters={"type": "object", "properties": {}},
                execute=_ok,
            )
        )
        pi.register_tool(
            ToolDefinition(
                name="dynamic_tool",
                label="Dynamic Tool",
                description="Dynamic test tool",
                prompt_snippet="Run dynamic test behavior",
                parameters={"type": "object", "properties": {}},
                execute=_ok,
            )
        )

    pi.on("session_start", on_session_start)


EXTENSION_FACTORIES = [_tools_factory]


@pytest.mark.tonio
async def test_filters_built_in_and_extension_tools_from_available_and_active_tools():
    harness = await create_harness(
        excluded_tool_names=["read", "ask_question"], extension_factories=EXTENSION_FACTORIES
    )
    try:
        all_tool_names = tool_names(harness.session.get_all_tools())
        assert "read" not in all_tool_names
        assert "ask_question" not in all_tool_names
        assert "bash" in all_tool_names
        assert "dynamic_tool" in all_tool_names
        assert sorted(harness.session.get_active_tool_names()) == ["bash", "dynamic_tool", "edit", "write"]
        assert "- read:" not in harness.session.system_prompt
        assert "ask_question" not in harness.session.system_prompt
        assert "- dynamic_tool: Run dynamic test behavior" in harness.session.system_prompt
    finally:
        harness.cleanup()


@pytest.mark.tonio
async def test_lets_excluded_tools_override_the_allowlist():
    harness = await create_harness(
        allowed_tool_names=["read", "bash", "ask_question"],
        excluded_tool_names=["read", "ask_question"],
        initial_active_tool_names=["read", "bash", "ask_question"],
        extension_factories=EXTENSION_FACTORIES,
    )
    try:
        assert tool_names(harness.session.get_all_tools()) == ["bash"]
        assert harness.session.get_active_tool_names() == ["bash"]
        assert "- bash:" in harness.session.system_prompt
        assert "- read:" not in harness.session.system_prompt
        assert "ask_question" not in harness.session.system_prompt
    finally:
        harness.cleanup()


@pytest.mark.tonio
async def test_matches_allowlist_and_denylist_patterns():
    harness = await create_harness(
        allowed_tool_names=["*_tool", "ask_*", "re*"],
        excluded_tool_names=["ask*"],
        initial_active_tool_names=["*_tool", "ask_*", "re*"],
        extension_factories=EXTENSION_FACTORIES,
    )
    try:
        # Patterns that match a tool activate it like its name.
        assert tool_names(harness.session.get_all_tools()) == ["dynamic_tool", "read"]
        assert sorted(harness.session.get_active_tool_names()) == ["dynamic_tool", "read"]
    finally:
        harness.cleanup()
