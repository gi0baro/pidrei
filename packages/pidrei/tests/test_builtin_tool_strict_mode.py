"""Mirror of pi coding-agent test/builtin-tool-strict-mode.test.ts.

pi's strict set includes `powershell`, which pidrei does not ship (POSIX-only).
"""

import dataclasses
import os
import shutil
import tempfile

import pytest
from tonio.colored import fs

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.resource_loader import DefaultResourceLoader
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei.core.tools import create_all_tool_definitions, create_all_tools
from pidrei.core.tools.tool_definition_wrapper import wrap_tool_definition
from pidrei_ai.providers.all import get_builtin_model
from pidrei_ai.types import JsonSchemaConstrainedSampling, TextContent


STRICT_TOOL_NAMES = ("read", "bash", "edit", "write")
PREFER_STRICT = JsonSchemaConstrainedSampling(strict="prefer")


@pytest.mark.parametrize("experimental", [None, "0", "1"])
def test_prefers_strict_sampling_regardless_of_experimental_mode(monkeypatch, experimental):
    if experimental is None:
        monkeypatch.delenv("PIDREI_EXPERIMENTAL", raising=False)
    else:
        monkeypatch.setenv("PIDREI_EXPERIMENTAL", experimental)
    definitions = create_all_tool_definitions(os.getcwd())
    tools = create_all_tools(os.getcwd())
    for name in STRICT_TOOL_NAMES:
        assert definitions[name].constrained_sampling == PREFER_STRICT
        assert tools[name].constrained_sampling == definitions[name].constrained_sampling
    for name in ("grep", "find", "ls"):
        assert definitions[name].constrained_sampling is None
    # Strictness is a provider-side conversion, not a change to the execution schema.
    assert definitions["read"].parameters["required"] == ["path"]
    assert definitions["bash"].parameters["required"] == ["command"]


def test_preserves_explicit_opt_outs_when_wrapping_definitions_for_execution():
    definitions = create_all_tool_definitions(os.getcwd())
    for name in STRICT_TOOL_NAMES:
        definition = definitions[name]
        override = dataclasses.replace(definition, constrained_sampling=False)
        assert wrap_tool_definition(override).constrained_sampling is False
        assert override.execute is definition.execute
        assert override.prepare_arguments is definition.prepare_arguments
        assert override.render_call is definition.render_call
        assert override.render_result is definition.render_result
        assert override.prompt_guidelines is definition.prompt_guidelines
        assert definition.constrained_sampling == PREFER_STRICT


@pytest.mark.tonio
@pytest.mark.parametrize("active_tools", [[], ["read"], list(STRICT_TOOL_NAMES)])
async def test_allows_extensions_to_re_register_tools_without_strict_sampling(active_tools):
    cwd = tempfile.mkdtemp(prefix="pidrei-non-strict-tools-")
    agent_dir = os.path.join(cwd, "agent")
    settings_manager = SettingsManager.in_memory({"defaultTools": active_tools})

    def factory(pi) -> None:
        async def on_session_start(_event, _ctx):
            definitions = create_all_tool_definitions(cwd)
            for name in STRICT_TOOL_NAMES:
                pi.register_tool(dataclasses.replace(definitions[name], constrained_sampling=False))
            pi.set_active_tools(active_tools)

        pi.on("session_start", on_session_start)

    resource_loader = DefaultResourceLoader(
        cwd=cwd,
        agent_dir=agent_dir,
        settings_manager=settings_manager,
        no_extensions=True,
        no_skills=True,
        no_prompt_templates=True,
        no_themes=True,
        extension_factories=[factory],
    )
    try:
        await resource_loader.reload()
        result = await create_agent_session(
            CreateAgentSessionOptions(
                cwd=cwd,
                agent_dir=agent_dir,
                model=get_builtin_model("anthropic", "claude-sonnet-4-5"),
                settings_manager=settings_manager,
                session_manager=SessionManager.in_memory(cwd),
                resource_loader=resource_loader,
            )
        )
        session = result.session
        try:
            original_prompt = session.system_prompt
            await session.bind_extensions(ExtensionBindings())
            assert session.get_active_tool_names() == active_tools
            assert session.system_prompt == original_prompt
            for name in STRICT_TOOL_NAMES:
                assert session.get_tool_definition(name).constrained_sampling is False
            for tool in session.agent.state.tools:
                assert tool.constrained_sampling is False
            if "read" in active_tools:
                await fs.Path(os.path.join(cwd, "sample.txt")).write_text("still works")
                read = next(tool for tool in session.agent.state.tools if tool.name == "read")
                read_result = await read.execute("read-test", {"path": "sample.txt"}, None, None)
                assert read_result.content == [TextContent(text="still works")]
        finally:
            session.dispose()
    finally:
        shutil.rmtree(cwd, ignore_errors=True)
