"""Mirror of pi coding-agent test/default-tools-setting.test.ts."""

import os
import shutil
import tempfile

import pytest

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.agent_session_services import (
    CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
    create_agent_session_from_services,
    create_agent_session_services,
)
from pidrei.core.extensions import ToolDefinition
from pidrei.core.resource_loader import DefaultResourceLoader
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.providers.all import get_builtin_model


async def _ok(*_args):
    return {"content": [{"type": "text", "text": "ok"}], "details": {}}


def _tool(name: str, label: str, description: str) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        label=label,
        description=description,
        parameters={"type": "object", "properties": {}},
        execute=_ok,
    )


class _Dirs:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="pidrei-default-tools-")
        self.agent_dir = os.path.join(self.root, "agent")
        os.makedirs(self.agent_dir)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def dirs(request):
    holder = _Dirs()
    request.addfinalizer(holder.cleanup)
    return holder


async def create_session(dirs, default_tools, options=None, extension_factories=None):
    settings_manager = SettingsManager.in_memory({"defaultTools": default_tools})
    resource_loader = DefaultResourceLoader(
        cwd=dirs.root,
        agent_dir=dirs.agent_dir,
        settings_manager=settings_manager,
        extension_factories=extension_factories or [],
    )
    await resource_loader.reload()

    session_options = CreateAgentSessionOptions(
        cwd=dirs.root,
        agent_dir=dirs.agent_dir,
        model=get_builtin_model("anthropic", "claude-sonnet-4-5"),
        settings_manager=settings_manager,
        session_manager=SessionManager.in_memory(dirs.root),
        resource_loader=resource_loader,
    )
    for key, value in (options or {}).items():
        setattr(session_options, key, value)
    return (await create_agent_session(session_options)).session


@pytest.mark.tonio
async def test_uses_the_configured_list_as_the_initial_built_in_selection(dirs):
    session = await create_session(dirs, ["grep", "find"])
    try:
        assert sorted(tool.name for tool in session.get_all_tools()) == [
            "bash",
            "edit",
            "find",
            "grep",
            "ls",
            "read",
            "write",
        ]
        assert session.get_active_tool_names() == ["grep", "find"]
        assert "- grep:" in session.system_prompt
        assert "- read:" not in session.system_prompt
    finally:
        session.dispose()


@pytest.mark.tonio
async def test_keeps_extension_and_sdk_custom_tools_enabled(dirs):
    def factory(pi) -> None:
        pi.register_tool(_tool("static_tool", "Static Tool", "Statically registered extension tool"))

        def on_session_start(_event, _ctx):
            pi.register_tool(_tool("dynamic_tool", "Dynamic Tool", "Dynamically registered extension tool"))

        pi.on("session_start", on_session_start)

    session = await create_session(
        dirs,
        ["grep"],
        options={"custom_tools": [_tool("sdk_tool", "SDK Tool", "SDK custom tool")]},
        extension_factories=[factory],
    )
    try:
        await session.bind_extensions(ExtensionBindings())

        assert sorted(session.get_active_tool_names()) == ["dynamic_tool", "grep", "sdk_tool", "static_tool"]
        all_tool_names = [tool.name for tool in session.get_all_tools()]
        for name in ("read", "dynamic_tool", "sdk_tool", "static_tool"):
            assert name in all_tool_names
    finally:
        session.dispose()


@pytest.mark.tonio
async def test_preserves_explicit_tool_option_precedence(dirs):
    allowlisted_session = await create_session(dirs, ["grep"], options={"tools": ["read"]})
    try:
        assert allowlisted_session.get_active_tool_names() == ["read"]
    finally:
        allowlisted_session.dispose()

    excluded_session = await create_session(dirs, ["read", "grep"], options={"exclude_tools": ["read"]})
    try:
        assert excluded_session.get_active_tool_names() == ["grep"]
    finally:
        excluded_session.dispose()

    tool_less_session = await create_session(dirs, ["read"], options={"no_tools": "all"})
    try:
        assert tool_less_session.get_all_tools() == []
        assert tool_less_session.get_active_tool_names() == []
    finally:
        tool_less_session.dispose()


@pytest.mark.tonio
async def test_applies_through_service_based_session_creation(dirs):
    settings_manager = SettingsManager.in_memory({"defaultTools": ["ls"]})
    services = await create_agent_session_services(
        CreateAgentSessionServicesOptions(cwd=dirs.root, agent_dir=dirs.agent_dir, settings_manager=settings_manager)
    )
    result = await create_agent_session_from_services(
        CreateAgentSessionFromServicesOptions(
            services=services,
            session_manager=SessionManager.in_memory(dirs.root),
            model=get_builtin_model("anthropic", "claude-sonnet-4-5"),
        )
    )
    session = result.session
    try:
        assert sorted(tool.name for tool in session.get_all_tools()) == [
            "bash",
            "edit",
            "find",
            "grep",
            "ls",
            "read",
            "write",
        ]
        assert session.get_active_tool_names() == ["ls"]
    finally:
        session.dispose()
