"""Mirror of pi coding-agent test/default-tools-setting.test.ts."""

import json
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
    resource_loader = await DefaultResourceLoader(
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
async def test_activates_an_inactive_extension_tool_with_plus_name(dirs):
    async def factory(pi) -> None:
        tool = _tool("inactive_tool", "Inactive Tool", "Extension tool registered inactive")
        tool.default_active = False
        pi.register_tool(tool)

    session = await create_session(dirs, ["+inactive_tool", "-write"], extension_factories=[factory])
    try:
        assert sorted(session.get_active_tool_names()) == ["bash", "edit", "inactive_tool", "read"]
    finally:
        session.dispose()


@pytest.mark.tonio
async def test_keeps_extension_and_sdk_custom_tools_enabled(dirs):
    async def factory(pi) -> None:
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
async def test_applies_plus_name_and_minus_name_tool_options_to_the_default_selection(dirs):
    async def factory(pi) -> None:
        tool = _tool("inactive_tool", "Inactive Tool", "Extension tool registered inactive")
        tool.default_active = False
        pi.register_tool(tool)
        pi.register_tool(_tool("active_tool", "Active Tool", "Extension tool registered active"))

    session = await create_session(
        dirs, ["+grep"], options={"tools": ["+inactive_tool", "-write"]}, extension_factories=[factory]
    )
    try:
        assert sorted(session.get_active_tool_names()) == [
            "active_tool",
            "bash",
            "edit",
            "grep",
            "inactive_tool",
            "read",
        ]
    finally:
        session.dispose()

    tool_less = await create_session(
        dirs, ["read"], options={"no_tools": "all", "tools": ["+inactive_tool"]}, extension_factories=[factory]
    )
    try:
        assert tool_less.get_active_tool_names() == ["inactive_tool"]
    finally:
        tool_less.dispose()


@pytest.mark.tonio
async def test_rejects_invalid_tool_modifier_options(dirs):
    with pytest.raises(
        ValueError, match=r"^Invalid tools option: tool names cannot be mixed with \+name or -name entries$"
    ):
        await create_session(dirs, [], options={"tools": ["read", "+grep"]})
    with pytest.raises(
        ValueError,
        match=r"^Invalid tools option: \+name and -name entries take exact tool names, not patterns: -gr\*$",
    ):
        await create_session(dirs, [], options={"tools": ["-gr*"]})


# --- reload ---------------------------------------------------------------------


async def inactive_tool(pi) -> None:
    tool = _tool("inactive_tool", "Inactive Tool", "Extension tool registered inactive")
    tool.default_active = False
    pi.register_tool(tool)


def write_settings(dirs, settings: dict) -> None:
    with open(os.path.join(dirs.agent_dir, "settings.json"), "w", encoding="utf-8") as file:
        json.dump(settings, file)


async def create_file_session(dirs, options=None):
    settings_manager = await SettingsManager(dirs.root, dirs.agent_dir)
    resource_loader = await DefaultResourceLoader(
        cwd=dirs.root,
        agent_dir=dirs.agent_dir,
        settings_manager=settings_manager,
        extension_factories=[inactive_tool],
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


# #10245
@pytest.mark.tonio
async def test_reload_activates_only_tools_newly_added_to_default_tools(dirs):
    session = await create_file_session(dirs)
    try:
        assert session.get_active_tool_names() == ["read", "bash", "edit", "write"]
        session.set_active_tools_by_name(["read", "edit", "write"])

        write_settings(dirs, {"defaultTools": ["+inactive_tool", "+grep"]})
        await session.reload()
        # bash was disabled during the session and is not newly added, so it stays off.
        assert sorted(session.get_active_tool_names()) == ["edit", "grep", "inactive_tool", "read", "write"]

        # Removing tools from the setting does not disable them.
        write_settings(dirs, {"defaultTools": ["-read"]})
        await session.reload()
        assert sorted(session.get_active_tool_names()) == ["edit", "grep", "inactive_tool", "read", "write"]
    finally:
        session.dispose()


@pytest.mark.tonio
async def test_reload_keeps_tools_removed_by_minus_name_tool_options_removed(dirs):
    write_settings(dirs, {"defaultTools": ["read"]})
    session = await create_file_session(dirs, {"tools": ["-bash", "+grep"]})
    try:
        assert session.get_active_tool_names() == ["read", "grep"]

        write_settings(dirs, {"defaultTools": ["read", "bash", "inactive_tool"]})
        await session.reload()
        assert sorted(session.get_active_tool_names()) == ["grep", "inactive_tool", "read"]
    finally:
        session.dispose()


@pytest.mark.tonio
async def test_reload_keeps_explicit_tool_options(dirs):
    allowlisted = await create_file_session(dirs, {"tools": ["read"]})
    try:
        write_settings(dirs, {"defaultTools": ["+grep"]})
        await allowlisted.reload()
        assert allowlisted.get_active_tool_names() == ["read"]
    finally:
        allowlisted.dispose()

    write_settings(dirs, {})
    builtinless = await create_file_session(dirs, {"no_tools": "builtin"})
    try:
        write_settings(dirs, {"defaultTools": ["+grep"]})
        await builtinless.reload()
        assert builtinless.get_active_tool_names() == []
    finally:
        builtinless.dispose()

    write_settings(dirs, {})
    excluded = await create_file_session(dirs, {"exclude_tools": ["grep"]})
    try:
        write_settings(dirs, {"defaultTools": ["+grep", "+inactive_tool"]})
        await excluded.reload()
        assert sorted(excluded.get_active_tool_names()) == ["bash", "edit", "inactive_tool", "read", "write"]
    finally:
        excluded.dispose()


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
