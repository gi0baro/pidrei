"""Mirror of pi's regressions/3592-no-builtin-tools-keeps-extension-tools.test.ts."""

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
from pidrei.core.resource_loader import DefaultResourceLoader
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.providers.all import get_builtin_model

from .test_2835_tools_allowlist_filters_extension_tools import dynamic_tool_factory


class _Dirs:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="pidrei-no-builtin-tools-")
        self.agent_dir = os.path.join(self.root, "agent")
        os.makedirs(self.agent_dir)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def dirs(request):
    holder = _Dirs()
    request.addfinalizer(holder.cleanup)
    return holder


async def create_session(dirs, *, no_tools=None, tools=None):
    settings_manager = SettingsManager.create(dirs.root, dirs.agent_dir)
    session_manager = SessionManager.in_memory(dirs.root)
    resource_loader = DefaultResourceLoader(
        cwd=dirs.root,
        agent_dir=dirs.agent_dir,
        settings_manager=settings_manager,
        extension_factories=[dynamic_tool_factory],
    )
    await resource_loader.reload()

    result = await create_agent_session(
        CreateAgentSessionOptions(
            cwd=dirs.root,
            agent_dir=dirs.agent_dir,
            model=get_builtin_model("anthropic", "claude-sonnet-4-5"),
            settings_manager=settings_manager,
            session_manager=session_manager,
            resource_loader=resource_loader,
            no_tools=no_tools,
            tools=tools,
        )
    )
    await result.session.bind_extensions(ExtensionBindings())
    return result.session


@pytest.mark.tonio
async def test_keeps_extension_tools_active_when_builtin_defaults_are_disabled(dirs):
    session = await create_session(dirs, no_tools="builtin")

    assert sorted(tool.name for tool in session.get_all_tools()) == [
        "bash",
        "dynamic_tool",
        "edit",
        "find",
        "grep",
        "ls",
        "read",
        "write",
    ]
    assert session.get_active_tool_names() == ["dynamic_tool"]
    assert "- dynamic_tool: Run dynamic test behavior" in session.system_prompt
    assert "- read:" not in session.system_prompt
    assert "- bash:" not in session.system_prompt
    session.dispose()


@pytest.mark.tonio
async def test_still_disables_all_tools_when_no_tools_is_all(dirs):
    session = await create_session(dirs, no_tools="all")

    assert session.get_all_tools() == []
    assert session.get_active_tool_names() == []
    assert "Available tools:\n(none)" in session.system_prompt
    session.dispose()


@pytest.mark.tonio
async def test_propagates_no_tools_through_service_based_session_creation(dirs):
    settings_manager = SettingsManager.create(dirs.root, dirs.agent_dir)
    session_manager = SessionManager.in_memory(dirs.root)
    services = await create_agent_session_services(
        CreateAgentSessionServicesOptions(cwd=dirs.root, agent_dir=dirs.agent_dir, settings_manager=settings_manager)
    )

    result = await create_agent_session_from_services(
        CreateAgentSessionFromServicesOptions(
            services=services,
            session_manager=session_manager,
            model=get_builtin_model("anthropic", "claude-sonnet-4-5"),
            no_tools="builtin",
        )
    )

    assert result.session.get_active_tool_names() == []
    assert "Available tools:\n(none)" in result.session.system_prompt
    assert "- read:" not in result.session.system_prompt
    result.session.dispose()
