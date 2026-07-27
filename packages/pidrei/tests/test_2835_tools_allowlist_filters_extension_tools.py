"""Mirror of pi's regressions/2835-tools-allowlist-filters-extension-tools.test.ts."""

import os
import shutil
import tempfile

import pytest

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.extensions import ToolDefinition
from pidrei.core.resource_loader import DefaultResourceLoader
from pidrei.core.sdk import CreateAgentSessionOptions, create_agent_session
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.providers.all import get_builtin_model


async def _ok(*_args):
    return {"content": [{"type": "text", "text": "ok"}], "details": {}}


def dynamic_tool_factory(pi) -> None:
    def on_session_start(_event, _ctx):
        pi.register_tool(
            ToolDefinition(
                name="dynamic_tool",
                label="Dynamic Tool",
                description="Tool registered from session_start",
                prompt_snippet="Run dynamic test behavior",
                parameters={"type": "object", "properties": {}},
                execute=_ok,
            )
        )

    pi.on("session_start", on_session_start)


class _Dirs:
    def __init__(self) -> None:
        self.root = tempfile.mkdtemp(prefix="pidrei-tools-filter-")
        self.agent_dir = os.path.join(self.root, "agent")
        os.makedirs(self.agent_dir)

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@pytest.fixture
def dirs(request):
    holder = _Dirs()
    request.addfinalizer(holder.cleanup)
    return holder


async def create_session(dirs, allowed_tool_names=None):
    settings_manager = await SettingsManager.create(dirs.root, dirs.agent_dir)
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
            tools=allowed_tool_names,
        )
    )
    await result.session.bind_extensions(ExtensionBindings())
    return result.session


@pytest.mark.tonio
async def test_allows_only_explicitly_listed_builtin_and_extension_tools(dirs):
    session = await create_session(dirs, ["read", "dynamic_tool"])

    assert sorted(tool.name for tool in session.get_all_tools()) == ["dynamic_tool", "read"]
    assert sorted(session.get_active_tool_names()) == ["dynamic_tool", "read"]
    assert "- read: Read file contents" in session.system_prompt
    assert "- dynamic_tool: Run dynamic test behavior" in session.system_prompt
    assert "- bash:" not in session.system_prompt
    assert "- edit:" not in session.system_prompt
    session.dispose()


@pytest.mark.tonio
async def test_disables_all_tools_when_the_allowlist_is_empty(dirs):
    session = await create_session(dirs, [])

    assert session.get_all_tools() == []
    assert session.get_active_tool_names() == []
    assert "Available tools:\n(none)" in session.system_prompt
    assert "dynamic_tool" not in session.system_prompt
    session.dispose()
