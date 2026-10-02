"""pidrei-only: tools of a restored or reloaded loadout that register late.

pi checks this in suite/agent-session-mcp.test.ts ("AgentSession MCP tools
after resume and reload") with deferred MCP tools that `tool_search` loaded
and an MCP server that connects after the session restored its tools. The MCP
extension and `tool_search` port later with codemode; here an extension
registers the tool when the test says so, which is what a server connecting
late amounts to for the session.
"""

import pytest

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.event_bus import EventBus
from pidrei.core.extensions import LoadExtensionsResult, ToolDefinition
from pidrei.core.extensions.loader import create_extension_runtime, load_extension_from_factory
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message
from pidrei_ai.types import TextContent
from pidrei_ai.utils.transcript import get_current_tools

from .agent_session_helpers import StubResourceLoader
from .harness import create_harness


EMPTY_SCHEMA = {"type": "object", "properties": {}}


class LateTool:
    """An extension whose `docs_search` tool registers at load when
    `at_load`, otherwise when the test calls `register()`. Registered inactive,
    so registering it does not activate it by itself."""

    def __init__(self, *, at_load: bool) -> None:
        self.at_load = at_load
        self.pi = None

    async def factory(self, pi) -> None:
        self.pi = pi
        if self.at_load:
            self.register()

    def register(self) -> None:
        async def execute(*_rest):
            return AgentToolResult(content=[TextContent(text="found")], details={})

        self.pi.register_tool(
            ToolDefinition(
                name="docs_search",
                label="docs_search",
                description="Search the docs.",
                parameters=EMPTY_SCHEMA,
                default_active=False,
                execute=execute,
            )
        )


class ReloadingResourceLoader(StubResourceLoader):
    """Loads the extensions again on `/reload`."""

    def __init__(self, factories: list, cwd: str) -> None:
        super().__init__()
        self._factories = factories
        self._cwd = cwd

    async def reload(self, **_kwargs) -> None:
        runtime = create_extension_runtime()
        extensions = [
            await load_extension_from_factory(factory, self._cwd, EventBus(), runtime, f"<inline:{index + 1}>")
            for index, factory in enumerate(self._factories)
        ]
        self._extensions_result = LoadExtensionsResult(extensions=extensions, runtime=runtime)


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def setup(harnesses, late: LateTool, session_manager=None, extension_factories=()):
    harness = await create_harness(
        session_manager=session_manager, extension_factories=[*extension_factories, late.factory]
    )
    harnesses.append(harness)
    await harness.session.bind_extensions(ExtensionBindings(shutdown_handler=lambda: None))
    return harness


async def load_docs_search(harnesses):
    """A first session that activated `docs_search` and recorded it in its transcript."""
    harness = await setup(harnesses, LateTool(at_load=True))
    harness.session.set_active_tools_by_name([*harness.session.get_active_tool_names(), "docs_search"])
    harness.set_responses([faux_assistant_message("loaded")])
    await harness.session.prompt("load")
    return harness


@pytest.mark.tonio
async def test_declares_restored_tools_again_on_resume_once_they_register(harnesses):
    first = await load_docs_search(harnesses)

    # The session restores its tools before the tool registers again.
    late = LateTool(at_load=False)
    second = await setup(harnesses, late, first.session_manager)
    assert "docs_search" not in second.session.get_active_tool_names()
    late.register()
    assert "docs_search" in second.session.get_active_tool_names()

    request_tools: list[list[str]] = []

    async def record(context, *_rest):
        request_tools.append([tool.name for tool in get_current_tools(context.messages)])
        return faux_assistant_message("done")

    second.set_responses([record])
    await second.session.prompt("use it")

    assert "docs_search" in request_tools[0]
    removals = [
        message
        for message in second.session.messages
        if message.role == "system" and len(message.tools_removed or []) > 0
    ]
    assert removals == []


@pytest.mark.parametrize(("loadout", "kept"), [(["read"], False), (None, True)], ids=["drops", "keeps"])
@pytest.mark.tonio
async def test_restored_tools_when_an_extension_sets_the_loadout_before_they_register(harnesses, loadout, kept):
    first = await load_docs_search(harnesses)

    # Like plan mode restoring its tools, or an extension adding one to the current loadout.
    async def set_loadout(pi) -> None:
        async def on_session_start(_event, _ctx):
            pi.set_active_tools(list(loadout) if loadout is not None else [*pi.get_active_tools(), "read"])

        pi.on("session_start", on_session_start)

    late = LateTool(at_load=False)
    second = await setup(harnesses, late, first.session_manager, [set_loadout])
    late.register()

    assert ("docs_search" in second.session.get_active_tool_names()) is kept


@pytest.mark.tonio
async def test_does_not_activate_restored_tools_that_register_after_the_next_prompt_starts(harnesses):
    first = await load_docs_search(harnesses)

    late = LateTool(at_load=False)
    second = await setup(harnesses, late, first.session_manager)
    second.set_responses([faux_assistant_message("done")])
    await second.session.prompt("go")
    late.register()

    assert "docs_search" in [tool.name for tool in second.session.get_all_tools()]
    assert "docs_search" not in second.session.get_active_tool_names()


@pytest.mark.tonio
async def test_declares_active_tools_again_after_reload_once_they_register(harnesses, tmp_path):
    late = LateTool(at_load=True)
    resource_loader = ReloadingResourceLoader([late.factory], str(tmp_path))
    await resource_loader.reload()
    harness = await create_harness(resource_loader=resource_loader)
    harnesses.append(harness)
    await harness.session.bind_extensions(ExtensionBindings(shutdown_handler=lambda: None))
    harness.session.set_active_tools_by_name([*harness.session.get_active_tool_names(), "docs_search"])

    late.at_load = False
    await harness.session.reload()
    assert "docs_search" not in harness.session.get_active_tool_names()
    late.register()

    assert "docs_search" in harness.session.get_active_tool_names()
