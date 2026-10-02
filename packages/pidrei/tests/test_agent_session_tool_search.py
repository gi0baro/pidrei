"""pidrei-only: `update_active_tools()` in a session.

pi's `tool_search` reads the active tools and sets them in two calls, which
nothing can interleave on its single thread. Loading tools with `tool_search`
in a session is covered with MCP tools in test_agent_session_mcp.py, as in pi.
"""

import threading

import pytest
import tonio.colored as tonio

from pidrei.core.extensions import ToolDefinition
from pidrei.extensions.tool_search import create_tool_search_extension
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent

from .harness import create_harness


QUERY_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def docs_tools(exposure: str):
    async def extension(pi) -> None:
        async def search(_id, params, *_rest):
            return AgentToolResult(content=[TextContent(text=f"{params['query']} guide\n{params['query']} faq")])

        async def fetch(*_rest):
            return AgentToolResult(content=[TextContent(text="page")])

        pi.register_tool(
            ToolDefinition(
                name="docs_search",
                label="docs_search",
                description="Search the docs.",
                parameters=QUERY_SCHEMA,
                exposure=exposure,
                execute=search,
            )
        )
        pi.register_tool(
            ToolDefinition(
                name="docs_fetch",
                label="docs_fetch",
                description="Fetch a page of the docs by path.",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}},
                exposure=exposure,
                execute=fetch,
            )
        )

    return extension


async def setup(harnesses, exposure: str, active: list[str]):
    harness = await create_harness(
        initial_active_tool_names=active,
        extension_factories=[create_tool_search_extension(), docs_tools(exposure)],
    )
    harnesses.append(harness)
    return harness


@pytest.mark.tonio
async def test_update_active_tools_sets_what_the_update_returns_for_the_active_tools(harnesses):
    harness = await setup(harnesses, "deferred", ["read", "tool_search"])
    seen: list[list[str]] = []

    def update(active: list[str]) -> list[str]:
        seen.append(active)
        return [*active, "docs_fetch"]

    harness.session.update_active_tools(update)
    assert seen == [["read", "tool_search"]]
    assert harness.session.get_active_tool_names() == ["read", "tool_search", "docs_fetch"]


@pytest.mark.tonio
async def test_update_active_tools_sets_nothing_when_the_update_returns_none(harnesses):
    harness = await setup(harnesses, "deferred", ["read", "tool_search"])
    harness.session.update_active_tools(lambda _active: None)
    assert harness.session.get_active_tool_names() == ["read", "tool_search"]


@pytest.mark.tonio
async def test_update_active_tools_holds_off_other_loadout_changes_while_the_update_runs(harnesses):
    """No other writer can change the active tools between the read the update
    gets and the write of its result: another thread cannot take the loadout
    guard while the update runs."""
    harness = await setup(harnesses, "deferred", ["read"])
    guard = harness.session._tool_loadout_guard
    update_entered = threading.Event()
    update_release = threading.Event()

    def update(active: list[str]) -> list[str]:
        # Held on the pool until the other thread has tried the guard.
        update_entered.set()
        update_release.wait(5.0)
        return active

    def try_acquire() -> bool:
        acquired = guard.acquire(blocking=False)
        if acquired:
            guard.release()
        return acquired

    async def run_update() -> None:
        await tonio.spawn_blocking(harness.session.update_active_tools, update)

    updating = tonio.spawn(run_update())
    try:
        await tonio.spawn_blocking(update_entered.wait, 5.0)
        assert update_entered.is_set()
        assert await tonio.spawn_blocking(try_acquire) is False
    finally:
        update_release.set()
    await updating
