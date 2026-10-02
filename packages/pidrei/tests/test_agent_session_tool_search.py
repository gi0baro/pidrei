"""pidrei-only: `tool_search` in a session, and `update_active_tools()`.

pi checks loading in suite/agent-session-mcp.test.ts ("activates tool_search
for deferred MCP tools and keeps loaded tools declared on the branch", "finds
nothing to load when every matching tool is already declared") with MCP tools;
those cases port with MCP. Here an extension registers the tools, which is all
the MCP extension does for the session.

`update_active_tools()` is pidrei-only: pi's `tool_search` reads the active
tools and sets them in two calls, which nothing can interleave on its single
thread.
"""

import threading

import pytest
import tonio.colored as tonio

from pidrei.core.extensions import ToolDefinition
from pidrei.extensions.tool_search import create_tool_search_extension
from pidrei.extensions.tool_search.tool import TOOL_SEARCH_DESCRIPTION
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import TextContent

from .harness import create_harness


QUERY_SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


def docs_tools(exposure: str, calls: list[str]):
    async def extension(pi) -> None:
        async def search(_id, params, *_rest):
            calls.append(f"search:{params['query']}")
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


async def setup(harnesses, exposure: str, active: list[str], calls: list[str] | None = None):
    harness = await create_harness(
        initial_active_tool_names=active,
        extension_factories=[create_tool_search_extension(), docs_tools(exposure, calls if calls is not None else [])],
    )
    harnesses.append(harness)
    return harness


def tool_result(harness, name: str):
    return next(
        message for message in harness.session.messages if message.role == "toolResult" and message.tool_name == name
    )


@pytest.mark.tonio
async def test_loads_deferred_tools_and_keeps_them_declared_on_the_branch(harnesses):
    calls: list[str] = []
    harness = await setup(harnesses, "deferred", ["tool_search"], calls)
    harness.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("tool_search", {"query": "search the docs", "limit": 1})], stop_reason="toolUse"
            ),
            faux_assistant_message([faux_tool_call("docs_search", {"query": "loaded"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    await harness.session.prompt("find a docs tool")

    assert harness.session.get_active_tool_names() == ["tool_search", "docs_search"]
    # The description does not depend on the registered tools.
    tool_search = next(tool for tool in harness.session.agent.state.tools if tool.name == "tool_search")
    assert tool_search.description == TOOL_SEARCH_DESCRIPTION

    assert tool_result(harness, "tool_search").content[0].text == (
        "Loaded 1 tool. They are available from your next call:\n- docs_search: Search the docs."
    )
    # Only the loaded tool is added; earlier declarations are not repeated.
    load_messages = [
        message
        for message in harness.session.messages
        if message.role == "system" and any(tool.name == "docs_search" for tool in message.tools_added or [])
    ]
    assert len(load_messages) == 1
    assert [tool.name for tool in load_messages[0].tools_added] == ["docs_search"]
    assert tool_result(harness, "docs_search").content[0].text == "loaded guide\nloaded faq"
    assert calls == ["search:loaded"]

    # Loads are recorded in the transcript: navigating back before the load
    # drops the tool, navigating to a later entry restores it.
    branch = harness.session_manager.get_branch()
    first_user = next(entry for entry in branch if entry["type"] == "message" and entry["message"].role == "user")
    await harness.session.navigate_tree(first_user["id"])
    assert "docs_search" not in harness.session.get_active_tool_names()
    await harness.session.navigate_tree(branch[-1]["id"])
    assert "docs_search" in harness.session.get_active_tool_names()


@pytest.mark.tonio
async def test_finds_nothing_to_load_when_every_matching_tool_is_already_declared(harnesses):
    harness = await setup(harnesses, "direct", ["tool_search", "docs_search", "docs_fetch"])
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("tool_search", {"query": "docs"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    await harness.session.prompt("go")
    assert tool_result(harness, "tool_search").content[0].text == "No matching tools found."


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
