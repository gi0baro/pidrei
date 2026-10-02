"""Mirror of pi's suite/agent-session-mcp.test.ts, with the codemode scripts in
Python (recipe `codemode-python`).

Translations:
- Scripts: `Promise.allSettled` is `all_settled`, `tools[name](...)` is
  `call_tool(name, ...)`, `return x` is a last expression line, and options are
  keyword arguments. Keys inside data keep their wire spelling; names the
  script makes up are snake_case.
- pi's fake server answers `initialize` after a delay (`Infinity`: never).
  Here it answers at once, or when the test releases it: a releasing extension
  loaded before the MCP extension sets it at the event the case is about
  (`before_agent_start`, `tool_call`), so the case proves the wait instead of
  timing it. A server that never answers is never released (the fixture
  releases every server at teardown).
- `vi.waitFor` polling becomes a wait on `Changes`, which the wrapped
  `ExtensionAPI.register_tool` and the fake transport factory notify.
- pi binds the UI context with `bindExtensions` after creating the harness;
  here the harness binds once, with it.
- "prefers the mcp.json server over a registered ..." waits for the startup
  connections through `/mcp` (whose handler waits for them) where pi sleeps
  10 ms.
"""

import json
import os
import re
import threading
from collections.abc import Callable
from typing import Any

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.extensions import ToolDefinition
from pidrei.core.extensions.loader import ExtensionAPI
from pidrei.core.extensions.runner import emit_session_shutdown_event
from pidrei.core.extensions.types import ToolAnnotations
from pidrei.extensions.codemode import create_codemode_extension
from pidrei.extensions.mcp import MCP_SERVERS_SECTION, create_mcp_extension
from pidrei.extensions.mcp.config import LoadedMcpConfig, McpServerEntry
from pidrei.extensions.mcp.tools import create_mcp_tool_name
from pidrei.extensions.tool_search import create_tool_search_extension
from pidrei.extensions.tool_search.tool import TOOL_SEARCH_DESCRIPTION
from pidrei_agent.types import AgentToolResult
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_ai.types import ImageContent
from pidrei_mcp import LATEST_PROTOCOL_VERSION
from pidrei_mcp.testing import create_in_memory_transport_pair

from .agent_session_helpers import StubResourceLoader
from .harness import (
    create_harness,
    create_test_extensions_result,
    create_test_ui_context,
    get_assistant_texts,
    get_message_text,
    get_tool_result,
)


TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
_WAIT_S = 10

SERVER_TOOLS = [
    {
        "name": "search",
        "description": "Search the docs.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "outputSchema": {
            "type": "object",
            "properties": {"hits": {"type": "array", "items": {"type": "string"}}},
            "required": ["hits"],
        },
    },
    {
        "name": "fail",
        "description": "Always fails.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"title": "Fail", "destructiveHint": True, "readOnlyHint": False, "idempotentHint": "yes"},
    },
    {"name": "shot", "description": "Returns an image.", "inputSchema": {"type": "object", "properties": {}}},
]


class Changes:
    """Lets a test wait until a condition on the session holds: checked again
    on every tool registration and every transport the fake factory creates."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiters: list[tuple[Callable[[], bool], tonio.Event]] = []

    def notify(self) -> None:
        with self._lock:
            waiters = list(self._waiters)
        for predicate, event in waiters:
            if predicate():
                event.set()

    async def until(self, predicate: Callable[[], bool]) -> None:
        event = tonio.Event()
        with self._lock:
            self._waiters.append((predicate, event))
        try:
            if not predicate():
                await event.wait(_WAIT_S)
            assert predicate()
        finally:
            with self._lock:
                self._waiters.remove((predicate, event))


@pytest.fixture
def changes(monkeypatch):
    changes = Changes()
    original = ExtensionAPI.register_tool

    def register_tool(self, tool):
        original(self, tool)
        changes.notify()

    monkeypatch.setattr(ExtensionAPI, "register_tool", register_tool)
    return changes


class Releases:
    """Servers whose answer to `initialize` waits for the test."""

    def __init__(self) -> None:
        self.events: list[tonio.Event] = []

    def new(self) -> tonio.Event:
        event = tonio.Event()
        self.events.append(event)
        return event

    def release_all(self) -> None:
        for event in self.events:
            event.set()


@pytest.fixture
async def harnesses():
    """Harnesses a test created, each shut down afterwards (closing its MCP
    connections and codemode pool); then every held `initialize` is released."""
    created: list = []
    releases = Releases()
    try:
        yield created, releases
    finally:
        for harness in created:
            await emit_session_shutdown_event(
                harness.session.extension_runner, {"type": "session_shutdown", "reason": "quit"}
            )
        releases.release_all()
        for harness in created:
            harness.cleanup()


def releasing_on(event_name: str, release: tonio.Event):
    """An extension that releases a held server when `event_name` is emitted;
    loaded before the MCP extension, its handler runs first."""

    async def extension(pi) -> None:
        async def handler(_event, _ctx):
            release.set()

        pi.on(event_name, handler)

    return extension


async def fake_server(
    calls: list[str],
    *,
    list_tools: Callable[[], list] | None = None,
    resources: bool = False,
    instructions: str | None = None,
    release: tonio.Event | None = None,
):
    """Minimal MCP server over an in-memory transport. Records the tool calls
    it receives. With `release`, it answers `initialize` once that is set."""
    client, server = create_in_memory_transport_pair()

    def respond(request: dict[str, Any]) -> Any:
        method, params = request["method"], request.get("params") or {}
        match method:
            case "initialize":
                return {
                    "protocolVersion": LATEST_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}, **({"resources": {}} if resources else {})},
                    "serverInfo": {"name": "docs", "version": "1.0.0"},
                    **({"instructions": instructions} if instructions else {}),
                }
            case "tools/list":
                return {"tools": list_tools() if list_tools is not None else SERVER_TOOLS}
            case "resources/list":
                return {
                    "resources": [
                        {"uri": "docs://intro", "name": "intro", "mimeType": "text/markdown", "_meta": {"x": 1}},
                        # MCP App user interfaces are left out.
                        {"uri": "ui://docs/viewer", "name": "viewer", "mimeType": "text/html;profile=mcp-app"},
                    ]
                }
            case "resources/templates/list":
                return {
                    "resourceTemplates": [
                        {
                            "uriTemplate": "docs://pages/{slug}",
                            "name": "page",
                            "icons": [{"src": "data:image/png;base64,AAAA"}],
                        }
                    ]
                }
            case "resources/read":
                uri = params["uri"]
                calls.append(f"read:{uri}")
                return {"contents": [{"uri": uri, "mimeType": "text/markdown", "text": f"# {uri}"}]}
            case "tools/call":
                name, arguments = params["name"], params.get("arguments") or {}
                calls.append(f"{name}:{json.dumps(arguments, separators=(',', ':'))}")
                if name == "search":
                    hits = [f"{arguments.get('query')} guide", f"{arguments.get('query')} faq"]
                    return {"content": [{"type": "text", "text": "\n".join(hits)}], "structuredContent": {"hits": hits}}
                if name == "shot":
                    return {"content": [{"type": "image", "data": TINY_PNG_BASE64, "mimeType": "image/png"}]}
                return {"content": [{"type": "text", "text": "server exploded"}], "isError": True}
            case _:
                return {}

    async def on_message(message):
        if "id" not in message or "method" not in message:
            return
        response = {"jsonrpc": "2.0", "id": message["id"], "result": respond(message)}
        if message["method"] == "initialize" and release is not None:

            async def answer_later() -> None:
                await release.wait()
                server.send(response)

            tonio.spawn.without_tracking(answer_later())
            return
        server.send(response)

    server.on_message(on_message)
    await server.start()
    return client, server


def ui_bindings(notifications: list[str] | None = None, **overrides) -> ExtensionBindings:
    if notifications is not None:
        overrides["notify"] = lambda message, *_rest: notifications.append(message)
    return ExtensionBindings(ui_context=create_test_ui_context(**overrides))


def has_tool(harness, name: str) -> bool:
    return any(tool.name == name for tool in harness.session.get_all_tools())


async def setup(
    harnesses,
    changes,
    exposure: str,
    list_tools: Callable[[], list] | None = None,
    *,
    auto_enable_codemode: bool | None = None,
    built_in_tools: list[str] | None = None,
    extension_factories: tuple = (),
    tool_exposure: dict[str, str] | None = None,
    resources: bool = False,
    without_tool_search: bool = False,
    description: str | None = None,
    instructions: str | None = None,
):
    created, _releases = harnesses
    calls: list[str] = []
    notifications: list[str] = []
    servers: list = []
    config: dict[str, Any] = {"url": "http://unused.invalid", "exposure": exposure}
    if tool_exposure:
        config["toolExposure"] = tool_exposure
    if description:
        config["description"] = description
    entry = McpServerEntry(name="docs", config=config, source="test")

    async def load_config(_ctx):
        return LoadedMcpConfig(servers=[entry], errors=[], auto_enable_codemode=auto_enable_codemode)

    async def create_transport(_entry, _cwd, _auth_provider):
        client, server = await fake_server(calls, list_tools=list_tools, resources=resources, instructions=instructions)
        servers.append(server)
        return client

    # `built_in_tools` are the built-in tools active at the start. The MCP
    # extension activates codemode or tool_search.
    harness = await create_harness(
        initial_active_tool_names=built_in_tools if built_in_tools is not None else [],
        extension_factories=[
            *extension_factories,
            create_codemode_extension(),
            *([] if without_tool_search else [create_tool_search_extension()]),
            create_mcp_extension(load_config=load_config, create_transport=create_transport),
        ],
        extension_bindings=ui_bindings(notifications),
    )
    created.append(harness)
    # The first prompt waits only for servers with direct tools; wait for the others here.
    await changes.until(lambda: has_tool(harness, "mcp__docs__search"))
    return harness, calls, servers, notifications


def declared_tool_names(harness) -> list[str]:
    return [
        tool.name
        for message in harness.session.messages
        if message.role == "system"
        for tool in message.tools_added or []
    ]


def servers_section(harness) -> str | None:
    """The `mcp_servers` prompt section as the model currently has it."""
    section = None
    for message in harness.session.messages:
        if message.role == "system" and message.sections and MCP_SERVERS_SECTION in message.sections:
            section = message.sections[MCP_SERVERS_SECTION]
    return section


def nested_tool_names(harness) -> list[str]:
    return harness.session.get_callable_tool_names()


def script_value(harness) -> str:
    """The script's value: the last output line of the latest codemode result."""
    return get_message_text(get_tool_result(harness, "codemode")).split("\n")[-1]


def codemode_call(code: str):
    return faux_assistant_message([faux_tool_call("codemode", {"code": code})], stop_reason="toolUse")


def codemode_description(harness) -> str:
    return next((tool.description for tool in harness.session.agent.state.tools if tool.name == "codemode"), "")


# AgentSession MCP integration


@pytest.mark.tonio
async def test_exposes_codemode_only_mcp_tools_through_codemode_and_hides_them_from_the_model(harnesses, changes):
    harness, calls, _servers, _notifications = await setup(harnesses, changes, "codemode")
    search_name = create_mcp_tool_name("docs", "search")
    # MCP tools resolve to their CallToolResult, errors included.
    harness.set_responses(
        [
            codemode_call(
                f"""a, b = await all_settled(tools.{search_name}(query='mcp'), tools.{search_name}(query='pi'))
failure = await tools.mcp__docs__fail()
shot = await tools.mcp__docs__shot()
block = shot['content'][0]
if block['type'] == 'image':
    image(block)
hits = [hit for settled in (a, b) if settled['status'] == 'fulfilled' for hit in settled['value']['structuredContent']['hits']]
text({{
    'hits': hits,
    'failed': failure.get('isError'),
    'failure': failure['content'][0].get('text'),
    'found': [tool['name'] for tool in ALL_TOOLS if 'search' in tool['name']],
}})"""
            ),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("search the docs")

    # Codemode was activated for the codemode-exposed server; MCP tools are never declared.
    assert harness.session.get_active_tool_names() == ["codemode"]
    assert declared_tool_names(harness) == ["codemode"]
    assert nested_tool_names(harness) == [search_name, "mcp__docs__fail", "mcp__docs__shot"]
    # The description lists neither the server nor its tools; scripts search for them.
    assert "mcp__docs" not in codemode_description(harness)
    assert "Shared MCP Types:" not in codemode_description(harness)

    result = get_tool_result(harness, "codemode")
    assert result.is_error is False, get_message_text(result)
    # Output items keep the order the script produced them in.
    assert result.content[1] == ImageContent(data=TINY_PNG_BASE64, mime_type="image/png")
    assert json.loads(result.content[2].text) == {
        "hits": ["mcp guide", "mcp faq", "pi guide", "pi faq"],
        "failed": True,
        "failure": "server exploded",
        "found": [search_name],
    }
    assert len(result.content) == 3
    # The two searches run concurrently (`all_settled`), so the server sees them in either order.
    assert sorted(calls[:2]) == ['search:{"query":"mcp"}', 'search:{"query":"pi"}']
    assert calls[2:] == ["fail:{}", "shot:{}"]


@pytest.mark.tonio
async def test_keeps_codemode_only_mcp_tools_callable_across_tree_navigation(harnesses, changes):
    harness, *_ = await setup(harnesses, changes, "codemode")
    search_name = create_mcp_tool_name("docs", "search")
    harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])
    await harness.session.prompt("first")
    await harness.session.prompt("second")
    assert harness.session.get_active_tool_names() == ["codemode"]
    assert search_name in nested_tool_names(harness)

    first_assistant = next(
        entry
        for entry in harness.session_manager.get_branch()
        if entry["type"] == "message" and entry["message"].role == "assistant"
    )
    await harness.session.navigate_tree(first_assistant["id"])

    assert harness.session.get_active_tool_names() == ["codemode"]
    assert search_name in nested_tool_names(harness)


@pytest.mark.tonio
@pytest.mark.parametrize("reverse", [False, True])
async def test_routes_tools_whose_names_differ_only_in_dash_and_underscore(harnesses, changes, reverse):
    # Regression: #10239.
    tools = [
        {"name": "read-file", "description": "dashed", "inputSchema": {"type": "object", "properties": {}}},
        {"name": "read_file", "description": "underscored", "inputSchema": {"type": "object", "properties": {}}},
    ]
    if reverse:
        tools.reverse()
    harness, calls, *_ = await setup(harnesses, changes, "codemode", lambda: [*tools, *SERVER_TOOLS])
    code = "for query in ['dashed', 'underscored']:\n    await call_tool((await search_tools(query))[0]['name'])"
    harness.set_responses([codemode_call(code), faux_assistant_message("done")])

    await harness.session.prompt("go")

    assert calls == ["read-file:{}", "read_file:{}"]


@pytest.mark.tonio
async def test_rejects_direct_model_calls_to_codemode_only_mcp_tools(harnesses, changes):
    harness, calls, *_ = await setup(harnesses, changes, "codemode")
    harness.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call(create_mcp_tool_name("docs", "search"), {"query": "x"})], stop_reason="toolUse"
            ),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("go")

    result = get_tool_result(harness, "mcp__docs__search")
    assert result.is_error is True
    assert get_message_text(result) == "Tool mcp__docs__search not found"
    assert calls == []


@pytest.mark.tonio
async def test_declares_directly_exposed_mcp_tools_to_the_model(harnesses, changes):
    harness, *_ = await setup(harnesses, changes, "direct")
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("mcp__docs__search", {"query": "direct"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("go")

    assert declared_tool_names(harness) == ["mcp__docs__search", "mcp__docs__fail", "mcp__docs__shot"]
    assert "codemode" not in harness.session.get_active_tool_names()
    # Boolean annotation hints are passed on for permission extensions.
    annotations = {tool.name: tool.annotations for tool in harness.session.get_all_tools()}
    assert annotations["mcp__docs__fail"] == ToolAnnotations(destructive_hint=True, read_only_hint=False)
    assert annotations["mcp__docs__search"] is None
    assert get_message_text(get_tool_result(harness, "mcp__docs__search")) == "direct guide\ndirect faq"


@pytest.mark.tonio
@pytest.mark.parametrize("exposure", ["direct", "codemode"])
async def test_withdraws_and_restores_mcp_tools_the_server_changes(harnesses, changes, exposure):
    tools = SERVER_TOOLS
    harness, _calls, servers, _notifications = await setup(harnesses, changes, exposure, lambda: tools)
    harness.set_responses([faux_assistant_message("ready")])
    await harness.session.prompt("start")

    # Direct tools are declared to the model, codemode tools are only callable from codemode.
    def reachable() -> list[str]:
        return harness.session.get_active_tool_names() if exposure == "direct" else nested_tool_names(harness)

    async def list_changed(condition: Callable[[], bool]) -> None:
        servers[0].send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
        await changes.until(condition)

    assert "mcp__docs__fail" in reachable()

    tools = [tool for tool in SERVER_TOOLS if tool["name"] != "fail"]
    await list_changed(lambda: "mcp__docs__fail" not in reachable())
    assert "mcp__docs__search" in reachable()
    assert "mcp__docs__fail" not in codemode_description(harness)

    tools = SERVER_TOOLS
    await list_changed(lambda: "mcp__docs__fail" in reachable())


@pytest.mark.tonio
async def test_lists_and_reads_resources_with_codex_s_resource_tools(harnesses, changes):
    harness, calls, *_ = await setup(harnesses, changes, "direct", resources=True)
    harness.set_responses(
        [
            faux_assistant_message(
                [
                    faux_tool_call("list_mcp_resources", {}),
                    faux_tool_call("list_mcp_resource_templates", {"server": "docs"}),
                    faux_tool_call("read_mcp_resource", {"server": "docs", "uri": "docs://pages/setup"}),
                ],
                stop_reason="toolUse",
            ),
            faux_assistant_message(
                [faux_tool_call("read_mcp_resource", {"server": "nope", "uri": "docs://x"})], stop_reason="toolUse"
            ),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("read")

    # The resource tools take the exposure of the servers they reach.
    assert {"list_mcp_resources", "list_mcp_resource_templates", "read_mcp_resource"} <= set(
        harness.session.get_active_tool_names()
    )
    assert json.loads(get_message_text(get_tool_result(harness, "list_mcp_resources"))) == {
        "resources": [{"server": "docs", "uri": "docs://intro", "name": "intro", "mimeType": "text/markdown"}]
    }
    assert json.loads(get_message_text(get_tool_result(harness, "list_mcp_resource_templates"))) == {
        "server": "docs",
        "resourceTemplates": [{"server": "docs", "uriTemplate": "docs://pages/{slug}", "name": "page"}],
    }
    results = [
        message
        for message in harness.session.messages
        if message.role == "toolResult" and message.tool_name == "read_mcp_resource"
    ]
    assert get_message_text(results[0]) == "# docs://pages/setup"
    assert results[1].is_error is True
    assert get_message_text(results[1]) == 'MCP server "nope" has no resources. Servers with resources: docs'
    assert calls == ["read:docs://pages/setup"]
    read_tool = next(tool for tool in harness.session.get_all_tools() if tool.name == "read_mcp_resource")
    assert read_tool.annotations == ToolAnnotations(read_only_hint=True)


@pytest.mark.tonio
async def test_makes_the_resource_tools_callable_from_codemode_for_codemode_servers(harnesses, changes):
    harness, calls, *_ = await setup(harnesses, changes, "codemode", resources=True)
    harness.set_responses(
        [
            codemode_call(
                """listed = await tools.list_mcp_resources(server='docs')
read = await tools.read_mcp_resource(server='docs', uri=listed['resources'][0]['uri'])
{'uris': [resource['uri'] for resource in listed['resources']], 'text': read['contents'][0].get('text')}"""
            ),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("read")

    assert harness.session.get_active_tool_names() == ["codemode"]
    assert json.loads(script_value(harness)) == {"uris": ["docs://intro"], "text": "# docs://intro"}
    assert calls == ["read:docs://intro"]


@pytest.mark.tonio
async def test_applies_per_tool_exposure_overrides(harnesses, changes):
    harness, *_ = await setup(harnesses, changes, "hidden", tool_exposure={"search": "direct", "s*": "codemode"})
    harness.set_responses([faux_assistant_message("ready")])
    await harness.session.prompt("start")

    # `search` is declared, `shot` is only callable from codemode, `fail` keeps the server's `hidden`.
    assert declared_tool_names(harness) == ["codemode", "mcp__docs__search"]
    assert nested_tool_names(harness) == ["mcp__docs__search", "mcp__docs__shot"]
    assert "mcp__docs__shot" not in codemode_description(harness)
    assert "mcp__docs__fail" not in codemode_description(harness)


@pytest.mark.tonio
async def test_describes_the_server_with_its_configured_description_and_returns_its_instructions_to_scripts(
    harnesses, changes
):
    harness, *_ = await setup(
        harnesses,
        changes,
        "codemode",
        description="Search the product docs",
        instructions="Always search before reading.",
        built_in_tools=["tool_search"],
    )
    harness.set_responses(
        [
            codemode_call(
                """docs = await describe_namespace('mcp__docs')
aliases = [await describe_namespace(name) for name in ['docs', 'mcp__docs']]
{'docs': docs, 'same_for_aliases': all(alias == docs for alias in aliases), 'none': await describe_namespace('mcp__nope')}"""
            ),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("go")

    # The instructions stay out of every tool description.
    def description(name: str) -> str:
        return next((tool.description for tool in harness.session.agent.state.tools if tool.name == name), "")

    assert "mcp__docs" not in description("codemode")
    assert description("tool_search") == TOOL_SEARCH_DESCRIPTION
    for name in ("codemode", "tool_search"):
        assert "Always search" not in description(name)
    assert json.loads(script_value(harness)) == {
        "docs": {
            "name": "mcp__docs",
            "description": "Search the product docs",
            "instructions": "Always search before reading.",
            "tools": ["mcp__docs__search", "mcp__docs__fail", "mcp__docs__shot"],
        },
        "same_for_aliases": True,
        "none": None,
    }
    # The system prompt lists the server with its configured description rather than its instructions.
    assert "- mcp__docs (codemode): Search the product docs" in servers_section(harness)


@pytest.mark.tonio
async def test_lists_servers_by_the_first_line_of_their_instructions_without_a_configured_description(
    harnesses, changes
):
    harness, *_ = await setup(harnesses, changes, "deferred", instructions="Docs search.\nLong guidance.")
    harness.set_responses([faux_assistant_message("done")])

    await harness.session.prompt("go")

    section = servers_section(harness)
    assert "- mcp__docs (tool_search): Docs search.\n" in section
    assert "Long guidance" not in section


@pytest.mark.tonio
async def test_does_not_activate_codemode_when_auto_enable_codemode_is_false(harnesses, changes):
    harness, _calls, _servers, notifications = await setup(harnesses, changes, "codemode", auto_enable_codemode=False)
    harness.set_responses([faux_assistant_message("ready")])
    await harness.session.prompt("start")

    assert harness.session.get_active_tool_names() == []
    assert "mcp__docs__search" in nested_tool_names(harness)
    assert notifications == [
        (
            "MCP tools are only reachable from the codemode or tool_search tool, but neither is active "
            "(autoEnableCodemode is false); they cannot be called."
        )
    ]


@pytest.mark.tonio
async def test_treats_codemode_mcp_tools_as_reachable_through_an_active_tool_search(harnesses, changes):
    harness, _calls, _servers, notifications = await setup(
        harnesses, changes, "codemode", auto_enable_codemode=False, built_in_tools=["tool_search"]
    )
    harness.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("tool_search", {"query": "search the docs", "limit": 1})], stop_reason="toolUse"
            ),
            faux_assistant_message([faux_tool_call("mcp__docs__search", {"query": "loaded"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    await harness.session.prompt("go")

    assert harness.session.get_active_tool_names() == ["tool_search", "mcp__docs__search"]
    assert get_message_text(get_tool_result(harness, "mcp__docs__search")) == "loaded guide\nloaded faq"
    assert notifications == []


@pytest.mark.tonio
async def test_reaches_deferred_mcp_tools_through_codemode_when_tool_search_is_not_available(harnesses, changes):
    harness, _calls, _servers, notifications = await setup(
        harnesses, changes, "deferred", built_in_tools=["codemode"], without_tool_search=True
    )
    harness.set_responses([faux_assistant_message("ready")])
    await harness.session.prompt("start")

    assert harness.session.get_active_tool_names() == ["codemode"]
    assert "mcp__docs__search" in nested_tool_names(harness)
    assert notifications == []


@pytest.mark.tonio
async def test_warns_when_deferred_mcp_tools_have_neither_tool_search_nor_codemode(harnesses, changes):
    harness, _calls, _servers, notifications = await setup(harnesses, changes, "deferred", without_tool_search=True)
    harness.set_responses([faux_assistant_message("ready")])
    await harness.session.prompt("start")

    assert harness.session.get_active_tool_names() == []
    assert notifications == [
        "MCP tools are only reachable from the codemode or tool_search tool, but neither is active; they cannot be called."
    ]


@pytest.mark.tonio
async def test_does_not_activate_another_extension_s_tool_named_codemode(harnesses, changes):
    # Registered first, so it wins over the codemode extension's codemode.
    async def other_codemode(pi) -> None:
        async def execute(*_args):
            return AgentToolResult(content=[])

        pi.register_tool(
            ToolDefinition(
                name="codemode",
                label="codemode",
                description="Another extension's codemode tool.",
                parameters={"type": "object", "properties": {}},
                default_active=False,
                execute=execute,
            )
        )

    harness, *_ = await setup(harnesses, changes, "codemode", extension_factories=(other_codemode,))
    harness.set_responses([faux_assistant_message("ready")])
    await harness.session.prompt("start")

    assert harness.session.get_active_tool_names() == []


async def setup_slow(
    harnesses,
    changes,
    *,
    exposure: str | None = None,
    tool_exposure: dict[str, str] | None = None,
    name: str = "slow",
    instructions: str | None = None,
    release: tonio.Event | None = None,
    release_on: str | None = None,
    startup_wait_ms: float | None = None,
):
    """A server named `slow` that answers `initialize` once `release` is set,
    without waiting for it. `release_on` releases it at that event."""
    created, _releases = harnesses
    calls: list[str] = []
    notifications: list[str] = []
    config: dict[str, Any] = {"url": "http://unused.invalid"}
    if exposure:
        config["exposure"] = exposure
    if tool_exposure:
        config["toolExposure"] = tool_exposure
    entry = McpServerEntry(name=name, config=config, source="test")

    async def load_config(_ctx):
        return LoadedMcpConfig(servers=[entry], errors=[])

    async def create_transport(_entry, _cwd, _auth_provider):
        client, _server = await fake_server(calls, instructions=instructions, release=release)
        return client

    harness = await create_harness(
        initial_active_tool_names=[],
        extension_factories=[
            *([releasing_on(release_on, release)] if release_on and release is not None else []),
            create_codemode_extension(),
            create_tool_search_extension(),
            create_mcp_extension(
                load_config=load_config, create_transport=create_transport, startup_wait_ms=startup_wait_ms
            ),
        ],
        extension_bindings=ui_bindings(notifications),
    )
    created.append(harness)
    return harness, calls, notifications


def held(harnesses) -> tonio.Event:
    _created, releases = harnesses
    return releases.new()


@pytest.mark.tonio
async def test_does_not_hold_the_first_prompt_for_codemode_servers_that_are_still_connecting(harnesses, changes):
    # The server never answers `initialize`.
    harness, *_ = await setup_slow(harnesses, changes, release=held(harnesses))
    harness.set_responses([faux_assistant_message("ready")])

    await harness.session.prompt("start")

    assert get_assistant_texts(harness) == ["ready"]
    # Codemode is activated from the config, before the server connects.
    assert declared_tool_names(harness) == ["codemode"]


@pytest.mark.tonio
async def test_keeps_the_codemode_description_unchanged_when_the_server_connects(harnesses, changes):
    release = held(harnesses)
    harness, *_ = await setup_slow(harnesses, changes, release=release)
    before = codemode_description(harness)
    assert before
    release.set()
    await changes.until(lambda: "mcp__slow__search" in harness.session.get_callable_tool_names())
    assert codemode_description(harness) == before


@pytest.mark.tonio
async def test_lists_a_server_before_it_connects_and_appends_its_summary_with_the_next_prompt(harnesses, changes):
    release = held(harnesses)
    harness, *_ = await setup_slow(harnesses, changes, instructions="Slow docs.\nMore.", release=release)
    harness.set_responses([faux_assistant_message("one"), faux_assistant_message("two")])

    await harness.session.prompt("first")
    release.set()
    await changes.until(lambda: "mcp__slow__search" in harness.session.get_callable_tool_names())
    await harness.session.prompt("second")

    system_messages = [message for message in harness.session.messages if message.role == "system"]
    # The first request lists the server by name; its summary follows as an appended patch.
    assert len(system_messages) == 2
    assert "- mcp__slow (codemode)\n" in system_messages[0].sections[MCP_SERVERS_SECTION]
    assert list(system_messages[1].sections) == [MCP_SERVERS_SECTION]
    assert "- mcp__slow (codemode): Slow docs." in system_messages[1].sections[MCP_SERVERS_SECTION]
    first_assistant = next(
        index for index, message in enumerate(harness.session.messages) if message.role == "assistant"
    )
    assert harness.session.messages.index(system_messages[1]) > first_assistant


@pytest.mark.tonio
async def test_waits_for_servers_with_direct_tools_before_listing_the_servers(harnesses, changes):
    release = held(harnesses)
    harness, *_ = await setup_slow(
        harnesses,
        changes,
        exposure="direct",
        tool_exposure={"shot": "codemode"},
        instructions="Slow docs.",
        release=release,
        release_on="before_agent_start",
    )
    harness.set_responses([faux_assistant_message("ready")])

    await harness.session.prompt("start")

    assert "- mcp__slow (codemode): Slow docs." in servers_section(harness)


@pytest.mark.tonio
async def test_leaves_servers_with_only_direct_tools_out_of_the_servers_section(harnesses, changes):
    harness, *_ = await setup_slow(harnesses, changes, exposure="direct")
    harness.set_responses([faux_assistant_message("ready")])

    await harness.session.prompt("start")

    assert servers_section(harness) is None


@pytest.mark.tonio
async def test_waits_for_the_servers_a_codemode_script_names(harnesses, changes):
    harness, calls, _notifications = await setup_slow(
        harnesses, changes, release=held(harnesses), release_on="tool_call"
    )
    harness.set_responses(
        [
            codemode_call("(await tools.mcp__slow__search(query='q')).get('structuredContent')"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("go")

    assert get_tool_result(harness, "codemode").is_error is False, get_message_text(
        get_tool_result(harness, "codemode")
    )
    assert json.loads(script_value(harness)) == {"hits": ["q guide", "q faq"]}
    assert calls == ['search:{"query":"q"}']


@pytest.mark.tonio
async def test_waits_for_servers_whose_script_identifiers_differ_from_their_names(harnesses, changes):
    harness, calls, _notifications = await setup_slow(
        harnesses, changes, name="slow-docs", release=held(harnesses), release_on="tool_call"
    )
    harness.set_responses(
        [
            codemode_call(
                "await tools.mcp__slow_docs__search(query='q')\n"
                "namespace = await describe_namespace('slow_docs')\n"
                "namespace['name'] if namespace else None"
            ),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("go")

    assert get_tool_result(harness, "codemode").is_error is False, get_message_text(
        get_tool_result(harness, "codemode")
    )
    assert script_value(harness) == "mcp__slow_docs"
    assert calls == ['search:{"query":"q"}']


@pytest.mark.tonio
async def test_does_not_wait_for_servers_a_codemode_script_does_not_name(harnesses, changes):
    harness, *_ = await setup_slow(harnesses, changes, release=held(harnesses))
    harness.set_responses([codemode_call("1 + 1"), faux_assistant_message("done")])

    await harness.session.prompt("go")

    assert script_value(harness) == "2"


@pytest.mark.tonio
async def test_waits_for_servers_before_tool_search_searches(harnesses, changes):
    harness, *_ = await setup_slow(
        harnesses, changes, exposure="deferred", release=held(harnesses), release_on="tool_call"
    )
    harness.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("tool_search", {"query": "search the docs", "limit": 1})], stop_reason="toolUse"
            ),
            faux_assistant_message([faux_tool_call("mcp__slow__search", {"query": "late"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("go")

    assert get_message_text(get_tool_result(harness, "mcp__slow__search")) == "late guide\nlate faq"


@pytest.mark.tonio
async def test_holds_the_first_prompt_for_servers_with_direct_tools(harnesses, changes):
    harness, *_ = await setup_slow(
        harnesses, changes, exposure="direct", release=held(harnesses), release_on="before_agent_start"
    )
    harness.set_responses([faux_assistant_message("ready")])

    await harness.session.prompt("start")

    assert "mcp__slow__search" in declared_tool_names(harness)


@pytest.mark.tonio
async def test_holds_the_first_prompt_for_servers_with_direct_tools_only_up_to_startup_wait_ms(harnesses, changes):
    harness, _calls, notifications = await setup_slow(
        harnesses, changes, exposure="direct", release=held(harnesses), startup_wait_ms=20
    )
    harness.set_responses([faux_assistant_message("ready")])

    await harness.session.prompt("start")

    assert get_assistant_texts(harness) == ["ready"]
    assert harness.session.get_active_tool_names() == []
    assert notifications == ["MCP servers are still connecting; their tools become available once connected."]


@pytest.mark.tonio
async def test_does_not_let_codemode_call_itself_or_inactive_direct_tools(harnesses, changes):
    harness, *_ = await setup(harnesses, changes, "direct")
    harness.set_responses([faux_assistant_message("ready")])
    await harness.session.prompt("start")
    harness.session.set_active_tools_by_name(["codemode", "mcp__docs__search"])

    assert nested_tool_names(harness) == ["mcp__docs__search"]


@pytest.mark.tonio
async def test_finds_tools_from_scripts_with_search_tools_and_describe_tool(harnesses, changes):
    harness, calls, *_ = await setup(harnesses, changes, "codemode")
    harness.set_responses(
        [
            codemode_call(
                """match = (await search_tools('search the docs', limit=1))[0]
none = await search_tools('docs', namespace='mcp__other')
declaration = await describe_tool(match['name'])
result = await call_tool(match['name'], query='found')
text({
    'name': match['name'],
    'same_as_all_tools': [tool for tool in ALL_TOOLS if tool['name'] == match['name']][0]['description'] == match['description'],
    'none': len(none),
    'declared': 'codemode tool declaration:' in (declaration or ''),
    'missing': (await describe_tool('nope')) is None,
    'hits': result['structuredContent']['hits'],
})"""
            ),
            faux_assistant_message("done"),
        ]
    )

    await harness.session.prompt("go")

    result = get_tool_result(harness, "codemode")
    assert result.is_error is False, get_message_text(result)
    assert json.loads(result.content[1].text) == {
        "name": "mcp__docs__search",
        "same_as_all_tools": True,
        "none": 0,
        "declared": True,
        "missing": True,
        "hits": ["found guide", "found faq"],
    }
    assert calls == ['search:{"query":"found"}']


@pytest.mark.tonio
async def test_activates_tool_search_for_deferred_mcp_tools_and_keeps_loaded_tools_declared_on_the_branch(
    harnesses, changes
):
    # No built-in tools are active; the MCP extension activates tool_search, not codemode.
    harness, calls, *_ = await setup(harnesses, changes, "deferred")
    search_name = create_mcp_tool_name("docs", "search")
    harness.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("tool_search", {"query": "search the docs", "limit": 1})], stop_reason="toolUse"
            ),
            faux_assistant_message([faux_tool_call(search_name, {"query": "loaded"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    await harness.session.prompt("find a docs tool")

    assert harness.session.get_active_tool_names() == ["tool_search", search_name]
    # The description does not depend on the connected servers.
    tool_search = next(tool for tool in harness.session.agent.state.tools if tool.name == "tool_search")
    assert tool_search.description == TOOL_SEARCH_DESCRIPTION

    assert get_message_text(get_tool_result(harness, "tool_search")) == (
        f"Loaded 1 tool. They are available from your next call:\n- {search_name}: Search the docs."
    )
    # Only the loaded tool is added; earlier declarations are not repeated.
    load_messages = [
        message
        for message in harness.session.messages
        if message.role == "system" and any(tool.name == search_name for tool in message.tools_added or [])
    ]
    assert len(load_messages) == 1
    assert [tool.name for tool in load_messages[0].tools_added] == [search_name]
    assert get_message_text(get_tool_result(harness, search_name)) == "loaded guide\nloaded faq"
    assert calls == ['search:{"query":"loaded"}']

    # Loads are recorded in the transcript: navigating back before the load
    # drops the tool, navigating to a later entry restores it.
    branch = harness.session_manager.get_branch()
    first_user = next(entry for entry in branch if entry["type"] == "message" and entry["message"].role == "user")
    await harness.session.navigate_tree(first_user["id"])
    assert search_name not in harness.session.get_active_tool_names()
    await harness.session.navigate_tree(branch[-1]["id"])
    assert search_name in harness.session.get_active_tool_names()


@pytest.mark.tonio
async def test_finds_nothing_to_load_when_every_matching_tool_is_already_declared(harnesses, changes):
    harness, *_ = await setup(harnesses, changes, "direct", built_in_tools=["tool_search"])
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("tool_search", {"query": "docs"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    await harness.session.prompt("go")
    assert get_message_text(get_tool_result(harness, "tool_search")) == "No matching tools found."


# AgentSession MCP servers registered by extensions


async def setup_registered(harnesses, plugins: list, configured: list[McpServerEntry] | None = None):
    """`configured` are the mcp.json servers; `plugins` register servers through the extension API."""
    created, _releases = harnesses
    connected: list[McpServerEntry] = []

    async def load_config(_ctx):
        return LoadedMcpConfig(servers=list(configured or []), errors=[])

    async def create_transport(entry, _cwd, _auth_provider):
        connected.append(entry)
        client, _server = await fake_server([])
        return client

    harness = await create_harness(
        initial_active_tool_names=[],
        extension_factories=[
            *plugins,
            create_codemode_extension(),
            create_mcp_extension(load_config=load_config, create_transport=create_transport),
        ],
        extension_bindings=ui_bindings(),
    )
    created.append(harness)
    return harness, connected


@pytest.mark.tonio
async def test_connects_servers_registered_while_extensions_load(harnesses, changes):
    async def plugin(pi) -> None:
        pi.register_mcp_server("plugin", {"url": "http://plugin.invalid", "exposure": "direct"})

    harness, connected = await setup_registered(harnesses, [plugin])
    harness.set_responses([faux_assistant_message("ready")])
    await harness.session.prompt("start")

    assert [(entry.name, entry.scope) for entry in connected] == [("plugin", "extension")]
    assert "mcp__plugin__search" in harness.session.get_active_tool_names()


@pytest.mark.tonio
async def test_connects_and_disconnects_servers_registered_during_the_session(harnesses, changes):
    apis: list = []

    async def capture(pi) -> None:
        apis.append(pi)

    harness, connected = await setup_registered(harnesses, [capture])
    (pi,) = apis

    pi.register_mcp_server("late", {"url": "http://late.invalid"})
    await changes.until(lambda: "mcp__late__search" in harness.session.get_callable_tool_names())
    assert [entry.name for entry in connected] == ["late"]
    # Codemode-exposed tools need the codemode tool, which is activated for them.
    assert "codemode" in harness.session.get_active_tool_names()

    pi.unregister_mcp_server("late")
    await changes.until(lambda: "mcp__late__search" not in harness.session.get_callable_tool_names())


@pytest.mark.tonio
@pytest.mark.parametrize("name", ["my-docs", "my_docs"])
async def test_prefers_the_mcp_json_server_over_a_registered_server(harnesses, changes, name):
    # "my_docs" shares the namespace of "my-docs" (#10239).
    configured = McpServerEntry(name="my-docs", config={"url": "http://config.invalid"}, source="mcp.json")

    async def plugin(pi) -> None:
        pi.register_mcp_server(name, {"url": "http://plugin.invalid"})

    harness, connected = await setup_registered(harnesses, [plugin], [configured])
    # `/mcp` waits for the startup connections.
    await harness.session.prompt("/mcp")
    assert connected == [configured]


@pytest.mark.tonio
async def test_rejects_names_another_extension_registered(harnesses, changes):
    errors: list[str] = []

    async def first(pi) -> None:
        pi.register_mcp_server("taken", {"url": "http://x.invalid"})
        # Registering again replaces the extension's own registration.
        pi.register_mcp_server("taken", {"url": "http://y.invalid"})
        pi.register_mcp_server("my-server", {"url": "http://x.invalid"})

    async def second(pi) -> None:
        for name in ("taken", "my_server"):
            try:
                pi.register_mcp_server(name, {"url": "http://z.invalid"})
            except Exception as caught:
                errors.append(str(caught))

    await setup_registered(harnesses, [first, second])
    assert len(errors) == 2
    assert re.search(r'MCP server "taken" is already registered by extension', errors[0])
    # Names that differ only in - and _ share a namespace (#10239).
    assert errors[1] == 'MCP server "my_server" conflicts with registered server "my-server"'


@pytest.mark.tonio
async def test_reports_registered_servers_when_no_extension_connects_them(harnesses):
    created, _releases = harnesses
    errors: list[str] = []

    async def orphan(pi) -> None:
        pi.register_mcp_server("orphan", {"url": "http://orphan.invalid"})

    harness = await create_harness(
        extension_factories=[orphan],
        extension_bindings=ExtensionBindings(on_error=lambda error: errors.append(error.error)),
    )
    created.append(harness)

    assert len(errors) == 1
    assert 'MCP server "orphan" is registered, but no loaded extension' in errors[0]


# AgentSession MCP tools after resume and reload


class _ReloadingLoader(StubResourceLoader):
    """Loads the extension factories again on reload, as `/reload` does."""

    def __init__(self, extensions_result, factories: list) -> None:
        super().__init__(extensions_result)
        self._factories = factories

    async def reload(self, **_kwargs):
        self._extensions_result = await create_test_extensions_result(self._factories, os.getcwd())


async def setup_restored(harnesses, changes, session_manager=None, extension_factories=(), release=None):
    """A deferred `docs` server that answers `initialize` once `release` is
    set (at once without one); `connected` counts its connections. `/reload`
    loads the extensions again."""
    created, _releases = harnesses
    connected: list[str] = []
    servers = [
        McpServerEntry(name="docs", config={"url": "http://unused.invalid", "exposure": "deferred"}, source="test")
    ]

    async def load_config(_ctx):
        return LoadedMcpConfig(servers=servers, errors=[])

    async def create_transport(entry, _cwd, _auth_provider):
        connected.append(entry.name)
        changes.notify()
        client, _server = await fake_server([], release=release)
        return client

    factories = [
        *extension_factories,
        create_tool_search_extension(),
        create_mcp_extension(load_config=load_config, create_transport=create_transport),
    ]
    loader = _ReloadingLoader(await create_test_extensions_result(factories, os.getcwd()), factories)
    harness = await create_harness(
        resource_loader=loader, session_manager=session_manager, extension_bindings=ui_bindings()
    )
    created.append(harness)
    return harness, connected


async def load_docs_search(harness) -> None:
    harness.set_responses(
        [
            faux_assistant_message(
                [faux_tool_call("tool_search", {"query": "search the docs", "limit": 1})], stop_reason="toolUse"
            ),
            faux_assistant_message("loaded"),
        ]
    )
    await harness.session.prompt("load")
    assert "mcp__docs__search" in harness.session.get_active_tool_names()


@pytest.mark.tonio
async def test_declares_tools_tool_search_loaded_again_on_resume_once_their_server_connects(harnesses, changes):
    first, _ = await setup_restored(harnesses, changes)
    await load_docs_search(first)

    # The session restores its tools before the server connects again.
    second, _ = await setup_restored(harnesses, changes, first.session_manager)
    await changes.until(lambda: "mcp__docs__search" in second.session.get_active_tool_names())
    second.set_responses(
        [
            faux_assistant_message([faux_tool_call("mcp__docs__search", {"query": "again"})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    await second.session.prompt("use it")

    assert get_message_text(get_tool_result(second, "mcp__docs__search")) == "again guide\nagain faq"
    removals = [message for message in second.session.messages if message.role == "system" and message.tools_removed]
    assert removals == []


@pytest.mark.tonio
@pytest.mark.parametrize(("loadout", "kept"), [(["read"], False), (None, True)], ids=["drops", "keeps"])
async def test_restored_tools_when_an_extension_sets_the_loadout_before_they_register(
    harnesses, changes, loadout, kept
):
    first, _ = await setup_restored(harnesses, changes)
    await load_docs_search(first)

    # Like plan mode restoring its tools, or an extension adding one to the current loadout.
    async def set_loadout(pi) -> None:
        async def on_session_start(_event, _ctx):
            pi.set_active_tools(list(loadout) if loadout else [*pi.get_active_tools(), "read"])

        pi.on("session_start", on_session_start)

    second, _ = await setup_restored(harnesses, changes, first.session_manager, (set_loadout,))
    await changes.until(lambda: has_tool(second, "mcp__docs__search"))

    assert ("mcp__docs__search" in second.session.get_active_tool_names()) is kept


@pytest.mark.tonio
async def test_does_not_activate_restored_tools_that_register_after_the_next_prompt_starts(harnesses, changes):
    first, _ = await setup_restored(harnesses, changes)
    await load_docs_search(first)

    # The first prompt does not wait for servers without direct tools.
    release = held(harnesses)
    second, _ = await setup_restored(harnesses, changes, first.session_manager, release=release)
    second.set_responses([faux_assistant_message("done")])
    await second.session.prompt("go")
    release.set()
    await changes.until(lambda: has_tool(second, "mcp__docs__search"))

    assert "mcp__docs__search" not in second.session.get_active_tool_names()


@pytest.mark.tonio
async def test_declares_tools_tool_search_loaded_again_after_reload(harnesses, changes):
    harness, connected = await setup_restored(harnesses, changes)
    await load_docs_search(harness)

    await harness.session.reload()

    await changes.until(lambda: connected == ["docs", "docs"])
    await changes.until(lambda: "mcp__docs__search" in harness.session.get_active_tool_names())
