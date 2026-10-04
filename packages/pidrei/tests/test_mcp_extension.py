"""Mirror of pi's mcp-extension.test.ts.

Translations:
- Config loading and the transport factory are awaited (they read files and
  resolve config values).
- The in-memory fake server answers from its message listener; its replies
  are delivered on the transport's channel, as pi's `queueMicrotask` defers
  them.
- A failing `send` returns a failed `SendResult` where pi's throws.
- Where pi waits a turn (`setTimeout(0)`) for a dropped connection or a log
  write, the test waits on an Event the connection's `on_change` or the log
  sets.
- The connect retry delays are shortened through the module constant, so the
  retry cases do not sleep a quarter second.
- The `~` case runs a Python stdio server where pi's runs a node one.
"""

import json
import os
import re
import sys
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.auth_storage import InMemoryAuthStorageBackend
from pidrei.core.tools.truncate import truncate_middle
from pidrei.extensions.mcp import MAX_SERVERS_SECTION_CHARS, McpServerListing, render_servers_section, runtime
from pidrei.extensions.mcp.config import McpServerEntry, get_mcp_tool_exposure, load_mcp_config
from pidrei.extensions.mcp.runtime import (
    McpOAuthCredentialStore,
    McpServerConnection,
    McpServerLog,
    create_default_transport,
)
from pidrei.extensions.mcp.tools import ConvertMcpResultOptions, convert_mcp_result, create_mcp_tool_name
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import ImageContent, TextContent
from pidrei_mcp import LATEST_PROTOCOL_VERSION, McpAuthRequiredError, McpHttpError, McpSessionExpiredError, SendResult
from pidrei_mcp.oauth import McpOAuthAuthorizationRequiredError
from pidrei_mcp.testing import create_in_memory_transport_pair

from .mcp_oauth_server import loopback_servers


# Config values are resolved at connect time, so the literal reference must survive loading.
TOKEN_HEADER = "Bearer ${TOKEN}"
_WAIT_S = 5


def _setup(tmp_path, global_config, project_config):
    agent_dir = tmp_path / "agent"
    cwd = tmp_path / "project"
    agent_dir.mkdir(parents=True)
    (cwd / ".pidrei").mkdir(parents=True)
    (agent_dir / "mcp.json").write_text(json.dumps(global_config))
    (cwd / ".pidrei" / "mcp.json").write_text(json.dumps(project_config))
    return {"agent_dir": str(agent_dir), "cwd": str(cwd)}


# MCP config


@pytest.mark.tonio
async def test_merges_global_and_trusted_project_servers_and_validates_entries(tmp_path):
    paths = _setup(
        tmp_path,
        {
            "mcpServers": {
                "shared": {"command": "global-cmd"},
                "remote": {"url": "https://example.com/mcp", "headers": {"Authorization": TOKEN_HEADER}},
                "off": {"command": "x", "enabled": False},
                "bad": {"args": ["no command"]},
                "legacy": {"type": "sse", "url": "https://example.com/sse"},
                "badUrl": {"url": "example.com/mcp"},
                "bad name": {"command": "x"},
            }
        },
        {"mcpServers": {"shared": {"command": "project-cmd", "exposure": "direct"}}},
    )

    trusted = await load_mcp_config(**paths, project_trusted=True)
    # Disabled servers are kept so /mcp can enable them again.
    assert [(server.name, server.scope, server.config) for server in trusted.servers] == [
        ("shared", "project", {"command": "project-cmd", "exposure": "direct"}),
        ("remote", "global", {"url": "https://example.com/mcp", "headers": {"Authorization": TOKEN_HEADER}}),
        ("off", "global", {"command": "x", "enabled": False}),
    ]
    assert len(trusted.errors) == 4
    assert 'server "bad" needs either "command"' in trusted.errors[0]
    assert "legacy SSE transport is not supported" in trusted.errors[1]
    assert 'server "badUrl": url must be an http or https URL' in trusted.errors[2]
    assert 'invalid server name "bad name"' in trusted.errors[3]

    # Untrusted projects cannot add or override servers, since stdio servers run commands.
    untrusted = await load_mcp_config(**paths, project_trusted=False)
    assert next(server for server in untrusted.servers if server.name == "shared").config == {"command": "global-cmd"}


@pytest.mark.tonio
async def test_rejects_server_names_that_differ_only_in_dash_and_underscore(tmp_path):
    # Regression: #10239.
    paths = _setup(tmp_path, {"mcpServers": {"work-files": {"command": "a"}, "work_files": {"command": "b"}}}, {})
    loaded = await load_mcp_config(**paths, project_trusted=False)
    assert [server.name for server in loaded.servers] == ["work-files"]
    assert len(loaded.errors) == 1
    assert 'server "work_files" conflicts with "work-files"' in loaded.errors[0]


@pytest.mark.tonio
async def test_validates_exposure_and_reads_auto_enable_codemode_with_project_precedence(tmp_path):
    paths = _setup(
        tmp_path,
        {
            "autoEnableCodemode": False,
            "mcpServers": {
                "later": {"command": "x", "exposure": "deferred"},
                # `codemode-deferred` is an alias for `codemode`.
                "scripts": {
                    "command": "x",
                    "exposure": "codemode-deferred",
                    "toolExposure": {"a": "codemode-deferred"},
                },
                "off": {"command": "x", "exposure": "hidden"},
                "wrong": {"command": "x", "exposure": "model-only"},
                "described": {"command": "x", "description": "Docs search"},
                "badDescription": {"command": "x", "description": 1},
            },
        },
        {"autoEnableCodemode": "yes", "mcpServers": {}},
    )

    untrusted = await load_mcp_config(**paths, project_trusted=False)
    assert untrusted.auto_enable_codemode is False
    assert [(server.name, server.config.get("exposure")) for server in untrusted.servers] == [
        ("later", "deferred"),
        ("scripts", "codemode"),
        ("off", "hidden"),
        ("described", None),
    ]
    assert untrusted.servers[1].config["toolExposure"] == {"a": "codemode"}
    assert untrusted.servers[3].config["description"] == "Docs search"
    assert len(untrusted.errors) == 2
    assert 'server "wrong": exposure must be one of' in untrusted.errors[0]
    assert 'server "badDescription": description must be a string' in untrusted.errors[1]

    trusted = await load_mcp_config(**paths, project_trusted=True)
    assert trusted.auto_enable_codemode is False
    assert any("autoEnableCodemode must be a boolean" in error for error in trusted.errors)


@pytest.mark.tonio
async def test_validates_the_oauth_callback_url_scope_and_client_name(tmp_path):
    paths = _setup(
        tmp_path,
        {
            "mcpServers": {
                "ok": {
                    "url": "https://a.example/mcp",
                    "oauth": {"callbackUrl": "http://localhost:8080/callback", "scope": "a b"},
                },
                "ipv6": {
                    "url": "https://a.example/mcp",
                    "oauth": {"callbackUrl": "http://[::1]/cb", "callbackPort": 9000},
                },
                "same": {
                    "url": "https://a.example/mcp",
                    "oauth": {"callbackUrl": "http://127.0.0.1:2/cb", "callbackPort": 2},
                },
                "remote": {"url": "https://a.example/mcp", "oauth": {"callbackUrl": "https://example.com/callback"}},
                "both": {
                    "url": "https://a.example/mcp",
                    "oauth": {"callbackUrl": "http://127.0.0.1:1/cb", "callbackPort": 2},
                },
                "scope": {"url": "https://a.example/mcp", "oauth": {"scope": ["a"]}},
                "named": {"url": "https://a.example/mcp", "oauth": {"clientName": "Claude Code"}},
                "unnamed": {"url": "https://a.example/mcp", "oauth": {"clientName": " "}},
                "metadata": {
                    "url": "https://a.example/mcp",
                    "oauth": {"authServerMetadataUrl": "https://idp.example/m"},
                },
                "plainMetadata": {
                    "url": "https://a.example/mcp",
                    "oauth": {"authServerMetadataUrl": "http://idp.example/m"},
                },
            }
        },
        {},
    )
    loaded = await load_mcp_config(**paths, project_trusted=False)
    assert [server.name for server in loaded.servers] == ["ok", "ipv6", "same", "named", "metadata"]
    expected = [
        'server "remote": oauth.callbackUrl must be an http URI on localhost',
        'server "both": oauth.callbackUrl and oauth.callbackPort name different ports',
        'server "scope": oauth.scope must be a string',
        'server "unnamed": oauth.clientName must be a non-empty string',
        'server "plainMetadata": oauth.authServerMetadataUrl must be an https URL',
    ]
    assert len(loaded.errors) == len(expected)
    for error, fragment in zip(loaded.errors, expected, strict=True):
        assert fragment in error


@pytest.mark.tonio
async def test_resolves_per_tool_exposure_from_exact_names_then_patterns_in_order(tmp_path):
    paths = _setup(
        tmp_path,
        {
            "mcpServers": {
                "gh": {
                    "command": "x",
                    "exposure": "deferred",
                    "toolExposure": {
                        "get_*": "codemode",
                        "get_me": "direct",
                        "*delete*": "hidden",
                        "get_file.*": "direct",
                    },
                },
                "bad": {"command": "x", "toolExposure": {"a": "visible"}},
            }
        },
        {},
    )
    loaded = await load_mcp_config(**paths, project_trusted=False)
    assert len(loaded.errors) == 1
    assert 'server "bad": toolExposure "a" must be one of' in loaded.errors[0]
    config = loaded.servers[0].config
    assert get_mcp_tool_exposure(config, "get_me") == "direct"
    assert get_mcp_tool_exposure(config, "get_issue") == "codemode"
    assert get_mcp_tool_exposure(config, "get_delete_hint") == "codemode"
    assert get_mcp_tool_exposure(config, "delete_repo") == "hidden"
    assert get_mcp_tool_exposure(config, "list_issues") == "deferred"
    # Only `*` is special.
    assert get_mcp_tool_exposure({"command": "x", "toolExposure": {"get_file.*": "direct"}}, "get_file_x") == "codemode"


@pytest.mark.tonio
async def test_validates_provider_auth_and_accepts_it_only_in_the_global_mcp_json(tmp_path):
    paths = _setup(
        tmp_path,
        {
            "mcpServers": {
                "radius": {"url": "https://radius.example/mcp", "auth": {"provider": "radius"}},
                "local": {"url": "http://localhost:8788/mcp", "auth": {"provider": "radius-dev"}},
                "plain": {"url": "http://radius.example/mcp", "auth": {"provider": "radius"}},
                "empty": {"url": "https://radius.example/mcp", "auth": {"provider": ""}},
            }
        },
        {"mcpServers": {"radius": {"url": "https://evil.example/mcp", "auth": {"provider": "radius"}}}},
    )
    loaded = await load_mcp_config(**paths, project_trusted=True)
    # The project entry cannot replace the global one: it would send the credential to its own URL.
    assert [(server.name, server.scope, server.config.get("url")) for server in loaded.servers] == [
        ("radius", "global", "https://radius.example/mcp"),
        ("local", "global", "http://localhost:8788/mcp"),
    ]
    expected = [
        'server "plain": auth requires an https URL',
        'server "empty": auth.provider must be a provider name',
        'server "radius": auth is only allowed in the global mcp.json',
    ]
    assert len(loaded.errors) == len(expected)
    for error, fragment in zip(loaded.errors, expected, strict=True):
        assert fragment in error


# MCP tools


def test_creates_provider_safe_tool_names():
    assert create_mcp_tool_name("docs", "search") == "mcp__docs__search"
    assert create_mcp_tool_name("my-server", "get.item/v2") == "mcp__my_server__get_item_v2"
    long = create_mcp_tool_name("server", "x" * 100)
    assert len(long) == 64
    assert re.fullmatch(r"mcp__server__x+_[0-9a-f]{8}", long)
    assert create_mcp_tool_name("server", f"{'x' * 100}y") != long
    # Names that sanitize to one already taken by another tool get a hash suffix.
    taken = create_mcp_tool_name("s", "a_b")
    second = create_mcp_tool_name("s", "a-b", lambda name: name == taken)
    assert re.fullmatch(r"mcp__s__a_b_[0-9a-f]{8}", second)


@pytest.mark.tonio
async def test_converts_results_passing_the_call_tool_result_to_scripts_and_flagging_errors():
    blocks = [
        {"type": "resource_link", "uri": "file:///a", "name": "a"},
        {"type": "resource", "resource": {"uri": "file:///b", "text": "b text"}},
        {"type": "audio", "data": "", "mimeType": "audio/wav"},
    ]
    assert await convert_mcp_result(
        "docs", "t", {"content": blocks, "structuredContent": {"ok": True}, "_meta": {"trace": "x"}}
    ) == AgentToolResult(
        content=[
            TextContent(text='[Resource file:///a "a"]'),
            TextContent(text="b text"),
            TextContent(text="[audio audio/wav omitted]"),
        ],
        details={"server": "docs", "tool": "t"},
        # Scripts get the server's blocks as sent, without `_meta`.
        structured_content={"content": blocks, "structuredContent": {"ok": True}},
    )
    assert (await convert_mcp_result("docs", "t", {"content": [], "structuredContent": {"n": 1}})).content == [
        TextContent(text='{\n  "n": 1\n}')
    ]
    assert await convert_mcp_result(
        "docs", "t", {"content": [{"type": "text", "text": "nope"}], "isError": True}
    ) == AgentToolResult(
        content=[TextContent(text="nope")],
        details={"server": "docs", "tool": "t"},
        structured_content={"content": [{"type": "text", "text": "nope"}], "isError": True},
        is_error=True,
    )
    assert (await convert_mcp_result("docs", "t", {"content": [], "isError": True})).content == [
        TextContent(text="MCP tool docs/t returned an error")
    ]


@pytest.mark.tonio
async def test_points_resource_links_to_read_mcp_resource_and_saves_binary_resources():
    saved: list[tuple[str | bytes, str]] = []

    async def save_output(data, extension):
        saved.append((data, extension))
        return f"/tmp/saved{extension}"

    converted = await convert_mcp_result(
        "docs",
        "t",
        {
            "content": [
                {
                    "type": "resource_link",
                    "uri": "docs://guide",
                    "name": "guide",
                    "title": "The Guide",
                    "mimeType": "text/markdown",
                    "size": 2048,
                    "description": "How to use it",
                },
                {
                    "type": "resource",
                    "resource": {"uri": "file:///r/report.pdf", "mimeType": "application/pdf", "blob": "JVBERg=="},
                },
                {"type": "resource", "resource": {"uri": "docs://logo", "mimeType": "image/png", "blob": "AAAA"}},
            ]
        },
        ConvertMcpResultOptions(save_output=save_output, readable_resources=True),
    )
    assert converted.content == [
        TextContent(
            text='[Resource docs://guide "The Guide" (text/markdown, 2.0KB): How to use it. '
            'Read it with read_mcp_resource (server "docs")]'
        ),
        TextContent(text="[Binary resource file:///r/report.pdf (application/pdf, 4B) saved to /tmp/saved.pdf]"),
        ImageContent(data="AAAA", mime_type="image/png"),
    ]
    assert saved == [(b"%PDF", ".pdf")]


@pytest.mark.tonio
async def test_cuts_the_middle_of_model_facing_text_over_20kb_and_keeps_the_full_result_for_scripts():
    saved: list[str | bytes] = []

    async def save_output(data, _extension):
        saved.append(data)
        return "/tmp/full.txt"

    full = "\n".join(f"line {index + 1}" for index in range(3000))
    image = {"type": "image", "data": "AAAA", "mimeType": "image/png"}
    result = {"content": [{"type": "text", "text": full}, image]}
    converted = await convert_mcp_result("docs", "snapshot", result, ConvertMcpResultOptions(save_output=save_output))
    assert len(converted.content) == 2
    text = converted.content[0].text
    # Codex's format: a header, the start and end of the text, then the file with the full text.
    tokens = -(-len(full) // 4)
    assert re.match(
        rf"Warning: truncated output \(original token count: {tokens}\)\nTotal output lines: 3000\n\nline 1\nline 2\n",
        text,
    )
    assert re.search(r"…\d+ chars truncated…", text)
    assert text.endswith("line 3000\n\n[Full output: /tmp/full.txt (read it with offset/limit)]")
    assert len(text.encode()) < 21 * 1024
    assert converted.content[1] == ImageContent(data="AAAA", mime_type="image/png")
    assert converted.details == {"server": "docs", "tool": "snapshot", "fullOutputPath": "/tmp/full.txt"}
    assert saved == [full]
    assert converted.structured_content == result

    # Text within the limit is not saved.
    await convert_mcp_result(
        "docs", "small", {"content": [{"type": "text", "text": "ok"}]}, ConvertMcpResultOptions(save_output=save_output)
    )
    assert len(saved) == 1


def test_cuts_multi_byte_text_only_at_character_boundaries():
    text = f"{'é' * 20_000}end"
    result = truncate_middle(text, 1001)
    assert result.truncated
    assert "�" not in result.content
    assert result.content.endswith("end")
    head, tail = re.split(r"…\d+ chars truncated…", result.content)
    assert len(head.encode()) <= 500
    assert len(tail.encode()) <= 501
    assert len(head) + len(tail) + result.removed_chars == len(text)


# MCP connections


class _FakeServers:
    """In-memory servers that answer initialize, tools/list, and tools/call with "ok"."""

    def __init__(self) -> None:
        self.servers: list = []

    async def create(self, *, methods: list[str] | None = None, no_tools: bool = False):
        client, server = create_in_memory_transport_pair()
        self.servers.append(server)

        async def on_message(message):
            if "id" not in message or "method" not in message:
                return
            method = message["method"]
            if methods is not None:
                methods.append(method)
            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "protocolVersion": LATEST_PROTOCOL_VERSION,
                        "capabilities": {"prompts": {}} if no_tools else {"tools": {}},
                        "serverInfo": {"name": "fake", "version": "1.0.0"},
                    },
                }
            elif method == "tools/list":
                response = (
                    {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "Method not found"}}
                    if no_tools
                    else {"jsonrpc": "2.0", "id": message["id"], "result": {"tools": []}}
                )
            elif method == "resources/read":
                response = {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"contents": [{"uri": "docs://a", "text": "ok"}]},
                }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"content": [{"type": "text", "text": "ok"}]},
                }
            server.send(response)

        server.on_message(on_message)
        await server.start()
        return client


def _expire_tool_calls(transport) -> None:
    """Simulates the HTTP transport's 404 for a session the server no longer knows."""
    send = transport.send

    def expiring(message):
        if message.get("method") == "tools/call":
            return SendResult.failed(McpSessionExpiredError("gone"))
        return send(message)

    transport.send = expiring


class _Connected:
    def __init__(self, entry, transports, log=None, on_change=None) -> None:
        self.opened = 0

        async def create_transport(_entry, _cwd, _auth_provider):
            factory = transports[self.opened]
            self.opened += 1
            return await factory()

        self.connection = McpServerConnection(
            entry=entry,
            cwd=os.getcwd(),
            create_transport=create_transport,
            credentials=McpOAuthCredentialStore(InMemoryAuthStorageBackend()),
            log=log,
            on_tools=lambda _connection: None,
            on_change=on_change,
        )


def _stdio_entry(name: str = "fake") -> McpServerEntry:
    return McpServerEntry(name=name, config={"command": "unused"}, source="test")


@pytest.fixture
def fake_servers():
    return _FakeServers()


@pytest.fixture
def short_retry_delays(monkeypatch):
    monkeypatch.setattr(runtime, "_CONNECT_RETRY_DELAYS_MS", (1, 1))


@pytest.mark.tonio
async def test_starts_a_new_session_and_retries_once_when_the_session_expired(fake_servers):
    async def expiring():
        transport = await fake_servers.create()
        _expire_tool_calls(transport)
        return transport

    connected = _Connected(_stdio_entry(), [expiring, fake_servers.create])
    connection = connected.connection
    calls = [tonio.spawn(connection.call_tool("echo", {})), tonio.spawn(connection.call_tool("echo", {}))]
    results = [await call for call in calls]
    assert results == [{"content": [{"type": "text", "text": "ok"}]}] * 2
    assert connected.opened == 2
    await connection.close()


@pytest.mark.tonio
async def test_closes_the_client_of_an_expired_session_with_the_connection(fake_servers):
    """pidrei-only: the client of a session the server forgot is left open
    for its calls in flight, and pi never closes it. Here the connection
    keeps it and closes it with its other clients."""
    closed = tonio.Event()

    async def expiring():
        transport = await fake_servers.create()
        _expire_tool_calls(transport)

        async def on_close() -> None:
            closed.set()

        transport.on_close(on_close)
        return transport

    connected = _Connected(_stdio_entry(), [expiring, fake_servers.create])
    connection = connected.connection
    assert await connection.call_tool("echo", {}) == {"content": [{"type": "text", "text": "ok"}]}
    assert connected.opened == 2
    assert not closed.is_set()

    await connection.close()

    await closed.wait(_WAIT_S)
    assert closed.is_set()


@pytest.mark.tonio
async def test_a_tool_refresh_failing_after_the_close_changes_nothing(fake_servers, monkeypatch):
    """pidrei-only: a refresh whose client was closed or replaced while it
    listed the tools fails for that reason; its error is not the
    connection's."""
    listing = tonio.Event()
    refreshed = tonio.Event()
    changes: list[tuple[str, str | None]] = []
    refresh_tools = McpServerConnection._refresh_tools

    async def refresh_then_report(self, client) -> None:
        await refresh_tools(self, client)
        refreshed.set()

    async def holding():
        transport = await fake_servers.create()
        send = transport.send
        lists: list[None] = []

        def held(message):
            if message.get("method") == "tools/list":
                lists.append(None)
                if len(lists) > 1:
                    # The refresh's list is never answered.
                    listing.set()
                    return SendResult.succeeded()
            return send(message)

        transport.send = held
        return transport

    async def on_change(connection) -> None:
        changes.append((connection.state, connection.error))

    monkeypatch.setattr(McpServerConnection, "_refresh_tools", refresh_then_report)
    connection = _Connected(_stdio_entry(), [holding], on_change=on_change).connection
    await connection.get_client()
    fake_servers.servers[0].send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    await listing.wait(_WAIT_S)
    assert listing.is_set()

    await connection.close()

    await refreshed.wait(_WAIT_S)
    assert refreshed.is_set()
    assert connection.error is None
    assert changes[-1] == ("closed", None)


@pytest.mark.tonio
async def test_a_cancelled_call_that_needed_a_sign_in_still_marks_the_server(fake_servers):
    """pidrei-only: a call run by a codemode script is cancelled with the
    script's scope. One that had found the server asking for a sign-in
    still leaves the connection marked, and its listeners told."""
    closing = tonio.Event()
    gate = tonio.Event()
    marked = tonio.Event()
    transports: list = []

    async def asking_for_sign_in():
        transport = await fake_servers.create()
        send, close = transport.send, transport.close

        def rejecting(message):
            if message.get("method") == "tools/call":
                return SendResult.failed(McpOAuthAuthorizationRequiredError())
            return send(message)

        async def parked_close() -> None:
            closing.set()
            await gate.wait(_WAIT_S)
            await close()

        transport.send = rejecting
        transport.close = parked_close
        transports.append(close)
        return transport

    async def on_change(connection) -> None:
        if connection.state == "needs-auth":
            marked.set()

    connection = _Connected(_stdio_entry(), [asking_for_sign_in], on_change=on_change).connection
    await connection.get_client()

    async with tonio.scope(cancel_on_exc=True) as scope:
        scope.spawn(connection.call_tool("echo", {}))
        # The call is closing the client when its scope is cancelled.
        await closing.wait(_WAIT_S)
        assert closing.is_set()
        scope.cancel()
    gate.set()

    await marked.wait(_WAIT_S)
    assert marked.is_set()
    assert connection.state == "needs-auth"
    # The fake's close was cut with the call: close the transport for real.
    await transports[0]()
    await connection.close()


def _parked_close(transport, closing: tonio.Event, gate: tonio.Event) -> None:
    """Makes the transport's close wait for `gate`, as a close still in flight."""
    close = transport.close

    async def parked() -> None:
        closing.set()
        await gate.wait(_WAIT_S)
        await close()

    transport.close = parked


@pytest.mark.tonio
async def test_a_connect_made_while_a_call_drops_its_client_for_a_sign_in_is_not_marked_over(fake_servers):
    """pidrei-only: the client is dropped and the server marked as needing a
    sign-in in one step, before the client is closed. A connect another call
    makes during that close is the newer state, and stays."""
    closing = tonio.Event()
    gate = tonio.Event()

    async def asking_for_sign_in():
        transport = await fake_servers.create()
        send = transport.send

        def rejecting(message):
            if message.get("method") == "tools/call":
                return SendResult.failed(McpOAuthAuthorizationRequiredError())
            return send(message)

        transport.send = rejecting
        _parked_close(transport, closing, gate)
        return transport

    connection = _Connected(_stdio_entry(), [asking_for_sign_in, fake_servers.create]).connection
    await connection.get_client()

    async def rejected_call() -> None:
        with pytest.raises(Exception, match="requires sign-in"):
            await connection.call_tool("echo", {})

    try:
        async with tonio.scope(cancel_on_exc=True) as scope:
            scope.spawn(rejected_call())
            await closing.wait(_WAIT_S)
            assert closing.is_set()
            assert connection.state == "needs-auth"
            # Another call connects while the old client is still closing.
            await connection.get_client()
            assert connection.state == "connected"
            gate.set()
        assert connection.state == "connected"
    finally:
        gate.set()
        await connection.close()


@pytest.mark.tonio
async def test_a_connect_made_while_a_sign_out_closes_its_client_is_not_marked_over(fake_servers):
    """pidrei-only: as above, for a sign-out."""
    closing = tonio.Event()
    gate = tonio.Event()

    async def signed_in():
        transport = await fake_servers.create()
        _parked_close(transport, closing, gate)
        return transport

    connection = _Connected(_stdio_entry(), [signed_in, fake_servers.create]).connection
    await connection.get_client()
    try:
        async with tonio.scope(cancel_on_exc=True) as scope:
            scope.spawn(connection.sign_out())
            await closing.wait(_WAIT_S)
            assert closing.is_set()
            assert connection.state == "needs-auth"
            await connection.get_client()
            assert connection.state == "connected"
            gate.set()
        assert connection.state == "connected"
    finally:
        gate.set()
        await connection.close()


@pytest.mark.tonio
async def test_a_failing_change_listener_fails_the_call_that_needed_a_sign_in_with_its_own_error(fake_servers):
    """pidrei-only: marking the server runs on its own coroutine, whose
    failure reaches the call as itself, not as the spawn's exception group."""

    async def asking_for_sign_in():
        transport = await fake_servers.create()
        send = transport.send

        def rejecting(message):
            if message.get("method") == "tools/call":
                return SendResult.failed(McpOAuthAuthorizationRequiredError())
            return send(message)

        transport.send = rejecting
        return transport

    async def on_change(connection) -> None:
        if connection.state == "needs-auth":
            raise RuntimeError("listener failed")

    connection = _Connected(_stdio_entry(), [asking_for_sign_in], on_change=on_change).connection
    try:
        with pytest.raises(RuntimeError, match="^listener failed$"):
            await connection.call_tool("echo", {})
    finally:
        await connection.close()


@pytest.mark.tonio
async def test_close_returns_once_the_connect_in_flight_has_ended():
    """pidrei-only: pi leaves a connect in flight running after the close.
    Here the close ends it and waits for it, so nothing of the connection
    runs once `close()` has returned."""
    creating = tonio.Event()
    release = tonio.Event()
    states: list[str] = []

    async def held():
        creating.set()
        await release.wait(_WAIT_S)
        raise RuntimeError("no transport")

    async def on_change(connection) -> None:
        states.append(connection.state)
        if connection.state == "closed":
            # The close has started: the transport factory may answer now.
            release.set()

    connection = _Connected(_stdio_entry(), [held], on_change=on_change).connection

    async def connect() -> None:
        try:
            await connection.get_client()
        except Exception:
            # Fails as closed, which is not what the test is about.
            pass

    async with tonio.scope(cancel_on_exc=True) as scope:
        scope.spawn(connect())
        await creating.wait(_WAIT_S)
        assert creating.is_set()
        await connection.close()
        # The connect's own last change (it failed as closed) came before the close returned.
        assert states == ["connecting", "closed", "closed"]


@pytest.mark.tonio
async def test_expands_home_in_the_command_arguments_and_cwd_of_stdio_servers(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "work").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    # Answers every tool call with its working directory.
    (home / "server.py").write_text(
        "import json, os, sys\n"
        "for line in sys.stdin:\n"
        "    message = json.loads(line)\n"
        "    if 'id' not in message:\n"
        "        continue\n"
        "    if message['method'] == 'initialize':\n"
        "        result = {'protocolVersion': '2025-06-18', 'capabilities': {'tools': {}},"
        " 'serverInfo': {'name': 'cwd', 'version': '1'}}\n"
        "    elif message['method'] == 'tools/list':\n"
        "        result = {'tools': [{'name': 'cwd', 'inputSchema': {'type': 'object'}}]}\n"
        "    else:\n"
        "        result = {'content': [{'type': 'text', 'text': os.getcwd()}]}\n"
        "    print(json.dumps({'jsonrpc': '2.0', 'id': message['id'], 'result': result}), flush=True)\n"
    )
    connection = McpServerConnection(
        entry=McpServerEntry(
            name="home", config={"command": sys.executable, "args": ["~/server.py"], "cwd": "~/work"}, source="test"
        ),
        cwd=str(tmp_path),
        create_transport=create_default_transport,
        credentials=McpOAuthCredentialStore(InMemoryAuthStorageBackend()),
        on_tools=lambda _connection: None,
    )
    try:
        result = await connection.call_tool("cwd", {})
        assert os.path.realpath(result["content"][0]["text"]) == os.path.realpath(home / "work")
    finally:
        await connection.close()


@pytest.mark.tonio
async def test_connects_to_servers_without_the_tools_capability_without_listing_tools(fake_servers):
    methods: list[str] = []

    async def no_tools():
        return await fake_servers.create(no_tools=True, methods=methods)

    connection = _Connected(_stdio_entry(), [no_tools]).connection
    await connection.get_client()
    assert connection.state == "connected"
    assert connection.tools == []
    assert methods == ["initialize"]
    await connection.close()


@pytest.mark.tonio
async def test_marks_a_dropped_connection_and_reconnects_on_the_next_call(fake_servers):
    disconnected = tonio.Event()

    async def on_change(connection):
        if connection.state == "disconnected":
            disconnected.set()

    connected = _Connected(_stdio_entry(), [fake_servers.create, fake_servers.create], on_change=on_change)
    connection = connected.connection
    await connection.get_client()
    await fake_servers.servers[-1].close()
    await disconnected.wait(_WAIT_S)
    assert connection.state == "disconnected"
    assert connection.error == "Connection closed"
    assert await connection.call_tool("echo", {}) == {"content": [{"type": "text", "text": "ok"}]}
    assert connection.state == "connected"
    assert connected.opened == 2
    await connection.close()


def _http_entry(**config) -> McpServerEntry:
    return McpServerEntry(name="fake", config={"url": "http://unused.invalid", **config}, source="test")


def _failing(fake_servers, error):
    async def create():
        transport = await fake_servers.create()
        transport.send = lambda _message: SendResult.failed(error)
        return transport

    return create


@pytest.mark.tonio
async def test_retries_http_connections_that_fail_with_a_transient_error(fake_servers, short_retry_delays):
    connected = _Connected(
        _http_entry(headers={"Authorization": "x"}),
        [_failing(fake_servers, McpHttpError(503, "MCP HTTP request failed with status 503")), fake_servers.create],
    )
    await connected.connection.get_client()
    assert connected.connection.state == "connected"
    assert connected.opened == 2
    await connected.connection.close()

    failing = _Connected(
        _http_entry(headers={"Authorization": "x"}),
        [_failing(fake_servers, McpHttpError(400, "MCP HTTP request failed with status 400: bad"))],
    )
    with pytest.raises(Exception, match="status 400: bad"):
        await failing.connection.get_client()
    assert failing.connection.state == "failed"
    assert failing.opened == 1


@pytest.mark.tonio
async def test_retries_resource_reads_but_not_tool_calls_after_a_transient_http_error(fake_servers, short_retry_delays):
    failed: set[str] = set()

    async def create():
        transport = await fake_servers.create()
        send = transport.send

        # The first read and the first call fail with 502.
        def failing_once(message):
            method = message.get("method", "")
            if method in ("resources/read", "tools/call") and method not in failed:
                failed.add(method)
                return SendResult.failed(McpHttpError(502, "MCP HTTP request failed with status 502"))
            return send(message)

        transport.send = failing_once
        return transport

    connection = _Connected(_stdio_entry(), [create]).connection
    assert await connection.read_resource("docs://a") == {"contents": [{"uri": "docs://a", "text": "ok"}]}
    with pytest.raises(McpHttpError, match="status 502"):
        await connection.call_tool("echo", {})
    await connection.close()


class _Response:
    """pi's `new Response(null, { status: 401 })`."""

    status = 401

    @property
    def headers(self) -> dict[str, str]:
        return {}


@pytest.mark.tonio
async def test_asks_oauth_servers_that_keep_rejecting_requests_for_a_new_sign_in(fake_servers):
    connection = _Connected(_http_entry(), [_failing(fake_servers, McpAuthRequiredError(_Response()))]).connection
    with pytest.raises(Exception, match=re.escape('MCP server "fake" requires sign-in. Run /mcp to sign in.')):
        await connection.get_client()
    assert connection.state == "needs-auth"
    await connection.close()


@pytest.mark.tonio
async def test_sends_the_provider_token_and_asks_for_the_provider_login_when_the_server_rejects_it(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        authorizations: list[str | None] = []

        async def handle(request, _origin):
            value = request.headers.get("authorization")
            authorizations.append(None if value is None else bytes(value).decode("latin-1"))
            await request.read()
            await request.respond(401)

        origin = await http_servers.listen(handle)

        async def provider_token(provider: str) -> str | None:
            return "tok" if provider == "radius" else None

        connection = McpServerConnection(
            entry=McpServerEntry(
                name="radius", config={"url": f"{origin}/mcp", "auth": {"provider": "radius"}}, source="test"
            ),
            cwd=os.getcwd(),
            create_transport=create_default_transport,
            credentials=McpOAuthCredentialStore(InMemoryAuthStorageBackend()),
            provider_token=provider_token,
            on_tools=lambda _connection: None,
        )
        try:
            assert connection.oauth_url is None
            with pytest.raises(
                Exception, match=re.escape('MCP server "radius" requires sign-in. Run /login radius to sign in.')
            ):
                await connection.get_client()
            assert connection.state == "needs-auth"
            assert authorizations == ["Bearer tok"]
        finally:
            await connection.close()


class _RecordingLog(McpServerLog):
    """Sets `written` once `count` messages were appended."""

    def __init__(self, path: str, count: int) -> None:
        super().__init__(path)
        self._count = count
        self.written = tonio.Event()

    async def write(self, server, params):
        await super().write(server, params)
        self._count -= 1
        if self._count == 0:
            self.written.set()


@pytest.mark.tonio
async def test_appends_server_log_messages_to_the_log_file(fake_servers, tmp_path):
    path = tmp_path / "mcp.log"
    log = _RecordingLog(str(path), 2)
    connection = _Connected(_stdio_entry(), [fake_servers.create], log=log).connection
    await connection.get_client()
    server = fake_servers.servers[-1]
    server.send(
        {
            "jsonrpc": "2.0",
            "method": "notifications/message",
            "params": {"level": "warning", "logger": "db", "data": "slow\nquery"},
        }
    )
    server.send(
        {"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "error", "data": {"code": 7}}}
    )
    await log.written.wait(_WAIT_S)
    lines = re.sub(r"^\S+ ", "", path.read_text(), flags=re.MULTILINE)
    assert lines == '[fake] warning db: slow\n    query\n[fake] error {"code":7}\n'
    await connection.close()


@pytest.mark.tonio
async def test_closing_ends_a_connect_in_flight(fake_servers):
    # PiDrei-only: pi leaves the connect running until its request times out.
    initializing = tonio.Event()

    async def silent():
        client, server = create_in_memory_transport_pair()

        async def on_message(_message):
            initializing.set()

        server.on_message(on_message)
        await server.start()
        return client

    # A short request timeout bounds the test if the close did not end the connect.
    entry = McpServerEntry(name="fake", config={"command": "unused", "timeout": 2}, source="test")
    connection = _Connected(entry, [silent]).connection
    opening = tonio.spawn(connection.get_client())
    await initializing.wait(_WAIT_S)
    assert initializing.is_set()
    await connection.close()
    with pytest.raises(ExceptionGroup) as raised:
        await opening
    (error,) = raised.value.exceptions
    assert str(error) == 'MCP server "fake" failed to connect: MCP connection closed'
    assert connection.state == "closed"


@pytest.mark.tonio
async def test_resolves_the_oauth_client_secret_lazily(fake_servers):
    connection = _Connected(_http_entry(oauth={"clientSecret": "!exit 1"}), [fake_servers.create]).connection
    assert await connection.call_tool("echo", {}) == {"content": [{"type": "text", "text": "ok"}]}
    with pytest.raises(Exception, match="oauth.clientSecret"):
        await connection.oauth_settings()
    await connection.close()


# MCP servers section


def _listed(name: str, description: str | None = None, exposure: str | None = None) -> McpServerListing:
    config: dict = {"command": "x"}
    if description:
        config["description"] = description
    if exposure:
        config["exposure"] = exposure
    return McpServerListing(entry=McpServerEntry(name=name, config=config, source="test"))


def test_lists_servers_with_how_their_tools_are_reached_and_the_first_line_of_their_description():
    section = render_servers_section(
        [
            _listed("docs", "Docs search.\nMore."),
            _listed("later", None, "deferred"),
            _listed("direct", "Declared.", "direct"),
            McpServerListing(
                entry=_listed("plain").entry, connection=SimpleNamespace(instructions="From instructions.")
            ),
        ]
    )
    assert section.split("\n")[1:] == [
        "- mcp__docs (codemode): Docs search.",
        "- mcp__later (tool_search)",
        "- mcp__plain (codemode): From instructions.",
    ]
    assert render_servers_section([_listed("direct", "Declared.", "direct")]) is None


def test_shortens_descriptions_to_fit_the_size_limit():
    section = render_servers_section([_listed(f"server{index}", "x" * 400) for index in range(40)]) or ""
    assert len(section) <= MAX_SERVERS_SECTION_CHARS
    assert len(section.split("\n")) == 41
    assert "- mcp__server39 (codemode): x" in section


def test_leaves_out_the_last_servers_when_their_names_alone_do_not_fit():
    section = (
        render_servers_section([_listed(f"server-with-a-long-name-{index}", "desc") for index in range(200)]) or ""
    )
    assert len(section) <= MAX_SERVERS_SECTION_CHARS
    lines = section.split("\n")
    assert re.fullmatch(r"- … \d+ more servers; find their tools with search_tools\(\)", lines[-1])
    omitted = int(re.search(r"(\d+) more", lines[-1]).group(1))
    assert len(lines) - 2 + omitted == 200
