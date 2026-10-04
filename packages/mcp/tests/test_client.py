"""Mirror of pi mcp test/client.test.ts.

Delivery is real parallelism here: a message one side sends reaches the
other on its own delivery coroutine, after `send` returned. Where pi asserts
right after a send (one microtask later), these tests wait for the server
to have received the message, through an Event its listener sets.
"""

import threading
from typing import Any

import pytest
import tonio.colored as tonio
from fake_timers import fake_timers

from pidrei_mcp import (
    LATEST_PROTOCOL_VERSION,
    McpAbortError,
    McpClient,
    McpConnectionClosedError,
    McpError,
    McpTimeoutError,
)
from pidrei_mcp.testing import InMemoryTransport, create_in_memory_transport_pair
from pidrei_utils.cancel import AbortError, CancelToken


_WAIT_S = 5.0


class FakeServer:
    def __init__(self, transport: InMemoryTransport) -> None:
        self.transport = transport
        self.messages: list[dict[str, Any]] = []
        self.handlers: dict[str, Any] = {}
        self.closed = tonio.Event()
        self._lock = threading.Lock()
        self._arrived = tonio.Event()

    def set_handler(self, method: str, handler) -> None:
        self.handlers[method] = handler

    async def on_message(self, message: dict[str, Any]) -> None:
        with self._lock:
            self.messages.append(message)
        self._arrived.set()
        if "id" not in message or "method" not in message:
            return
        tonio.spawn.without_tracking(self._answer(message, self.handlers.get(message["method"])))

    async def on_close(self) -> None:
        self.closed.set()

    async def _answer(self, request: dict[str, Any], handler) -> None:
        try:
            if handler is None:
                raise McpError(-32601, f"Method not found: {request['method']}")
            await self.transport.send({"jsonrpc": "2.0", "id": request["id"], "result": await handler(request)})
        except Exception as error:
            mcp_error = error if isinstance(error, McpError) else McpError(-32603, str(error))
            response_error = {"code": mcp_error.code, "message": mcp_error.message}
            if mcp_error.data is not None:
                response_error["data"] = mcp_error.data
            await self.transport.send({"jsonrpc": "2.0", "id": request["id"], "error": response_error})

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.messages)

    async def wait_for(self, predicate) -> dict[str, Any]:
        """The first message received that matches `predicate`."""
        deadline = tonio.time.time() + _WAIT_S
        while True:
            self._arrived.clear()
            for message in self.snapshot():
                if predicate(message):
                    return message
            remaining = deadline - tonio.time.time()
            if remaining <= 0:
                raise AssertionError(f"server never received the message; got {self.snapshot()}")
            await self._arrived.wait(remaining)


async def create_server() -> tuple[InMemoryTransport, FakeServer]:
    client_transport, server_transport = create_in_memory_transport_pair()
    server = FakeServer(server_transport)
    server_transport.on_message(server.on_message)
    server_transport.on_close(server.on_close)
    await server_transport.start()

    async def initialize(_request):
        return {
            "protocolVersion": LATEST_PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": True}},
            "serverInfo": {"name": "test-server", "version": "1.0.0"},
            "instructions": "Use test tools.",
        }

    server.set_handler("initialize", initialize)
    return client_transport, server


async def connect() -> tuple[McpClient, FakeServer]:
    client_transport, server = await create_server()
    client = McpClient(name="test-client", version="2.0.0")
    await client.connect(client_transport)
    return client, server


def is_method(method: str):
    return lambda message: message.get("method") == method


def _hang(release: tonio.Event):
    async def handler(_request):
        await release.wait()
        return {"content": []}

    return handler


@pytest.mark.tonio
async def test_initializes_the_connection_before_exposing_server_information():
    client, server = await connect()
    assert client.connection_state == "connected"
    assert client.protocol_version == LATEST_PROTOCOL_VERSION
    assert client.server_info == {"name": "test-server", "version": "1.0.0"}
    assert client.server_capabilities == {"tools": {"listChanged": True}}
    assert client.instructions == "Use test tools."
    await server.wait_for(is_method("notifications/initialized"))
    assert server.snapshot() == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "2.0.0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    await client.close()


@pytest.mark.tonio
async def test_paginates_tools_and_preserves_protocol_tool_definitions():
    client, server = await connect()

    async def list_tools(request):
        cursor = (request.get("params") or {}).get("cursor")
        if cursor is None:
            return {
                "tools": [{"name": "search", "description": "Search", "inputSchema": {"type": "object"}}],
                "nextCursor": "page-2",
            }
        return {
            "tools": [
                {
                    "name": "read",
                    "inputSchema": {"type": "object"},
                    "outputSchema": {"type": "object"},
                    "annotations": {"readOnlyHint": True},
                }
            ],
            # Some servers end pagination with an empty cursor instead of omitting it.
            "nextCursor": "",
        }

    server.set_handler("tools/list", list_tools)
    assert await client.list_tools() == [
        {"name": "search", "description": "Search", "inputSchema": {"type": "object"}},
        {
            "name": "read",
            "inputSchema": {"type": "object"},
            "outputSchema": {"type": "object"},
            "annotations": {"readOnlyHint": True},
        },
    ]
    await client.close()


@pytest.mark.tonio
async def test_lists_and_reads_resources():
    client, server = await connect()

    async def list_resources(request):
        if (request.get("params") or {}).get("cursor") is None:
            return {"resources": [{"uri": "file:///a", "name": "a", "mimeType": "text/plain"}], "nextCursor": "2"}
        return {"resources": [{"uri": "file:///b"}]}

    async def list_templates(_request):
        return {"resourceTemplates": [{"uriTemplate": "repo://{owner}/{repo}", "name": "repo"}]}

    async def read(request):
        return {"contents": [{"uri": request["params"]["uri"], "text": "hello"}]}

    server.set_handler("resources/list", list_resources)
    server.set_handler("resources/templates/list", list_templates)
    server.set_handler("resources/read", read)
    # A missing name falls back to the URI.
    assert await client.list_resources() == [
        {"uri": "file:///a", "name": "a", "mimeType": "text/plain"},
        {"uri": "file:///b", "name": "file:///b"},
    ]
    assert await client.list_resource_templates() == [{"uriTemplate": "repo://{owner}/{repo}", "name": "repo"}]
    # Single pages pass the cursor through.
    assert await client.list_resources_page() == {
        "resources": [{"uri": "file:///a", "name": "a", "mimeType": "text/plain"}],
        "nextCursor": "2",
    }
    assert await client.list_resources_page("2") == {"resources": [{"uri": "file:///b", "name": "file:///b"}]}
    assert await client.read_resource("file:///a") == {"contents": [{"uri": "file:///a", "text": "hello"}]}

    async def read_broken(_request):
        return {"contents": [{"uri": "file:///a"}]}

    async def list_broken(_request):
        return {"resources": [{"name": "no uri"}]}

    server.set_handler("resources/read", read_broken)
    with pytest.raises(McpError, match="Invalid contents in MCP resources/read result"):
        await client.read_resource("file:///a")
    server.set_handler("resources/list", list_broken)
    with pytest.raises(McpError, match="Invalid entry in MCP resources/list result"):
        await client.list_resources()
    await client.close()


@pytest.mark.tonio
async def test_returns_structured_tool_content_and_surfaces_json_rpc_errors():
    client, server = await connect()

    async def call(request):
        params = request["params"]
        if params["name"] == "fail":
            raise McpError(1234, "tool failed", {"retryable": False})
        return {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"count": params["arguments"]["count"]},
        }

    server.set_handler("tools/call", call)
    assert await client.call_tool("count", {"count": 3}) == {
        "content": [{"type": "text", "text": "ok"}],
        "structuredContent": {"count": 3},
    }
    with pytest.raises(McpError) as caught:
        await client.call_tool("fail")
    assert (caught.value.code, caught.value.message, caught.value.data) == (1234, "tool failed", {"retryable": False})
    await client.close()


@pytest.mark.tonio
async def test_renews_the_timeout_on_progress():
    with fake_timers() as fake:
        client, server = await connect()
        release = tonio.Event()
        server.set_handler("tools/call", _hang(release))
        progress = []
        progressed = tonio.Event()

        async def on_progress(notification):
            progress.append(notification)
            progressed.set()

        call = tonio.spawn(client.call_tool("slow", {}, timeout_ms=50, on_progress=on_progress))
        request = await server.wait_for(is_method("tools/call"))
        token = request["params"]["_meta"]["progressToken"]
        fake.advance(40)
        await server.transport.send(
            {
                "jsonrpc": "2.0",
                "method": "notifications/progress",
                "params": {"progressToken": token, "progress": 1, "total": 2},
            }
        )
        await progressed.wait(_WAIT_S)
        # 80 ms since the request, 40 since the progress renewed its 50 ms timeout.
        fake.advance(40)
        release.set()
        assert await call == {"content": []}
        assert progress == [{"progressToken": 2, "progress": 1, "total": 2}]
        await client.close()


@pytest.mark.tonio
async def test_cancels_aborted_and_timed_out_requests():
    client, server = await connect()
    release = tonio.Event()
    server.set_handler("tools/call", _hang(release))
    try:
        cancel = CancelToken()

        async def abort_once_received():
            await server.wait_for(is_method("tools/call"))
            cancel.cancel(AbortError("stop"))

        tonio.spawn.without_tracking(abort_once_received())
        with pytest.raises(McpAbortError):
            await client.call_tool("wait", {}, cancel=cancel)
        assert await server.wait_for(is_method("notifications/cancelled")) == {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 2, "reason": "stop"},
        }

        with pytest.raises(McpTimeoutError):
            await client.call_tool("wait", {}, timeout_ms=5)
    finally:
        release.set()
        await client.close()


@pytest.mark.tonio
async def test_reports_transport_errors_without_failing_pending_requests():
    client_transport, server = await create_server()
    client = McpClient(name="test-client", version="1.0.0")
    await client.connect(client_transport)
    errors = []

    async def on_error(error):
        errors.append(error)

    client.on_error(on_error)
    release = tonio.Event()
    server.set_handler("tools/call", _hang(release))

    async def stray_line_then_respond():
        await server.wait_for(is_method("tools/call"))
        client_transport.emit_error(RuntimeError("stray log line"))
        release.set()

    tonio.spawn.without_tracking(stray_line_then_respond())
    assert await client.call_tool("wait") == {"content": []}
    # The error was delivered before the response, which comes after it.
    assert [str(error) for error in errors] == ["stray log line"]
    await client.close()


@pytest.mark.tonio
async def test_accepts_servers_that_answer_with_an_older_protocol_version():
    client_transport, server = await create_server()

    async def old(_request):
        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "serverInfo": {"name": "old-server", "version": "0.1.0"},
        }

    server.set_handler("initialize", old)
    client = McpClient(name="test-client", version="1.0.0")
    await client.connect(client_transport)
    assert client.protocol_version == "2024-11-05"
    await client.close()

    unsupported_transport, unsupported = await create_server()

    async def ancient(_request):
        return {
            "protocolVersion": "1999-01-01",
            "capabilities": {},
            "serverInfo": {"name": "ancient-server", "version": "0.1.0"},
        }

    unsupported.set_handler("initialize", ancient)
    rejected = McpClient(name="test-client", version="1.0.0")
    with pytest.raises(Exception, match="unsupported protocol version"):
        await rejected.connect(unsupported_transport)
    assert rejected.connection_state == "closed"


@pytest.mark.tonio
async def test_defaults_missing_tool_result_content_to_an_empty_list():
    client, server = await connect()

    async def structured(_request):
        return {"structuredContent": {"ok": True}}

    async def broken(_request):
        return {"content": "not a list"}

    server.set_handler("tools/call", structured)
    assert await client.call_tool("structured") == {"content": [], "structuredContent": {"ok": True}}
    server.set_handler("tools/call", broken)
    with pytest.raises(McpError, match="Invalid MCP tools/call result"):
        await client.call_tool("broken")
    await client.close()


@pytest.mark.tonio
async def test_does_not_send_notifications_cancelled_for_a_timed_out_initialize():
    client_transport, server = await create_server()
    release = tonio.Event()

    async def hang(_request):
        await release.wait()
        return {}

    server.set_handler("initialize", hang)
    try:
        client = McpClient(name="test-client", version="1.0.0", request_timeout_ms=5)
        with pytest.raises(McpTimeoutError):
            await client.connect(client_transport)
        # The failed connect closed the transport; the server sees the close
        # only after every message sent before it.
        await server.closed.wait(_WAIT_S)
        assert server.closed.is_set()
        assert not any(message.get("method") == "notifications/cancelled" for message in server.snapshot())
    finally:
        release.set()


@pytest.mark.tonio
async def test_notifies_close_listeners_once_when_the_transport_drops():
    client, server = await connect()
    calls = []
    closed = tonio.Event()

    async def on_close():
        calls.append(None)
        closed.set()

    client.on_close(on_close)
    release = tonio.Event()
    server.set_handler("tools/call", _hang(release))
    try:

        async def drop_once_received():
            await server.wait_for(is_method("tools/call"))
            await server.transport.close()

        tonio.spawn.without_tracking(drop_once_received())
        with pytest.raises(McpConnectionClosedError, match="MCP connection closed"):
            await client.call_tool("wait")
        assert client.connection_state == "closed"
        await closed.wait(_WAIT_S)
        await client.close()
        assert len(calls) == 1
    finally:
        release.set()


@pytest.mark.tonio
async def test_answers_roots_list_and_dispatches_notifications():
    client_transport, server = await create_server()
    client = McpClient(name="test-client", version="1.0.0", roots=[{"uri": "file:///workspace", "name": "workspace"}])
    await client.connect(client_transport)
    changed = []
    notified = tonio.Event()

    async def on_changed(params):
        changed.append(params)
        notified.set()

    client.on_notification("notifications/tools/list_changed", on_changed)
    await server.transport.send({"jsonrpc": "2.0", "id": "roots", "method": "roots/list"})
    await server.transport.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})
    assert await server.wait_for(lambda message: message.get("id") == "roots") == {
        "jsonrpc": "2.0",
        "id": "roots",
        "result": {"roots": [{"uri": "file:///workspace", "name": "workspace"}]},
    }
    await notified.wait(_WAIT_S)
    assert changed == [None]
    await client.close()


@pytest.mark.tonio
async def test_a_request_whose_caller_is_cancelled_is_cancelled_on_the_server():
    """pidrei-only: a coroutine awaiting a request can be cancelled by its
    scope, which pi has no counterpart for. The request is abandoned: the
    server is told, and the entry is gone, so a late answer is unknown."""
    client, server = await connect()
    release = tonio.Event()
    server.set_handler("tools/call", _hang(release))
    errors = []
    reported = tonio.Event()

    async def on_error(error):
        errors.append(str(error))
        reported.set()

    client.on_error(on_error)
    try:
        async with tonio.scope(cancel_on_exc=True) as scope:
            scope.spawn(client.call_tool("wait"))
            await server.wait_for(is_method("tools/call"))
            scope.cancel()
        assert await server.wait_for(is_method("notifications/cancelled")) == {
            "jsonrpc": "2.0",
            "method": "notifications/cancelled",
            "params": {"requestId": 2},
        }
        await server.transport.send({"jsonrpc": "2.0", "id": 2, "result": {"content": []}})
        await reported.wait(_WAIT_S)
        assert errors == ["Received response for unknown MCP request 2"]
    finally:
        release.set()
        await client.close()
