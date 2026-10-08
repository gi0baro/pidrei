"""Mirror of pi mcp test/streamable-http.test.ts.

Not ported: "calls fetch without a receiver", which pins JavaScript's `this`
binding for Cloudflare Workers' platform fetch.

The servers record what they receive in a `RequestLog`; where pi asserts on
a request the client sent without waiting for it (the GET stream opens on
its own after initialization), the test waits for the log to have it.
"""

import json
import threading
from typing import Any

import pytest
import tonio.colored as tonio
from mcp_helpers import header, headers_of, loopback_servers, read_json

from pidrei_mcp import (
    LATEST_PROTOCOL_VERSION,
    AuthProvider,
    McpAuthRequiredError,
    McpClient,
    McpConnectionClosedError,
    McpSessionExpiredError,
    StreamableHttpReconnectOptions,
    StreamableHttpTransport,
)
from pidrei_mcp.transports.streamable_http import SseEvent, consume_sse_stream


_WAIT_S = 5.0


class RequestLog:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._arrived = tonio.Event()

    def add(self, method: str, request: Any, message: dict[str, Any] | None = None) -> None:
        with self._lock:
            self.entries.append({"method": method, "headers": headers_of(request), "message": message})
        self._arrived.set()

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.entries)

    async def wait_for(self, predicate) -> dict[str, Any]:
        deadline = tonio.time.time() + _WAIT_S
        while True:
            self._arrived.clear()
            for entry in self.snapshot():
                if predicate(entry):
                    return entry
            remaining = deadline - tonio.time.time()
            if remaining <= 0:
                raise AssertionError(f"server never received the request; got {self.snapshot()}")
            await self._arrived.wait(remaining)


def _json_body(value: Any) -> bytes:
    return json.dumps(value).encode()


async def protocol_handler(request: Any, log: RequestLog, message: dict[str, Any] | None = None) -> None:
    if request.method == "GET":
        log.add("GET", request)
        await request.respond(405)
        return
    if request.method == "DELETE":
        log.add("DELETE", request)
        await request.respond(200)
        return
    if message is None:
        message = await read_json(request)
    log.add(request.method, request, message)
    if "id" not in message:
        await request.respond(202)
        return
    if message.get("method") == "initialize":
        await request.respond(
            200,
            headers={"content-type": "application/json", "mcp-session-id": "session-1"},
            body=_json_body(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {
                        "protocolVersion": LATEST_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "http-fixture", "version": "1.0.0"},
                    },
                }
            ),
        )
        return
    if message.get("method") == "tools/list":
        await request.respond(
            200,
            headers={"content-type": "application/json"},
            body=_json_body(
                {
                    "jsonrpc": "2.0",
                    "id": message["id"],
                    "result": {"tools": [{"name": "echo", "inputSchema": {"type": "object"}}]},
                }
            ),
        )
        return
    result = {"jsonrpc": "2.0", "id": message["id"], "result": {"content": [{"type": "text", "text": "hello"}]}}
    await request.respond(
        200,
        headers={"content-type": "text/event-stream"},
        body=f"id: tool-result\ndata: {json.dumps(result)}\n\n".encode(),
    )


async def start_server(http_servers, handler) -> tuple[str, RequestLog]:
    log = RequestLog()

    async def handle(request, _origin):
        await handler(request, log)

    origin = await http_servers.listen(handle)
    return f"{origin}/mcp", log


def _tool_name(message: dict[str, Any]) -> str | None:
    return (message.get("params") or {}).get("name")


async def _chunks(*parts: str):
    for part in parts:
        yield part.encode()


@pytest.mark.tonio
async def test_consume_sse_stream_parses_chunked_crlf_events_comments_ids_and_multiline_data():
    events: list[SseEvent] = []

    async def on_event(event: SseEvent) -> None:
        events.append(event)

    await consume_sse_stream(
        _chunks(': keepalive\r\nid: 7\r\ndata: {"one":\r\n', "data: 1}\r\n\r\n"), on_event=on_event
    )
    assert events == [SseEvent(id="7", data='{"one":\n1}')]


@pytest.mark.tonio
async def test_consume_sse_stream_rejects_events_whose_data_lines_exceed_the_limit_without_a_blank_line():
    sent = 0

    async def endless():
        nonlocal sent
        # Never sends a blank line, so the event is never dispatched.
        while sent <= 1000:
            sent += 1
            yield b"data: xxxxxxxxxxxxxxxx\n"

    async def on_event(_event: SseEvent) -> None:
        pass

    chunks = endless()
    with pytest.raises(RuntimeError, match="MCP SSE event exceeds 256 bytes"):
        await consume_sse_stream(chunks, on_event=on_event, max_event_bytes=256)
    await chunks.aclose()
    assert sent < 100


@pytest.mark.tonio
async def test_handles_json_and_sse_responses_with_session_and_protocol_headers(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        url, log = await start_server(http_servers, protocol_handler)
        transport = StreamableHttpTransport(url)
        client = McpClient(name="http-test", version="1.0.0")
        await client.connect(transport)
        assert transport.session_id == "session-1"
        assert await client.list_tools() == [{"name": "echo", "inputSchema": {"type": "object"}}]
        assert await client.call_tool("echo", {"text": "hello"}) == {"content": [{"type": "text", "text": "hello"}]}
        await log.wait_for(lambda entry: entry["method"] == "GET")
        await client.close()

        list_request = next(e for e in log.snapshot() if (e["message"] or {}).get("method") == "tools/list")
        assert list_request["headers"]["mcp-session-id"] == "session-1"
        assert list_request["headers"]["mcp-protocol-version"] == LATEST_PROTOCOL_VERSION
        assert any(entry["method"] == "DELETE" for entry in log.snapshot())


# pi #10565: the auth provider may refresh over the network, which closing must not wait for.
@pytest.mark.tonio
async def test_closes_the_session_with_the_last_requests_token_without_asking_the_auth_provider(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        url, log = await start_server(http_servers, protocol_handler)
        calls = 0

        async def token() -> str:
            nonlocal calls
            calls += 1
            return f"token-{calls}"

        client = McpClient(name="http-test", version="1.0.0")
        await client.connect(
            StreamableHttpTransport(url, open_get_stream=False, auth_provider=AuthProvider(token=token))
        )
        before = calls
        await client.close()

        assert calls == before
        deletes = [entry for entry in log.snapshot() if entry["method"] == "DELETE"]
        assert [entry["headers"].get("authorization") for entry in deletes] == [f"Bearer token-{before}"]
        assert deletes[0]["headers"].get("mcp-session-id") == "session-1"


@pytest.mark.tonio
async def test_classifies_authentication_failures(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:

        async def handler(request, _log):
            await request.read()
            await request.respond(
                401,
                headers={"www-authenticate": 'Bearer resource_metadata="https://example.com/meta"'},
                body=b"login required",
            )

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        with pytest.raises(McpAuthRequiredError) as caught:
            await client.connect(StreamableHttpTransport(url))
        error = caught.value
        assert (error.status, error.body, error.www_authenticate) == (
            401,
            "login required",
            'Bearer resource_metadata="https://example.com/meta"',
        )


@pytest.mark.tonio
async def test_fails_only_the_request_whose_sse_stream_breaks(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        release_slow = tonio.Event()

        async def handler(request, log):
            if request.method != "POST":
                await protocol_handler(request, log)
                return
            message = await read_json(request)
            if _tool_name(message) == "broken":
                await request.respond(200, headers={"content-type": "text/event-stream"}, body=b"data: not json\n\n")
                return
            if _tool_name(message) == "slow":
                await release_slow.wait()
            await protocol_handler(request, log, message)

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        errors = []

        async def on_error(error):
            errors.append(error)

        client.on_error(on_error)
        await client.connect(StreamableHttpTransport(url, open_get_stream=False))
        try:
            slow = tonio.spawn(client.call_tool("slow"))
            with pytest.raises(Exception, match="MCP response stream failed"):
                await client.call_tool("broken")
            release_slow.set()
            assert await slow == {"content": [{"type": "text", "text": "hello"}]}
            assert len(errors) == 1
        finally:
            release_slow.set()
            await client.close()


@pytest.mark.tonio
async def test_opens_the_get_stream_after_initialization_and_sends_last_event_id_only_when_resuming(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        order: list[str] = []

        async def handler(request, log):
            if request.method == "POST":
                message = await read_json(request)
                order.append(str(message.get("method")))
                await protocol_handler(request, log, message)
                return
            order.append(request.method)
            await protocol_handler(request, log)

        url, log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        await client.connect(StreamableHttpTransport(url))
        await client.call_tool("echo")
        await client.list_tools()
        await log.wait_for(lambda entry: entry["method"] == "GET")
        await client.close()
        assert order.index("GET") > order.index("notifications/initialized")
        assert all("last-event-id" not in entry["headers"] for entry in log.snapshot())


@pytest.mark.tonio
async def test_resumes_a_response_stream_the_server_closed_before_answering(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        resume_headers: list[str] = []

        async def handler(request, log):
            if request.method == "GET" and header(request, "last-event-id"):
                resume_headers.append(header(request, "last-event-id"))
                result = {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "resumed"}]}}
                await request.respond(
                    200,
                    headers={"content-type": "text/event-stream"},
                    body=f"id: 2\ndata: {json.dumps(result)}\n\n".encode(),
                )
                return
            if request.method != "POST":
                await protocol_handler(request, log)
                return
            message = await read_json(request)
            if message.get("method") != "tools/call":
                await protocol_handler(request, log, message)
                return
            # Priming event (ID, no data) and a retry hint, then the server drops the stream.
            await request.respond(
                200, headers={"content-type": "text/event-stream"}, body=b"id: 1\nretry: 5\ndata:\n\n"
            )

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        errors = []

        async def on_error(error):
            errors.append(error)

        client.on_error(on_error)
        await client.connect(StreamableHttpTransport(url, open_get_stream=False))
        assert await client.call_tool("echo") == {"content": [{"type": "text", "text": "resumed"}]}
        assert resume_headers == ["1"]
        assert errors == []
        await client.close()


@pytest.mark.tonio
async def test_fails_a_request_whose_response_stream_ends_without_an_answer(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:

        async def handler(request, log):
            if request.method != "POST":
                await protocol_handler(request, log)
                return
            message = await read_json(request)
            if message.get("method") != "tools/call":
                await protocol_handler(request, log, message)
                return
            await request.respond(200, headers={"content-type": "text/event-stream"}, body=b": nothing here\n\n")

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        await client.connect(StreamableHttpTransport(url, open_get_stream=False))
        with pytest.raises(Exception, match="MCP response stream failed: stream ended without a response"):
            await client.call_tool("echo", {}, timeout_ms=5_000)
        await client.close()


@pytest.mark.tonio
async def test_reconnects_the_get_stream_after_it_drops(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        last_event_ids: list[str | None] = []
        notification = {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}

        async def handler(request, log):
            if request.method != "GET":
                await protocol_handler(request, log)
                return
            last_event_ids.append(header(request, "last-event-id"))
            if len(last_event_ids) == 1:
                await request.respond(
                    200,
                    headers={"content-type": "text/event-stream"},
                    body=f"id: g1\ndata: {json.dumps(notification)}\n\n".encode(),
                )
                return
            stream = await request.send_response(200, headers={"content-type": "text/event-stream"})
            await stream.send_data(f"id: g2\ndata: {json.dumps(notification)}\n\n".encode())
            await request.peer_closed()

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        changes = []
        second_change = tonio.Event()

        async def on_changed(_params):
            changes.append(None)
            if len(changes) == 2:
                second_change.set()

        client.on_notification("notifications/tools/list_changed", on_changed)
        await client.connect(StreamableHttpTransport(url, reconnect=StreamableHttpReconnectOptions(initial_delay_ms=1)))
        await second_change.wait(_WAIT_S)
        assert second_change.is_set()
        assert last_event_ids == [None, "g1"]
        await client.close()


@pytest.mark.tonio
async def test_rejects_a_request_the_server_accepts_without_a_response(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:

        async def handler(request, log):
            if request.method != "POST":
                await protocol_handler(request, log)
                return
            message = await read_json(request)
            if message.get("method") != "tools/call":
                await protocol_handler(request, log, message)
                return
            await request.respond(202)

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        await client.connect(StreamableHttpTransport(url, open_get_stream=False))
        with pytest.raises(Exception, match="without a response"):
            await client.call_tool("echo")
        await client.close()


@pytest.mark.tonio
async def test_includes_the_response_body_in_http_errors(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:

        async def handler(request, _log):
            await request.read()
            await request.respond(400, body=b"Invalid Accept header")

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        with pytest.raises(Exception, match="MCP HTTP request failed with status 400: Invalid Accept header"):
            await client.connect(StreamableHttpTransport(url))


@pytest.mark.tonio
async def test_hands_401_and_insufficient_scope_403_responses_to_the_auth_provider_with_the_rejected_token(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        seen: list[dict[str, Any]] = []
        token = "old"

        async def handler(request, log):
            if request.method != "POST":
                await protocol_handler(request, log)
                return
            message = await read_json(request)
            authorization = header(request, "authorization")
            if authorization == "Bearer old":
                await request.respond(401, headers={"www-authenticate": "Bearer"})
                return
            if message.get("method") == "tools/call" and authorization == "Bearer new":
                await request.respond(
                    403, headers={"www-authenticate": 'Bearer error="insufficient_scope", scope="admin"'}
                )
                return
            await protocol_handler(request, log, message)

        async def current_token():
            return token

        async def on_unauthorized(context):
            nonlocal token
            seen.append({"status": context.response.status, "token": context.token})
            token = "new" if context.response.status == 401 else "admin"

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        await client.connect(
            StreamableHttpTransport(
                url,
                open_get_stream=False,
                auth_provider=AuthProvider(token=current_token, on_unauthorized=on_unauthorized),
            )
        )
        assert await client.call_tool("echo") == {"content": [{"type": "text", "text": "hello"}]}
        assert seen == [{"status": 401, "token": "old"}, {"status": 403, "token": "new"}]
        await client.close()


@pytest.mark.tonio
async def test_classifies_an_expired_established_session(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        posts = 0

        async def handler(request, log):
            nonlocal posts
            if request.method == "POST":
                posts += 1
                if posts > 2:
                    await request.read()
                    await request.respond(404, body=b"gone")
                    return
            await protocol_handler(request, log)

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        await client.connect(StreamableHttpTransport(url, open_get_stream=False))
        with pytest.raises(McpSessionExpiredError):
            await client.list_tools()
        await client.close()


@pytest.mark.tonio
async def test_closing_the_transport_ends_an_open_server_to_client_stream(monkeypatch):
    """pidrei-only: the GET stream's reader is parked on an open response;
    the close closes that response, which ends its connection."""
    async with loopback_servers(monkeypatch) as http_servers:
        hung_up = tonio.Event()
        notification = {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}

        async def handler(request, log):
            if request.method != "GET":
                await protocol_handler(request, log)
                return
            stream = await request.send_response(200, headers={"content-type": "text/event-stream"})
            await stream.send_data(f"data: {json.dumps(notification)}\n\n".encode())
            if await request.peer_closed():
                hung_up.set()

        url, _log = await start_server(http_servers, handler)
        client = McpClient(name="http-test", version="1.0.0")
        streaming = tonio.Event()

        async def on_changed(_params):
            streaming.set()

        client.on_notification("notifications/tools/list_changed", on_changed)
        await client.connect(StreamableHttpTransport(url))
        await streaming.wait(_WAIT_S)
        assert streaming.is_set()
        await client.close()
        await hung_up.wait(_WAIT_S)
        assert hung_up.is_set()


class _HeldStream:
    """An SSE response that stays open until it is closed. Its close waits
    for `gate`, as a close that has not finished yet."""

    status = 200

    def __init__(self, gate: tonio.Event, closing: tonio.Event) -> None:
        self.headers = {"content-type": "text/event-stream"}
        self._gate = gate
        self._closing = closing
        self._ended = tonio.Event()
        self.reading = tonio.Event()
        self.closed = tonio.Event()

    async def iter_bytes(self):
        self.reading.set()
        await self._ended.wait(_WAIT_S)
        yield b""

    async def read(self) -> bytes:
        return b""

    async def close(self) -> None:
        self._closing.set()
        await self._gate.wait(_WAIT_S)
        self._ended.set()
        self.closed.set()


class _Accepted:
    status = 202

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}

    async def read(self) -> bytes:
        return b""

    async def close(self) -> None:
        pass


@pytest.mark.tonio
async def test_a_close_whose_caller_is_cancelled_still_closes_the_responses_it_holds():
    """pidrei-only: a coroutine closing the transport can be cancelled by its
    scope. The responses are out of the transport's reach by then, so the
    close must not stop with its caller."""
    gate = tonio.Event()
    closing = tonio.Event()
    opened = tonio.Event()
    streams: list[_HeldStream] = []

    async def fetch(_url, *, method="GET", headers=None, body=None, timeout_ms=None):
        # The GET stream, and the stream answering the request.
        if method == "GET" or "id" in json.loads(body):
            stream = _HeldStream(gate, closing)
            streams.append(stream)
            if len(streams) == 2:
                opened.set()
            return stream
        return _Accepted()

    transport = StreamableHttpTransport("http://server.invalid/mcp", fetch=fetch)
    await transport.start()
    await transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    await transport.send({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "wait"}})
    await opened.wait(_WAIT_S)
    assert len(streams) == 2
    # A stream that is being read is registered with the transport.
    for stream in streams:
        await stream.reading.wait(_WAIT_S)
        assert stream.reading.is_set()

    async with tonio.scope(cancel_on_exc=True) as scope:
        scope.spawn(transport.close())
        # The close is parked closing a response when its scope is cancelled.
        await closing.wait(_WAIT_S)
        assert closing.is_set()
        scope.cancel()
    gate.set()

    for stream in streams:
        await stream.closed.wait(_WAIT_S)
    assert [stream.closed.is_set() for stream in streams] == [True, True]


@pytest.mark.tonio
async def test_closing_the_transport_ends_a_request_waiting_for_its_response():
    """pidrei-only: pi's close aborts every fetch in flight through the
    transport's signal. Here the close fires a token, and a request still
    waiting for its response head is cancelled where it is parked."""
    waiting = tonio.Event()
    cancelled = tonio.Event()
    never = tonio.Event()

    async def fetch(_url, *, method="GET", headers=None, body=None, timeout_ms=None):
        answered = False
        waiting.set()
        try:
            await never.wait(_WAIT_S)
            answered = True
        finally:
            if not answered:
                cancelled.set()
        return _Accepted()

    transport = StreamableHttpTransport("http://server.invalid/mcp", fetch=fetch)
    await transport.start()
    sent = transport.send({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "wait"}})
    await waiting.wait(_WAIT_S)
    assert waiting.is_set()

    await transport.close()

    await cancelled.wait(_WAIT_S)
    assert cancelled.is_set()
    with pytest.raises(McpConnectionClosedError):
        await sent


@pytest.mark.tonio
async def test_a_request_whose_headers_resolve_after_the_close_is_not_sent():
    """pidrei-only: a request can be waiting for its auth token when the
    transport closes. It fails as closed without reaching the server."""
    asked = tonio.Event()
    release = tonio.Event()
    requests: list[str] = []

    async def token() -> str | None:
        asked.set()
        await release.wait(_WAIT_S)
        return "tok"

    async def fetch(_url, *, method="GET", headers=None, body=None, timeout_ms=None):
        requests.append(method)
        return _Accepted()

    transport = StreamableHttpTransport(
        "http://server.invalid/mcp", fetch=fetch, auth_provider=AuthProvider(token=token)
    )
    await transport.start()
    sent = transport.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    await asked.wait(_WAIT_S)
    assert asked.is_set()

    await transport.close()
    release.set()

    with pytest.raises(McpConnectionClosedError):
        await sent
    assert requests == []
