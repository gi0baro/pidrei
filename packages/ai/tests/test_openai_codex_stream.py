"""Mirror of pi's Codex adapter suites.

packages/ai/test/openai-codex-stream.test.ts (both transports) and
openai-codex-cache-affinity-e2e.test.ts (live, skipped as in pi).

pi stubs two globals: `fetch` and `WebSocket`. Here the SSE transport is
injected through `OpenAICodexResponsesOptions.client` (as in the openai-responses
suite) and the WebSocket through `utils/websocket.connect` — the module seam the
adapter resolves per connect, so a stub replaces exactly what pi's
`vi.stubGlobal("WebSocket", ...)` replaces.

`vi.useFakeTimers()` splits into two narrower helpers: `frozen_now` for the
tests that move the wall clock (the connection-age limit) and `recording_sleep`
for those that assert retry backoff. Deliberately not one combined fake clock:
the cached-socket idle timer sleeps through `clock.sleep_ms` too, and a clock
that returns from every sleep at once would expire cached sockets mid-test.
Short real waits (50 ms) cover the connect/idle deadlines.
"""

import base64
import contextlib
import inspect
import json
import time
from compression import zstd
from dataclasses import dataclass

import pytest
import tonio.colored as tonio
from tonio.colored import time as tonio_time
from tonio.colored.sync import channel

from pidrei_ai.api import openai_codex_responses as codex
from pidrei_ai.api.openai_codex_responses import (
    OpenAICodexResponsesOptions,
    close_openai_codex_websocket_sessions,
    get_openai_codex_websocket_debug_stats,
    reset_openai_codex_websocket_debug_stats,
    stream as stream_codex,
    stream_simple as stream_simple_codex,
)
from pidrei_ai.types import (
    Context,
    GrammarConstrainedSampling,
    JsonSchemaConstrainedSampling,
    Model,
    ModelCost,
    OpenAIResponsesCompat,
    TextContent,
    Tool,
    ToolResultMessage,
    UserMessage,
)
from pidrei_ai.utils import clock, http, websocket
from pidrei_ai.utils.cancel import CancelToken


# --- shared fixtures ----------------------------------------------------------


def mock_token(account_id: str = "acc_test") -> str:
    payload = base64.b64encode(
        json.dumps({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}).encode()
    ).decode()
    return f"aaa.{payload}.bbb"


def make_model(model_id: str = "gpt-5.1-codex", **overrides) -> Model:
    defaults: dict = {
        "id": model_id,
        "name": "GPT-5.5" if model_id == "gpt-5.5" else "GPT-5.1 Codex",
        "api": "openai-codex-responses",
        "provider": "openai-codex",
        "base_url": "https://chatgpt.com/backend-api",
        "reasoning": True,
        "input": ["text"],
        "cost": ModelCost(),
        "context_window": 400_000,
        "max_tokens": 128_000,
    }
    defaults.update(overrides)
    return Model(**defaults)


def hello_context() -> Context:
    return Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Say hello", timestamp=int(time.time() * 1000))],
    )


HELLO_EVENTS = [
    {
        "type": "response.output_item.added",
        "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []},
    },
    {"type": "response.content_part.added", "part": {"type": "output_text", "text": ""}},
    {"type": "response.output_text.delta", "delta": "Hello"},
    {
        "type": "response.output_item.done",
        "item": {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Hello"}],
        },
    },
]


def completion_event(status: str = "completed", **response) -> dict:
    event_type = "response.incomplete" if status == "incomplete" else "response.completed"
    body: dict = {
        "status": status,
        "usage": {
            "input_tokens": 5,
            "output_tokens": 3,
            "total_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0},
        },
    }
    if status == "incomplete":
        body["incomplete_details"] = {"reason": "max_output_tokens"}
    body.update(response)
    return {"type": event_type, "response": body}


def sse_payload(status: str = "completed", include_done: bool = False, end_turn: bool | None = None) -> bytes:
    extra = {} if end_turn is None else {"end_turn": end_turn}
    events = [*HELLO_EVENTS, completion_event(status, **extra)]
    chunks = [f"data: {json.dumps(event)}" for event in events]
    if include_done:
        chunks.append("data: [DONE]")
    return ("\n\n".join(chunks) + "\n\n").encode()


# --- SSE transport stub -------------------------------------------------------


@dataclass(slots=True)
class CodexRequest:
    url: str
    headers: dict[str, str]
    body: bytes
    timeout_ms: float | None
    cancel: CancelToken | None

    def payload(self) -> dict:
        raw = zstd.decompress(self.body) if self.headers.get("content-encoding") == "zstd" else self.body
        return json.loads(raw)


class FakeCodexResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        chunk_delay: float = 0.0,
        stay_open: bool = False,
    ):
        self.status = status
        self.headers = headers if headers is not None else {"content-type": "text/event-stream"}
        self.closed = False
        self.closed_event = tonio.Event()
        self.drained = False
        self._chunks = chunks
        self._chunk_delay = chunk_delay
        self._stay_open = stay_open

    async def aiter_bytes(self):
        for index, chunk in enumerate(self._chunks):
            if index and self._chunk_delay:
                await tonio_time.sleep(self._chunk_delay)
            yield chunk
        if self._stay_open:
            # pi's "SSE body stays open" cases: the stream must finish on
            # response.completed, not on the body ending.
            await tonio_time.sleep(30)
        self.drained = True

    async def read_text(self) -> str:
        return b"".join(self._chunks).decode()

    async def close(self) -> None:
        self.closed = True
        self.closed_event.set()


class FakeCodexClient:
    """The `CodexSSEClient` seam: answers each POST from `handler`."""

    def __init__(self, handler):
        self.requests: list[CodexRequest] = []
        self._handler = handler

    @property
    def call_count(self) -> int:
        return len(self.requests)

    async def post(self, url, *, headers, body, timeout_ms, cancel):
        request = CodexRequest(url=url, headers=dict(headers), body=body, timeout_ms=timeout_ms, cancel=cancel)
        self.requests.append(request)
        result = self._handler(request) if callable(self._handler) else self._handler
        if inspect.isawaitable(result):
            result = await result
        if isinstance(result, BaseException):
            raise result
        return result


def sse_client(status: str = "completed", **response_kwargs) -> FakeCodexClient:
    return FakeCodexClient(lambda _request: FakeCodexResponse([sse_payload(status)], **response_kwargs))


def unexpected_sse_client() -> FakeCodexClient:
    def handler(_request):
        raise AssertionError("SSE transport must not be used")

    return FakeCodexClient(handler)


# --- WebSocket transport stub -------------------------------------------------


class FakeWebSocket:
    """A queued-event socket, the shape `utils/websocket.connect` returns."""

    def __init__(self, connection_id: int, on_send=None):
        self.connection_id = connection_id
        self.ready_state = websocket.READY_STATE_OPEN
        self.sent: list[dict] = []
        self.closes: list[tuple[int | None, str | None]] = []
        self._on_send = on_send
        self._sender, self._receiver = channel.unbounded()

    def send(self, data: str) -> None:
        self.sent.append(json.loads(data))
        if self._on_send is not None:
            self._on_send(self, json.loads(data))

    def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closes.append((code, reason))
        self.ready_state = websocket.READY_STATE_CLOSED

    async def receive_event(self):
        return await self._receiver.receive()

    def emit(self, event) -> None:
        self._sender.send(event)

    def emit_messages(self, events: list[dict]) -> None:
        for event in events:
            self._sender.send(websocket.MessageEvent(data=json.dumps(event)))


@contextlib.contextmanager
def stub_websocket(connect):
    """Replace the WebSocket connector (pi: `vi.stubGlobal("WebSocket", ...)`).

    `connect(url, headers, index)` returns a `FakeWebSocket` (or raises/hangs);
    yields the list of connect calls.
    """
    original = websocket.connect
    calls: list[dict] = []

    async def stub(url, headers, *, cancel=None):
        calls.append({"url": url, "headers": dict(headers)})
        result = connect(url, headers, len(calls))
        if inspect.isawaitable(result):
            result = await result
        return result

    websocket.connect = stub
    try:
        yield calls
    finally:
        websocket.connect = original


def responding_websocket(events_for):
    """A connector whose sockets answer each send with `events_for(socket, body)`."""
    sockets: list[FakeWebSocket] = []

    def connect(_url, _headers, index):
        def on_send(socket, body):
            events = events_for(socket, body)
            if events:
                socket.emit_messages(events)

        socket = FakeWebSocket(index, on_send)
        sockets.append(socket)
        return socket

    return connect, sockets


# --- clock helpers ------------------------------------------------------------


@contextlib.contextmanager
def frozen_now(start_ms: int = 1_772_100_000_000):
    """`vi.setSystemTime`: freeze `clock.now_ms`; the test moves it."""
    original = clock.now_ms
    state = {"now": start_ms}
    clock.now_ms = lambda: state["now"]
    try:
        yield state
    finally:
        clock.now_ms = original


@contextlib.contextmanager
def recording_sleep():
    """Record `clock.sleep_ms` durations and return at once (SSE-only tests)."""
    original_sleep = clock.sleep_ms
    original_now = clock.now_ms
    state = {"now": 1_778_976_000_000}  # 2026-05-13T00:00:00Z, pi's setSystemTime
    delays: list[float] = []

    async def sleep_ms(ms, cancel=None):
        delays.append(ms)
        state["now"] += int(ms)
        await tonio_time.sleep(0)

    clock.sleep_ms = sleep_ms
    clock.now_ms = lambda: state["now"]
    try:
        yield delays
    finally:
        clock.sleep_ms = original_sleep
        clock.now_ms = original_now


def reset_codex_state() -> None:
    """pi's `afterEach`."""
    close_openai_codex_websocket_sessions()
    reset_openai_codex_websocket_debug_stats()


@pytest.fixture(autouse=True)
def _codex_state(request):
    # A finalizer (predates tonio 0.9.14 yield-fixture support; equivalent either way).
    reset_codex_state()
    request.addfinalizer(reset_codex_state)


def text_of(message) -> str | None:
    return next((block.text for block in message.content if block.type == "text"), None)


# --- SSE transport ------------------------------------------------------------


@pytest.mark.tonio
async def test_streams_sse_responses_into_event_stream():
    token = mock_token()
    client = sse_client()

    result_stream = stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=token, transport="sse", client=client),
    )
    saw_text_delta = False
    saw_done = False
    async for event in result_stream:
        if event.type == "text_delta":
            saw_text_delta = True
        if event.type == "done":
            saw_done = True
            assert text_of(event.message) == "Hello"

    assert saw_text_delta
    assert saw_done

    request = client.requests[0]
    assert request.url == "https://chatgpt.com/backend-api/codex/responses"
    assert request.headers["Authorization"] == f"Bearer {token}"
    assert request.headers["chatgpt-account-id"] == "acc_test"
    assert request.headers["OpenAI-Beta"] == "responses=experimental"
    from pidrei_ai.utils.user_agent import ORIGINATOR, get_user_agent

    assert request.headers["originator"] == ORIGINATOR

    assert request.headers["User-Agent"] == get_user_agent()
    assert request.headers["accept"] == "text/event-stream"
    assert "x-api-key" not in request.headers


@pytest.mark.tonio
async def test_processes_a_terminal_sse_event_without_a_trailing_blank_line():
    # Regression test for https://github.com/earendil-works/pi/issues/9047
    client = FakeCodexClient(lambda _request: FakeCodexResponse([sse_payload("completed").rstrip()]))
    result = await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", client=client),
    ).result()

    assert result.stop_reason == "stop"
    assert text_of(result) == "Hello"


@pytest.mark.tonio
async def test_completes_after_response_completed_even_when_the_sse_body_stays_open():
    client = FakeCodexClient(
        lambda _request: FakeCodexResponse(
            [sse_payload("completed", include_done=True, end_turn=False)], stay_open=True
        )
    )
    result = await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", client=client),
    ).result()

    assert text_of(result) == "Hello"
    assert result.stop_reason == "stop"
    assert result.end_turn is False


@pytest.mark.tonio
async def test_maps_response_incomplete_to_stop_reason_length_with_open_body():
    client = FakeCodexClient(lambda _request: FakeCodexResponse([sse_payload("incomplete")], stay_open=True))
    result = await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", client=client),
    ).result()

    assert text_of(result) == "Hello"
    assert result.stop_reason == "length"


@pytest.mark.tonio
async def test_aborts_sse_request_after_the_configured_http_timeout():
    def handler(request):
        assert request.timeout_ms == 10
        return http.RequestTimeout("read timed out")

    client = FakeCodexClient(handler)
    result = await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", timeout_ms=10, client=client),
    ).result()

    assert client.call_count == 1
    assert result.stop_reason == "error"
    assert result.error_message == "Codex SSE response headers timed out after 10ms"


@pytest.mark.tonio
async def test_aborts_sse_body_reads_after_response_headers_arrive():
    first = (
        "\n\n".join(
            f"data: {json.dumps(event)}"
            for event in [
                HELLO_EVENTS[0],
                HELLO_EVENTS[1],
                {"type": "response.output_text.delta", "delta": "one"},
            ]
        )
        + "\n\n"
    ).encode()
    second = (f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': 'two'})}\n\n").encode()
    response = FakeCodexResponse([first, second, sse_payload()], chunk_delay=0.05)
    client = FakeCodexClient(lambda _request: response)

    cancel = CancelToken()
    result_stream = stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", cancel=cancel, client=client),
    )
    events: list[str] = []
    async for event in result_stream:
        events.append(f"text_delta:{event.delta}" if event.type == "text_delta" else event.type)
        if event.type == "text_delta" and event.delta == "one":
            cancel.cancel()

    result = await result_stream.result()
    assert result.stop_reason == "aborted"
    assert result.error_message == "Request was aborted"
    assert "text_delta:one" in events
    assert "text_delta:two" not in events
    # The aborted event is published by the stream's owner as soon as the
    # scope is cancelled; the producer's unwinding (which closes the body)
    # runs concurrently, so the close lands after it, not before.
    await response.closed_event.wait(1.0)
    assert response.closed


@pytest.mark.tonio
async def test_sets_session_headers_and_prompt_cache_key_when_session_id_is_provided():
    session_id = "test-session-123"
    client = sse_client()
    await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), session_id=session_id, transport="sse", client=client),
    ).result()

    request = client.requests[0]
    assert request.headers["session-id"] == session_id
    assert "session_id" not in request.headers
    assert request.headers["x-client-request-id"] == session_id
    assert request.payload()["prompt_cache_key"] == session_id


@pytest.mark.tonio
async def test_omits_sse_cache_affinity_when_cache_retention_is_none():
    client = sse_client()
    await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(
            api_key=mock_token(),
            cache_retention="none",
            session_id="one-off-summary",
            transport="sse",
            client=client,
        ),
    ).result()

    request = client.requests[0]
    assert "session-id" not in request.headers
    assert "x-client-request-id" not in request.headers
    assert "prompt_cache_key" not in request.payload()


@pytest.mark.tonio
async def test_clamps_prompt_cache_key_to_64_characters():
    captured: dict = {}

    async def on_payload(payload, _model):
        captured.update(payload)

    await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(
            api_key=mock_token(),
            transport="sse",
            session_id="x" * 67,
            on_payload=on_payload,
            client=sse_client(),
        ),
    ).result()

    assert captured["prompt_cache_key"] == "x" * 64


@pytest.mark.tonio
async def test_clamps_codex_session_id_header_to_64_characters():
    client = sse_client()
    await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", session_id="x" * 67, client=client),
    ).result()

    request = client.requests[0]
    assert request.headers["session-id"] == "x" * 64
    assert request.headers["x-client-request-id"] == "x" * 64


@pytest.mark.tonio
async def test_preserves_gpt55_xhigh_reasoning_effort_from_simple_options():
    from pidrei_ai.types import SimpleStreamOptions

    client = sse_client()
    model = make_model("gpt-5.5", thinking_level_map={"xhigh": "xhigh"})
    await stream_simple_codex(
        model,
        hello_context(),
        SimpleStreamOptions(api_key=mock_token(), reasoning="xhigh", transport="sse"),
    ).result()
    # stream_simple builds its own options, so the transport is asserted through
    # a second run that injects the client.
    await stream_codex(
        model,
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", reasoning_effort="xhigh", client=client),
    ).result()
    assert client.requests[0].payload()["reasoning"] == {"effort": "xhigh", "summary": "auto"}


@pytest.mark.tonio
@pytest.mark.parametrize("model_id", ["gpt-5.3-codex", "gpt-5.4", "gpt-5.5"])
async def test_clamps_minimal_reasoning_effort_to_low(model_id):
    client = sse_client()
    await stream_codex(
        make_model(model_id, thinking_level_map={"minimal": "low"}),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), reasoning_effort="minimal", transport="sse", client=client),
    ).result()

    assert client.requests[0].payload()["reasoning"] == {"effort": "low", "summary": "auto"}


@pytest.mark.tonio
async def test_forwards_required_tool_choice():
    client = sse_client()
    context = Context(
        messages=[UserMessage(content="Do not call ping. Respond with text instead.", timestamp=1)],
        tools=[
            Tool(
                name="ping",
                description="Ping",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
            )
        ],
    )
    await stream_codex(
        make_model("gpt-5.5"),
        context,
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", tool_choice="required", client=client),
    ).result()

    assert client.requests[0].payload()["tool_choice"] == "required"


@pytest.mark.tonio
async def test_sets_codex_strict_mode_explicitly_and_honors_constrained_sampling():
    captured: dict = {}

    async def on_payload(payload, _model):
        captured.update(payload)

    context = Context(
        messages=[UserMessage(content="Use a tool", timestamp=1)],
        tools=[
            Tool(
                name="optional",
                description="Optional constrained sampling",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]},
                constrained_sampling=False,
            ),
            Tool(
                name="strict",
                description="Strict constrained sampling",
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
                constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
            ),
        ],
    )
    await stream_codex(
        make_model("gpt-5.5"),
        context,
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", on_payload=on_payload, client=sse_client()),
    ).result()

    tools = captured["tools"]
    assert [(tool["type"], tool["name"], tool["strict"]) for tool in tools] == [
        ("function", "optional", None),
        ("function", "strict", True),
    ]


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("model_id", "service_tier", "multiplier"),
    [
        ("gpt-5.1-codex", "flex", 0.5),
        ("gpt-5.1-codex", "priority", 2),
        ("gpt-5.5", "flex", 0.5),
        ("gpt-5.5", "priority", 2.5),
    ],
)
async def test_uses_the_client_sent_service_tier_when_codex_echoes_default(model_id, service_tier, multiplier):
    events = [
        *HELLO_EVENTS,
        completion_event(
            service_tier="default",
            usage={
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "total_tokens": 2_000_000,
                "input_tokens_details": {"cached_tokens": 0},
            },
        ),
    ]
    body = ("\n\n".join(f"data: {json.dumps(event)}" for event in events) + "\n\n").encode()
    client = FakeCodexClient(lambda _request: FakeCodexResponse([body]))

    result = await stream_codex(
        make_model(model_id, cost=ModelCost(input=1, output=2)),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), service_tier=service_tier, transport="sse", client=client),
    ).result()

    assert result.usage.cost.input == 1 * multiplier
    assert result.usage.cost.output == 2 * multiplier
    assert result.usage.cost.total == 3 * multiplier


@pytest.mark.tonio
async def test_does_not_set_session_headers_when_session_id_is_absent():
    client = sse_client()
    await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", client=client),
    ).result()

    headers = client.requests[0].headers
    assert "session-id" not in headers
    assert "session_id" not in headers
    assert "x-client-request-id" not in headers


@pytest.mark.tonio
async def test_zstd_compresses_sse_request_bodies():
    client = sse_client()
    large_text = "compress me " * 400
    await stream_codex(
        make_model(),
        Context(
            system_prompt="You are a helpful assistant.",
            messages=[UserMessage(content=large_text, timestamp=1)],
        ),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", client=client),
    ).result()

    request = client.requests[0]
    assert request.headers["content-encoding"] == "zstd"
    assert isinstance(request.body, bytes)
    assert request.payload()["input"][0]["content"][0]["text"] == large_text

    await stream_codex(
        make_model(),
        Context(system_prompt="You are a helpful assistant.", messages=[UserMessage(content="hi", timestamp=1)]),
        OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", client=client),
    ).result()
    assert client.requests[1].headers["content-encoding"] == "zstd"


# --- SSE retries --------------------------------------------------------------


def rate_limited_response(headers: dict[str, str]) -> FakeCodexResponse:
    return FakeCodexResponse(
        [json.dumps({"error": {"code": "rate_limit_exceeded", "message": "rate limited"}}).encode()],
        status=429,
        headers=headers,
    )


@pytest.mark.tonio
@pytest.mark.parametrize(
    ("name", "headers", "expected_delay"),
    [
        ("retry-after-ms", {"content-type": "application/json", "retry-after-ms": "1500"}, 1500),
        ("retry-after seconds", {"content-type": "application/json", "retry-after": "60"}, 60_000),
        ("retry-after HTTP date", {"content-type": "application/json", "retry-after": "@date+45"}, 45_000),
    ],
)
async def test_uses_retry_after_headers_for_sse_retries(name, headers, expected_delay):
    with recording_sleep() as delays:
        if headers.get("retry-after") == "@date+45":
            from email.utils import formatdate

            headers = {**headers, "retry-after": formatdate(clock.now_ms() / 1000 + 45, usegmt=True)}

        codex_requests = {"count": 0}

        def handler(_request):
            codex_requests["count"] += 1
            if codex_requests["count"] == 1:
                return rate_limited_response(headers)
            return FakeCodexResponse([sse_payload()])

        client = FakeCodexClient(handler)
        result = await stream_codex(
            make_model(),
            hello_context(),
            OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", max_retries=1, client=client),
        ).result()

        assert text_of(result) == "Hello", result.error_message
        assert codex_requests["count"] == 2
        assert delays == [expected_delay]


@pytest.mark.tonio
@pytest.mark.parametrize("status", [429, 503])
async def test_fails_immediately_when_a_retry_delay_exceeds_the_limit(status):
    client = FakeCodexClient(
        lambda _request: FakeCodexResponse(
            [json.dumps({"error": {"code": "temporarily_unavailable", "message": "retry later"}}).encode()],
            status=status,
            headers={"content-type": "application/json", "retry-after": "2"},
        )
    )
    result = await stream_codex(
        make_model(),
        hello_context(),
        OpenAICodexResponsesOptions(
            api_key=mock_token(), transport="sse", max_retries=3, max_retry_delay_ms=1000, client=client
        ),
    ).result()

    assert result.stop_reason == "error"
    assert result.error_message == "Server requested 2s retry delay (max: 1s)"
    assert client.call_count == 1


@pytest.mark.tonio
async def test_uses_exponential_backoff_across_repeated_sse_retries():
    with recording_sleep() as delays:
        codex_requests = {"count": 0}

        def handler(_request):
            codex_requests["count"] += 1
            if codex_requests["count"] <= 3:
                return rate_limited_response({"content-type": "application/json"})
            return FakeCodexResponse([sse_payload()])

        client = FakeCodexClient(handler)
        result = await stream_codex(
            make_model(),
            hello_context(),
            OpenAICodexResponsesOptions(api_key=mock_token(), transport="sse", max_retries=3, client=client),
        ).result()

        assert text_of(result) == "Hello", result.error_message
        assert codex_requests["count"] == 4
        assert delays == [1000, 2000, 4000]


# --- WebSocket transport ------------------------------------------------------


@pytest.mark.tonio
async def test_forwards_auto_transport_from_simple_options_and_uses_cached_websocket_context():
    from pidrei_ai.types import SimpleStreamOptions

    connect, sockets = responding_websocket(lambda _socket, _body: [*HELLO_EVENTS, completion_event(end_turn=False)])
    with stub_websocket(connect) as calls:
        result = await stream_simple_codex(
            make_model(),
            Context(
                system_prompt="You are a helpful assistant.",
                messages=[UserMessage(content="Say hello", timestamp=1)],
            ),
            SimpleStreamOptions(api_key=mock_token(), session_id="session-auto", transport="auto"),
        ).result()

    assert result.end_turn is False
    assert len(sockets) == 1
    assert len(sockets[0].sent) == 1
    assert calls[0]["url"] == "wss://chatgpt.com/backend-api/codex/responses"
    assert calls[0]["headers"]["session-id"] == "session-auto"
    assert "session_id" not in calls[0]["headers"]
    assert calls[0]["headers"]["x-client-request-id"] == "session-auto"
    # `connectWebSocket` strips OpenAI-Beta before handing headers to the socket.
    assert not any(name.lower() == "openai-beta" for name in calls[0]["headers"])

    stats = get_openai_codex_websocket_debug_stats("session-auto")
    assert stats.cached_context_requests == 1
    assert stats.full_context_requests == 1


@pytest.mark.tonio
async def test_scopes_cached_websockets_to_the_authenticated_account():
    # Regression for pi #7284: rotating accounts must not reuse a socket
    # authenticated by another account.
    connect, _sockets = responding_websocket(
        lambda socket, _body: [completion_event(id=f"resp_{socket.connection_id}")]
    )
    sse = unexpected_sse_client()
    context = Context(system_prompt="", messages=[])

    def options(account_id: str) -> OpenAICodexResponsesOptions:
        return OpenAICodexResponsesOptions(
            api_key=mock_token(account_id),
            session_id="shared-session",
            transport="websocket-cached",
            client=sse,
        )

    with stub_websocket(connect) as calls:
        await stream_codex(make_model(), context, options("account-a")).result()
        await stream_codex(make_model(), context, options("account-b")).result()
        await stream_codex(make_model(), context, options("account-a")).result()

    assert [call["headers"].get("chatgpt-account-id") for call in calls] == ["account-a", "account-b"]
    assert [call["headers"].get("Authorization") for call in calls] == [
        f"Bearer {mock_token('account-a')}",
        f"Bearer {mock_token('account-b')}",
    ]
    assert sse.call_count == 0
    stats = get_openai_codex_websocket_debug_stats("shared-session")
    assert stats.connections_created == 2
    assert stats.connections_reused == 1


@pytest.mark.tonio
async def test_closes_one_shot_websockets_when_cache_retention_is_none():
    def events_for(socket, _body):
        return [completion_event(id=f"resp_{socket.connection_id}")]

    connect, sockets = responding_websocket(events_for)
    sse = unexpected_sse_client()
    options = {
        "api_key": mock_token(),
        "cache_retention": "none",
        "session_id": "one-off-summary",
        "transport": "auto",
        "client": sse,
    }
    with stub_websocket(connect):
        await stream_codex(make_model(), hello_context(), OpenAICodexResponsesOptions(**options)).result()
        await stream_codex(make_model(), hello_context(), OpenAICodexResponsesOptions(**options)).result()

    assert len(sockets) == 2
    assert all(socket.closes for socket in sockets)
    assert all("prompt_cache_key" not in socket.sent[0] for socket in sockets)
    assert get_openai_codex_websocket_debug_stats("one-off-summary") is None
    assert sse.call_count == 0


@pytest.mark.tonio
async def test_falls_back_to_sse_when_websocket_connect_times_out():
    async def connect(_url, _headers, _index):
        await tonio_time.sleep(30)
        raise AssertionError("connect should have been abandoned")

    client = sse_client()
    with stub_websocket(connect):
        result = await stream_codex(
            make_model(),
            hello_context(),
            OpenAICodexResponsesOptions(
                api_key=mock_token(),
                session_id="ws-connect-timeout",
                transport="auto",
                timeout_ms=300_000,
                websocket_connect_timeout_ms=50,
                client=client,
            ),
        ).result()

    assert text_of(result) == "Hello", result.error_message
    assert client.call_count == 1
    stats = get_openai_codex_websocket_debug_stats("ws-connect-timeout")
    assert stats.websocket_failures == 1
    assert stats.sse_fallbacks == 1
    assert stats.websocket_fallback_active is True
    assert stats.last_websocket_error == "WebSocket connect timeout after 50ms"


@pytest.mark.tonio
async def test_reconnects_once_when_the_websocket_connection_limit_is_reached_before_output_starts():
    def events_for(socket, _body):
        if socket.connection_id == 1:
            return [{"type": "error", "error": {"code": "websocket_connection_limit_reached"}}]
        return [completion_event(id="resp_1")]

    connect, sockets = responding_websocket(events_for)
    sse = unexpected_sse_client()
    with stub_websocket(connect):
        result = await stream_codex(
            make_model(),
            Context(system_prompt="", messages=[]),
            OpenAICodexResponsesOptions(api_key=mock_token(), client=sse),
        ).result()

    assert result.stop_reason == "stop", result.error_message
    assert len(sockets) == 2
    assert sse.call_count == 0


@pytest.mark.tonio
async def test_falls_back_to_sse_when_a_websocket_is_idle_before_the_first_event():
    connect, sockets = responding_websocket(lambda _socket, _body: [])
    client = sse_client()
    with stub_websocket(connect):
        result = await stream_codex(
            make_model(),
            hello_context(),
            OpenAICodexResponsesOptions(
                api_key=mock_token(),
                session_id="ws-idle-before-start",
                transport="auto",
                timeout_ms=50,
                client=client,
            ),
        ).result()

    assert len(sockets[0].sent) == 1
    assert text_of(result) == "Hello", result.error_message
    assert client.call_count == 1
    stats = get_openai_codex_websocket_debug_stats("ws-idle-before-start")
    assert stats.websocket_failures == 1
    assert stats.sse_fallbacks == 1
    assert stats.websocket_fallback_active is True


@pytest.mark.tonio
async def test_errors_when_a_websocket_is_idle_after_the_stream_started():
    connect, _sockets = responding_websocket(lambda _socket, _body: [HELLO_EVENTS[0]])
    sse = unexpected_sse_client()
    with stub_websocket(connect):
        result = await stream_codex(
            make_model(),
            hello_context(),
            OpenAICodexResponsesOptions(api_key=mock_token(), transport="auto", timeout_ms=50, client=sse),
        ).result()

    assert result.stop_reason == "error"
    assert result.error_message == "WebSocket idle timeout after 50ms"
    assert sse.call_count == 0


@pytest.mark.tonio
async def test_opens_a_fresh_cached_websocket_before_the_backend_connection_age_limit():
    def events_for(socket, _body):
        return [completion_event(id=f"resp_{socket.connection_id}")]

    connect, sockets = responding_websocket(events_for)
    session_id = "aged-ws-session"
    first_context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Say hello", timestamp=1)],
    )

    with frozen_now() as now, stub_websocket(connect):
        first = await stream_codex(
            make_model(),
            first_context,
            OpenAICodexResponsesOptions(api_key=mock_token(), session_id=session_id, transport="websocket-cached"),
        ).result()
        now["now"] += 56 * 60 * 1000
        second_context = Context(
            system_prompt="You are a helpful assistant.",
            messages=[*first_context.messages, first, UserMessage(content="Now finish", timestamp=2)],
        )
        await stream_codex(
            make_model(),
            second_context,
            OpenAICodexResponsesOptions(api_key=mock_token(), session_id=session_id, transport="websocket-cached"),
        ).result()

    assert len(sockets) == 2
    assert [socket.connection_id for socket in sockets if socket.sent] == [1, 2]
    stats = get_openai_codex_websocket_debug_stats(session_id)
    assert stats.connections_created == 2
    assert stats.connections_reused == 0


@pytest.mark.tonio
async def test_sends_only_response_input_deltas_in_websocket_cached_mode():
    sent_bodies: list[dict] = []

    def connect(_url, _headers, index):
        def on_send(socket, body):
            sent_bodies.append(body)
            response_id = f"resp_{len(sent_bodies)}"
            output_events = (
                [
                    {
                        "type": "response.output_item.added",
                        "item": {
                            "type": "custom_tool_call",
                            "id": "ctc_1",
                            "call_id": "call_1",
                            "name": "sample_tool",
                            "input": "",
                        },
                    },
                    {"type": "response.custom_tool_call_input.delta", "item_id": "ctc_1", "delta": "abc"},
                    {"type": "response.custom_tool_call_input.done", "item_id": "ctc_1", "input": "abc"},
                    {
                        "type": "response.output_item.done",
                        "item": {
                            "type": "custom_tool_call",
                            "id": "ctc_1",
                            "call_id": "call_1",
                            "name": "sample_tool",
                            "input": "abc",
                        },
                    },
                ]
                if len(sent_bodies) == 1
                else []
            )
            socket.emit_messages(
                [
                    {"type": "response.created", "response": {"id": response_id}},
                    *output_events,
                    completion_event(id=response_id),
                ]
            )

        return FakeWebSocket(index, on_send)

    model = make_model(compat=OpenAIResponsesCompat(supports_openai_grammar_tools=True))
    tools = [
        Tool(
            name="sample_tool",
            description="Sample tool",
            parameters={"type": "object", "properties": {"payload": {"type": "string"}}, "required": ["payload"]},
            constrained_sampling=GrammarConstrainedSampling(variants={"openai_lark": "start: /[a-z]+/"}),
        )
    ]
    first_context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Use the tool", timestamp=1)],
        tools=tools,
    )

    with stub_websocket(connect):
        first = await stream_codex(
            model,
            first_context,
            OpenAICodexResponsesOptions(api_key=mock_token(), session_id="session-1", transport="websocket-cached"),
        ).result()
        second_context = Context(
            system_prompt="You are a helpful assistant.",
            messages=[
                *first_context.messages,
                first,
                ToolResultMessage(
                    tool_call_id="call_1|ctc_1",
                    tool_name="sample_tool",
                    content=[TextContent(text="real result")],
                    is_error=False,
                    timestamp=2,
                ),
                UserMessage(content="Now finish", timestamp=3),
            ],
            tools=tools,
        )
        await stream_codex(
            model,
            second_context,
            OpenAICodexResponsesOptions(api_key=mock_token(), session_id="session-1", transport="websocket-cached"),
        ).result()

    assert len(sent_bodies) == 2
    assert sent_bodies[0]["store"] is False
    assert "previous_response_id" not in sent_bodies[0]
    assert sent_bodies[0]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "Use the tool"}]}]
    assert sent_bodies[1]["store"] is False
    assert sent_bodies[1]["previous_response_id"] == "resp_1"
    assert sent_bodies[1]["input"] == [
        {"type": "custom_tool_call_output", "call_id": "call_1", "output": "real result"},
        {"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]},
    ]

    stats = get_openai_codex_websocket_debug_stats("session-1")
    assert stats.requests == 2
    assert stats.connections_created == 1
    assert stats.connections_reused == 1
    assert stats.cached_context_requests == 2
    assert stats.store_true_requests == 0
    assert stats.full_context_requests == 1
    assert stats.delta_requests == 1
    assert stats.last_delta_input_items == 2
    assert stats.last_previous_response_id == "resp_1"


@pytest.mark.tonio
@pytest.mark.parametrize("recovery_transport", ["websocket", "sse"])
async def test_recovers_a_missing_cached_websocket_continuation(recovery_transport):
    session_id = f"missing-continuation-{recovery_transport}"
    sent_bodies: list[dict] = []
    client = sse_client()

    def connect(_url, _headers, index):
        def on_send(socket, body):
            sent_bodies.append({**body, "connection_id": socket.connection_id})
            if len(sent_bodies) == 2:
                socket.emit_messages(
                    [
                        {
                            "type": "codex.rate_limits",
                            "plan_type": "plus",
                            "rate_limits": {
                                "allowed": True,
                                "limit_reached": False,
                                "primary": {
                                    "used_percent": 7,
                                    "window_minutes": 10080,
                                    "reset_after_seconds": 556112,
                                    "reset_at": 1785269351,
                                },
                                "secondary": None,
                            },
                            "code_review_rate_limits": None,
                            "additional_rate_limits": None,
                            "credits": {"has_credits": False, "unlimited": False, "balance": "0"},
                            "promo": None,
                        },
                        {
                            "type": "error",
                            "status": 400,
                            "error": {
                                "code": "previous_response_not_found",
                                "message": "Previous response with id 'resp_1' not found.",
                                "param": "previous_response_id",
                            },
                        },
                    ]
                )
                return
            if len(sent_bodies) == 3 and recovery_transport == "sse":
                socket.emit(websocket.ErrorEvent(message="retry websocket failed"))
                return

            response = (
                {"response_id": "resp_1", "message_id": "msg_1", "text": "Hello"}
                if len(sent_bodies) == 1
                else {"response_id": "resp_2", "message_id": "msg_2", "text": "Recovered"}
            )
            socket.emit_messages(
                [
                    {"type": "response.created", "response": {"id": response["response_id"]}},
                    {
                        "type": "response.output_item.added",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": response["message_id"],
                            "role": "assistant",
                            "status": "in_progress",
                            "content": [],
                        },
                    },
                    {
                        "type": "response.output_item.done",
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": response["message_id"],
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": response["text"]}],
                        },
                    },
                    completion_event(id=response["response_id"]),
                ]
            )

        return FakeWebSocket(index, on_send)

    first_context = Context(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content="Say hello", timestamp=1)],
    )
    with stub_websocket(connect) as calls:
        first = await stream_codex(
            make_model(),
            first_context,
            OpenAICodexResponsesOptions(
                api_key=mock_token(), session_id=session_id, transport="websocket-cached", client=client
            ),
        ).result()
        second_context = Context(
            system_prompt="You are a helpful assistant.",
            messages=[*first_context.messages, first, UserMessage(content="Now finish", timestamp=2)],
        )
        second_stream = stream_codex(
            make_model(),
            second_context,
            OpenAICodexResponsesOptions(
                api_key=mock_token(), session_id=session_id, transport="websocket-cached", client=client
            ),
        )
        event_types = [event.type async for event in second_stream]
        second = await second_stream.result()

    assert second.stop_reason == "stop", second.error_message
    assert text_of(second) == ("Hello" if recovery_transport == "sse" else "Recovered")
    assert event_types.count("start") == 1
    assert "error" not in event_types
    assert len(calls) == 2
    assert len(sent_bodies) == 3
    assert [body["connection_id"] for body in sent_bodies] == [1, 1, 2]
    assert sent_bodies[1]["previous_response_id"] == "resp_1"
    assert sent_bodies[1]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "Now finish"}]}]
    assert "previous_response_id" not in sent_bodies[2]
    assert len(sent_bodies[2]["input"]) == 3
    assert sent_bodies[2]["input"][-1] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Now finish"}],
    }
    assert client.call_count == (1 if recovery_transport == "sse" else 0)

    stats = get_openai_codex_websocket_debug_stats(session_id)
    assert stats.requests == 3
    assert stats.connections_created == 2
    assert stats.connections_reused == 1
    assert stats.full_context_requests == 2
    assert stats.delta_requests == 1
    assert stats.websocket_failures == (1 if recovery_transport == "sse" else 0)
    assert stats.sse_fallbacks == (1 if recovery_transport == "sse" else 0)


@pytest.mark.tonio
async def test_concurrent_turns_on_one_session_do_not_share_a_socket():
    """pidrei-only: pi gets check-then-claim atomicity from its single thread.

    pi handles concurrent turns on one session (the cached socket is `busy`, so
    the second turn dials a one-shot) but relies on JavaScript never
    interleaving the check and the claim. Here they genuinely interleave, so the
    claim is under the state lock — and a socket must never take two sends.
    """

    sockets: list[FakeWebSocket] = []

    def connect(_url, _headers, index):
        def on_send(socket, _body):
            async def answer():
                # Slow enough that the second turn acquires while the first holds.
                await tonio_time.sleep(0.03)
                socket.emit_messages([completion_event(id=f"resp_{socket.connection_id}_{len(socket.sent)}")])

            tonio.spawn.without_tracking(answer())

        socket = FakeWebSocket(index, on_send)
        sockets.append(socket)
        return socket

    def turn():
        return stream_codex(
            make_model(),
            hello_context(),
            OpenAICodexResponsesOptions(api_key=mock_token(), session_id="shared", transport="auto"),
        )

    with stub_websocket(connect):
        # Cache a socket first: the claim only matters once there is something to
        # claim (two cold turns simply miss the cache and dial one socket each).
        await turn().result()
        overlapping = [turn(), turn()]
        results = [await stream.result() for stream in overlapping]
        # The one-shot must not have taken over the cache entry: the next turn
        # goes back to the socket that was cached, not to the extra connection.
        await turn().result()

    assert [result.stop_reason for result in results] == ["stop", "stop"]
    # One turn reuses the cached socket, the other is handed a one-shot.
    assert len(sockets) == 2
    assert [len(socket.sent) for socket in sockets] == [3, 1]
    stats = get_openai_codex_websocket_debug_stats("shared")
    assert (stats.connections_created, stats.connections_reused) == (2, 2)


@pytest.mark.tonio
async def test_skips_the_websocket_entirely_once_a_session_has_fallen_back():
    """pidrei-only: pi has no case for its own `websocketDisabledForSession`.

    The counter is bumped twice for two requests but the socket is only dialled
    once — the second request never reaches the transport.
    """

    def connect(_url, _headers, _index):
        raise RuntimeError("connect refused")

    client = sse_client()
    options = {
        "api_key": mock_token(),
        "session_id": "sticky-fallback",
        "transport": "auto",
        "client": client,
    }
    with stub_websocket(connect) as calls:
        first = await stream_codex(make_model(), hello_context(), OpenAICodexResponsesOptions(**options)).result()
        stats_after_first = get_openai_codex_websocket_debug_stats("sticky-fallback")
        second = await stream_codex(make_model(), hello_context(), OpenAICodexResponsesOptions(**options)).result()

    assert text_of(first) == "Hello", first.error_message
    assert text_of(second) == "Hello", second.error_message
    assert stats_after_first.sse_fallbacks == 1
    assert stats_after_first.websocket_failures == 1
    assert len(calls) == 1
    assert client.call_count == 2

    stats = get_openai_codex_websocket_debug_stats("sticky-fallback")
    assert stats.sse_fallbacks == 2
    assert stats.websocket_failures == 1
    assert stats.websocket_fallback_active is True


# --- URL / header units -------------------------------------------------------


def test_resolves_codex_urls():
    assert codex._resolve_codex_url(None) == "https://chatgpt.com/backend-api/codex/responses"
    assert codex._resolve_codex_url("https://host/backend-api/") == "https://host/backend-api/codex/responses"
    assert codex._resolve_codex_url("https://host/codex") == "https://host/codex/responses"
    assert codex._resolve_codex_url("https://host/codex/responses") == "https://host/codex/responses"
    assert codex._resolve_codex_websocket_url("https://host/codex") == "wss://host/codex/responses"
    assert codex._resolve_codex_websocket_url("http://host/codex") == "ws://host/codex/responses"


def test_rejects_a_token_without_an_account_id():
    with pytest.raises(RuntimeError, match="Failed to extract accountId from token"):
        codex._extract_account_id("not-a-jwt")


@pytest.mark.tonio
async def test_requires_an_api_key():
    result = await stream_codex(make_model(), hello_context(), OpenAICodexResponsesOptions()).result()
    assert result.stop_reason == "error"
    assert result.error_message == "No API key for provider: openai-codex"


# --- live e2e (skipped without a ChatGPT subscription, as in pi) ---------------


@pytest.mark.skip(reason="pi: it.skipIf(!codexToken) - requires a live ChatGPT Plus/Pro token")
@pytest.mark.tonio
async def test_handles_sse_requests_with_aligned_cache_affinity_identifiers():  # pragma: no cover
    raise AssertionError("live-only")
