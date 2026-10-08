"""Mirror of pi mcp src/transports/streamable-http.ts.

pi's transport hands every fetch one `AbortSignal`, which its close aborts.
Here the close ends what the transport has in flight, each by its own
means:

- a request still waiting for its response head is cancelled: every request
  runs under `run_cancellable` with a token the close fires, so it is
  unwound where it is parked and fails as closed (its connection is
  discarded). One that starts after the close is not sent;
- every response the transport holds (the server-to-client stream, the SSE
  streams answering requests) is closed, which closes its connection: a
  read parked on it ends with a read error (httpunk ends a parked read only
  by closing the socket), and its reader sees the transport closed and
  returns;
- a reconnect waiting out its backoff is woken;
- the session ends with a DELETE, spawned at once and joined at the end,
  bounded by its own one-second request timeout.

The DELETE and the closing of the responses run on their own coroutines,
which the close waits for: a caller that is cancelled stops waiting, and
both complete.

Only the close cancels. A request the client gives up on (its token, its
timeout) keeps its exchange, as in pi: the server is told with
`notifications/cancelled`. The auth provider's work (its token, its handling
of a 401) is not under the close's token either: a token refresh must not
be torn.

A response is registered as soon as its head arrives and is closed by
whoever unregisters it: its reader when done with it, or the close.

Every message sent is an exchange of its own, started on its own coroutine:
nothing orders two of them (see `transport.py`).
"""

import codecs
import re
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_http import http
from pidrei_utils import clock
from pidrei_utils.cancel import CancelToken, run_cancellable

from ..auth_provider import AuthProvider, UnauthorizedContext
from ..fetch import McpFetch, McpResponse, default_fetch
from ..protocol.jsonrpc import (
    JSON_RPC_ERROR_CODES,
    JsonRpcId,
    JsonRpcMessage,
    McpConnectionClosedError,
    is_json_rpc_request,
    is_json_rpc_response,
    parse_json,
    parse_json_rpc_message,
    stringify,
)
from ..url import parse_url
from .transport import DEFAULT_MAX_MESSAGE_BYTES, SendResult, TransportEvents


MAX_ERROR_BODY_BYTES = 8 * 1024
ERROR_MESSAGE_BODY_CHARS = 500
DEFAULT_RECONNECT_INITIAL_DELAY_MS = 1_000
DEFAULT_RECONNECT_MAX_DELAY_MS = 30_000
DEFAULT_RECONNECT_MAX_RETRIES = 5
_DELETE_TIMEOUT_MS = 1_000
_RETRY_FIELD = re.compile(r"[0-9]+")
_INSUFFICIENT_SCOPE = re.compile(r'(?:^|[\s,])error="?insufficient_scope"?', re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class SseEvent:
    data: str
    event: str | None = None
    id: str | None = None


async def consume_sse_stream(
    chunks: AsyncIterator[bytes],
    *,
    on_event: Callable[[SseEvent], Awaitable[None]],
    on_id: Callable[[str], Awaitable[None]] | None = None,
    on_retry: Callable[[int], Awaitable[None]] | None = None,
    max_event_bytes: int | None = None,
) -> None:
    """Decode an SSE body. `on_id` sees every `id` field, including events
    without data (resumption priming events); `on_retry` every valid `retry`
    field, in milliseconds. The owner of `chunks` closes it."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    limit = max_event_bytes or DEFAULT_MAX_MESSAGE_BYTES
    buffered = ""
    event_name: str | None = None
    event_id: str | None = None
    data_lines: list[str] = []
    # Bytes of the pending event's data, including the "\n" joins, so events
    # streamed as many short `data:` lines without a terminating blank line
    # cannot grow without bound.
    data_bytes = 0

    async def dispatch() -> None:
        nonlocal event_name, event_id, data_lines, data_bytes
        if not data_lines:
            event_name = None
            event_id = None
            return
        event = SseEvent(data="\n".join(data_lines), event=event_name or None, id=event_id or None)
        event_name = None
        event_id = None
        data_lines = []
        data_bytes = 0
        await on_event(event)

    async def process_line(raw_line: str) -> None:
        nonlocal event_name, event_id, data_bytes
        line = raw_line.removesuffix("\r")
        if line == "":
            await dispatch()
            return
        if line.startswith(":"):
            return
        field, colon, value = line.partition(":")
        if not colon:
            value = ""
        value = value.removeprefix(" ")
        if field == "data":
            data_bytes += len(value.encode("utf-8")) + (1 if data_lines else 0)
            if data_bytes > limit:
                raise RuntimeError(f"MCP SSE event exceeds {limit} bytes")
            data_lines.append(value)
        elif field == "event":
            event_name = value
        elif field == "id" and "\0" not in value:
            event_id = value
            if on_id is not None:
                await on_id(value)
        elif field == "retry" and _RETRY_FIELD.fullmatch(value):
            if on_retry is not None:
                await on_retry(int(value))

    async for chunk in chunks:
        buffered += decoder.decode(chunk)
        newline = buffered.find("\n")
        while newline >= 0:
            await process_line(buffered[:newline])
            buffered = buffered[newline + 1 :]
            newline = buffered.find("\n")
        if len(buffered.encode("utf-8")) > limit:
            raise RuntimeError(f"MCP SSE event exceeds {limit} bytes")
    buffered += decoder.decode(b"", final=True)
    if buffered:
        await process_line(buffered)
    await dispatch()


@dataclass(frozen=True, slots=True)
class StreamableHttpReconnectOptions:
    """Reconnection of dropped SSE streams (the GET stream, and response
    streams that carry event IDs)."""

    # Delay before the first reconnection attempt, unless the server sent a `retry` field. Default: 1000.
    initial_delay_ms: float | None = None
    # Upper bound for the exponential backoff. Default: 30000.
    max_delay_ms: float | None = None
    # Consecutive failed attempts before giving up on a stream. Default: 5.
    max_retries: int | None = None


@dataclass(frozen=True, slots=True)
class StreamableHttpTransportOptions:
    url: str
    headers: dict[str, str] | None = None
    fetch: McpFetch | None = None
    # Open the server-to-client GET stream after initialization. Default: true.
    open_get_stream: bool = True
    max_message_bytes: int | None = None
    auth_provider: AuthProvider | None = None
    reconnect: StreamableHttpReconnectOptions | None = None


class McpHttpError(Exception):
    def __init__(self, status: int, message: str, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class McpAuthRequiredError(McpHttpError):
    def __init__(self, response: McpResponse, body: str = "") -> None:
        super().__init__(401, "MCP server requires authentication", body)
        self.www_authenticate = response.headers.get("www-authenticate")


class McpSessionExpiredError(McpHttpError):
    def __init__(self, body: str = "") -> None:
        super().__init__(404, "MCP session expired", body)


def _content_type(response: McpResponse) -> str | None:
    value = response.headers.get("content-type")
    return None if value is None else value.split(";", 1)[0].strip().lower()


def _needs_authorization(response: McpResponse) -> bool:
    """401, or 403 with an `insufficient_scope` bearer challenge (step-up authorization)."""
    if response.status == 401:
        return True
    if response.status != 403:
        return False
    return _INSUFFICIENT_SCOPE.search(response.headers.get("www-authenticate") or "") is not None


def _is_transient_status(status: int) -> bool:
    """Statuses worth retrying when a stream fails to (re)open."""
    return status in (408, 429) or status >= 500


def _describe_http_failure(status: int, body: str) -> str:
    text = body.strip()
    snippet = f"{text[: ERROR_MESSAGE_BODY_CHARS - 3]}..." if len(text) > ERROR_MESSAGE_BODY_CHARS else text
    return f"MCP HTTP request failed with status {status}{f': {snippet}' if snippet else ''}"


class _StreamCursor:
    __slots__ = ("last_event_id", "received", "retry_ms")

    def __init__(self) -> None:
        self.last_event_id: str | None = None
        self.retry_ms: int | None = None
        # Whether the stream delivered any event since it was (re)opened.
        self.received = False


class StreamableHttpTransport(TransportEvents):
    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        fetch: McpFetch | None = None,
        open_get_stream: bool = True,
        max_message_bytes: int | None = None,
        auth_provider: AuthProvider | None = None,
        reconnect: StreamableHttpReconnectOptions | None = None,
    ) -> None:
        super().__init__()
        self.options = StreamableHttpTransportOptions(
            url=url,
            headers=dict(headers) if headers is not None else None,
            fetch=fetch,
            open_get_stream=open_get_stream,
            max_message_bytes=max_message_bytes,
            auth_provider=auth_provider,
            reconnect=reconnect,
        )
        self.url = parse_url(url).href
        self._fetch: McpFetch = fetch if fetch is not None else default_fetch
        self._lock = threading.Lock()
        # Set by `close()`: wakes a reconnect waiting out its backoff.
        self._closed_event = tonio.Event()
        # Fired by `close()`: ends every request still waiting for its response head.
        self._close_token = CancelToken()
        self._responses: set[Any] = set()
        self._started = False
        self._closed = False
        self._session_id: str | None = None
        self._protocol_version: str | None = None
        self._get_stream_started = False
        # Access token of the latest request, which `close()` reuses instead of asking the auth provider.
        self._last_token: str | None = None

    @property
    def session_id(self) -> str | None:
        return self._session_id

    async def start(self) -> None:
        with self._lock:
            if self._started:
                raise RuntimeError("MCP Streamable HTTP transport already started")
            if self._closed:
                raise McpConnectionClosedError()
            self._started = True

    def set_protocol_version(self, version: str) -> None:
        self._protocol_version = version

    def send(self, message: JsonRpcMessage) -> SendResult:
        with self._lock:
            if not self._started or self._closed:
                return SendResult.failed(McpConnectionClosedError())
        result = SendResult()
        tonio.spawn.without_tracking(self._post(message, result))
        return result

    async def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            delete_session = self._started and self._session_id is not None
            responses = list(self._responses)
            self._responses.clear()
        self._closed_event.set()
        self._close_token.cancel(McpConnectionClosedError())
        # Each response is closed on its own coroutine, and the close waits
        # for them. The responses are out of the transport's reach by now, so
        # a caller that is cancelled must only stop waiting: nothing else
        # would close them. The DELETE touches no transport state: it runs
        # beside the teardown, and the close waits for it too (up to its
        # timeout), as pi waits for it.
        closing = tonio.spawn(*(_close_quietly(response) for response in responses)) if responses else None
        deleting = tonio.spawn(self._delete_session()) if delete_session else None
        try:
            if closing is not None:
                await closing
            if deleting is not None:
                await deleting
        finally:
            self._emit_close()

    async def _delete_session(self) -> None:
        # Best effort: the session expires on the server anyway. The auth provider may refresh tokens
        # over the network, so it is not asked here and closing never waits for a refresh.
        try:
            headers = self._build_headers(None, self._last_token)
            response = await self._fetch(self.url, method="DELETE", headers=headers, timeout_ms=_DELETE_TIMEOUT_MS)
            await _close_quietly(response)
        except Exception:
            pass

    async def _post(self, message: JsonRpcMessage, result: SendResult) -> None:
        try:
            await self._send_message(message)
        except Exception as error:
            result.settle(error)
        else:
            result.settle()

    async def _send_message(self, message: JsonRpcMessage) -> None:
        response = await self._authorized_fetch(
            "POST",
            {"accept": "application/json, text/event-stream", "content-type": "application/json"},
            stringify(message).encode("utf-8"),
        )
        await self._check_response(response)
        self._capture_session(response)

        if not is_json_rpc_request(message):
            # Notifications and responses are acknowledged with 202 and carry no reply; ignore any body.
            await self._discard(response)
            # The server-to-client stream may only open once the session is initialized.
            if message.get("method") == "notifications/initialized":
                self._start_get_stream()
            return
        if response.status in (202, 204):
            await self._discard(response)
            raise McpHttpError(response.status, f"MCP server accepted request {message['method']} without a response")
        content_type = _content_type(response)
        if content_type == "application/json":
            try:
                body = parse_json(await response.read())
            finally:
                await self._discard(response)
            for item in body if isinstance(body, list) else [body]:
                self._emit_message(parse_json_rpc_message(item))
            return
        if content_type == "text/event-stream":
            tonio.spawn.without_tracking(self._consume_response_stream(response, message["id"]))
            return
        await self._discard(response)
        raise McpHttpError(response.status, f"Unsupported MCP response content type: {content_type or 'missing'}")

    async def _open(self, method: str, headers: dict[str, str], body: bytes | None) -> McpResponse:
        """One request, its response registered with the transport. The close
        cancels it while it waits for the response head, and one that starts
        after the close is not sent."""
        response = await run_cancellable(
            self._fetch(self.url, method=method, headers=headers, body=body),  # type: ignore[arg-type]
            self._close_token,
        )
        with self._lock:
            closed = self._closed
            if not closed:
                self._responses.add(response)
        if closed:
            await _close_quietly(response)
            raise McpConnectionClosedError()
        return response

    async def _discard(self, response: McpResponse) -> None:
        """Release a response the transport holds; the close may have taken it already."""
        with self._lock:
            owned = response in self._responses
            self._responses.discard(response)
        if owned:
            await _close_quietly(response)

    async def _authorized_fetch(self, method: str, extra: dict[str, str], body: bytes | None) -> McpResponse:
        """Fetch with auth headers. A 401 (or a 403 asking for more scope) is
        handed to the auth provider once, and the request is retried with
        whatever credentials it left behind."""
        provider = self.options.auth_provider
        on_unauthorized = provider.on_unauthorized if provider is not None else None
        attempt = 0
        while True:
            headers, token = await self._headers(extra)
            response = await self._open(method, headers, body)
            if attempt > 0 or on_unauthorized is None or not _needs_authorization(response):
                return response
            try:
                await on_unauthorized(
                    UnauthorizedContext(response=response, server_url=self.url, fetch=self._fetch, token=token)
                )
            finally:
                await self._discard(response)
            attempt += 1

    async def _headers(self, extra: dict[str, str] | None = None) -> tuple[dict[str, str], str | None]:
        provider = self.options.auth_provider
        token = await provider.token() if provider is not None else None
        self._last_token = token
        return self._build_headers(extra, token), token or None

    def _build_headers(self, extra: dict[str, str] | None, token: str | None) -> dict[str, str]:
        # Header names are case-insensitive: lower-cased keys make each `set` a replace, as `Headers.set` is.
        headers = {name.lower(): value for name, value in (self.options.headers or {}).items()}
        for name, value in (extra or {}).items():
            headers[name.lower()] = value
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        if self._protocol_version:
            headers["mcp-protocol-version"] = self._protocol_version
        if token:
            headers["authorization"] = f"Bearer {token}"
        return headers

    def _capture_session(self, response: McpResponse) -> None:
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

    async def _check_response(self, response: McpResponse) -> None:
        if 200 <= response.status < 300:
            return
        try:
            body = (await response.read()).decode("utf-8", "replace")[:MAX_ERROR_BODY_BYTES]
        except Exception:
            body = ""
        await self._discard(response)
        if response.status == 401:
            raise McpAuthRequiredError(response, body)
        if response.status == 404 and self._session_id:
            raise McpSessionExpiredError(body)
        raise McpHttpError(response.status, _describe_http_failure(response.status, body), body)

    async def _consume_sse(
        self,
        response: McpResponse,
        cursor: _StreamCursor,
        on_message: Callable[[JsonRpcMessage], None] | None = None,
    ) -> None:
        async def on_id(event_id: str) -> None:
            cursor.last_event_id = event_id

        async def on_retry(delay_ms: int) -> None:
            cursor.retry_ms = delay_ms

        async def on_event(event: SseEvent) -> None:
            cursor.received = True
            # Events without data prime resumption; other event types are not JSON-RPC.
            if not event.data.strip() or (event.event is not None and event.event != "message"):
                return
            try:
                message = parse_json_rpc_message(parse_json(event.data))
            except Exception as error:
                self._emit_error(error)
                return
            if on_message is not None:
                on_message(message)
            self._emit_message(message)

        await consume_sse_stream(
            response.iter_bytes(),
            on_event=on_event,
            on_id=on_id,
            on_retry=on_retry,
            max_event_bytes=self.options.max_message_bytes or DEFAULT_MAX_MESSAGE_BYTES,
        )

    async def _consume_response_stream(self, response: McpResponse, request_id: JsonRpcId) -> None:
        """Read the SSE stream answering one request. When the stream ends or
        breaks before the response arrives and the server assigned event IDs,
        resume it with GET and `Last-Event-ID`, as the server may close
        response streams at will. Otherwise only this request fails."""
        cursor = _StreamCursor()
        answered = False

        def on_message(message: JsonRpcMessage) -> None:
            nonlocal answered
            if is_json_rpc_response(message) and message["id"] == request_id:
                answered = True

        stream: McpResponse | None = response
        failure: Exception | None = None
        attempt = 0
        while True:
            if stream is not None:
                try:
                    await self._consume_sse(stream, cursor, on_message)
                    failure = None
                except Exception as error:
                    failure = error
                await self._discard(stream)
            if answered or self._closed:
                return
            if failure is not None and not self._is_retryable(failure):
                break
            if cursor.last_event_id is None or attempt >= self._max_retries():
                break
            if cursor.received:
                attempt = 0
            cursor.received = False
            delay = self._reconnect_delay(attempt, cursor.retry_ms)
            attempt += 1
            if not await self._sleep(delay):
                return
            try:
                stream = await self._open_sse_stream(cursor.last_event_id)
            except Exception as error:
                failure = error
                if not self._is_retryable(error):
                    break
                stream = None
        if self._closed:
            return
        reason = "stream ended without a response" if failure is None else str(failure)
        self._emit_message(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": JSON_RPC_ERROR_CODES.internal_error,
                    "message": f"MCP response stream failed: {reason}",
                },
            }
        )

    def _start_get_stream(self) -> None:
        with self._lock:
            if not self.options.open_get_stream or self._get_stream_started or self._closed:
                return
            self._get_stream_started = True
        tonio.spawn.without_tracking(self._run_get_stream())

    async def _run_get_stream(self) -> None:
        """Keep the server-to-client stream open, reconnecting with backoff when it drops."""
        cursor = _StreamCursor()
        attempt = 0
        while not self._closed:
            try:
                stream = await self._open_sse_stream(cursor.last_event_id)
                # The server does not offer a GET stream.
                if stream is None:
                    return
                opened_at = clock.monotonic()
                try:
                    await self._consume_sse(stream, cursor)
                except Exception:
                    await self._discard(stream)
                    raise
                await self._discard(stream)
                # A stream that stayed up for a while counts as healthy, even if it was idle.
                if cursor.received or (clock.monotonic() - opened_at) * 1000 > self._max_delay():
                    attempt = 0
            except Exception as error:
                if self._closed:
                    return
                if not self._is_retryable(error):
                    self._emit_error(error)
                    return
            cursor.received = False
            if attempt >= self._max_retries():
                self._emit_error(RuntimeError("MCP server-to-client stream dropped and could not be reopened"))
                return
            delay = self._reconnect_delay(attempt, cursor.retry_ms)
            attempt += 1
            if not await self._sleep(delay):
                return

    async def _open_sse_stream(self, last_event_id: str | None) -> McpResponse | None:
        """Open a GET SSE stream; None when the server answers 405 (no GET stream)."""
        headers = {"accept": "text/event-stream"}
        if last_event_id is not None:
            headers["last-event-id"] = last_event_id
        response = await self._authorized_fetch("GET", headers, None)
        if response.status == 405:
            await self._discard(response)
            return None
        await self._check_response(response)
        self._capture_session(response)
        content_type = _content_type(response)
        if content_type != "text/event-stream":
            await self._discard(response)
            raise McpHttpError(
                response.status, f"Unsupported MCP GET response content type: {content_type or 'missing'}"
            )
        return response

    def _is_retryable(self, error: Exception) -> bool:
        """Network failures and transient statuses are retried; auth, session,
        and protocol errors are not. A connection dropped mid-body is a
        network failure too."""
        if isinstance(error, McpHttpError):
            return _is_transient_status(error.status)
        return isinstance(error, http.TransportError | OSError)

    def _reconnect_delay(self, attempt: int, server_delay_ms: int | None) -> float:
        if server_delay_ms is not None:
            return server_delay_ms
        reconnect = self.options.reconnect
        initial = DEFAULT_RECONNECT_INITIAL_DELAY_MS
        if reconnect is not None and reconnect.initial_delay_ms is not None:
            initial = reconnect.initial_delay_ms
        return min(initial * 2**attempt, self._max_delay())

    def _max_delay(self) -> float:
        reconnect = self.options.reconnect
        if reconnect is not None and reconnect.max_delay_ms is not None:
            return reconnect.max_delay_ms
        return DEFAULT_RECONNECT_MAX_DELAY_MS

    def _max_retries(self) -> int:
        reconnect = self.options.reconnect
        if reconnect is not None and reconnect.max_retries is not None:
            return reconnect.max_retries
        return DEFAULT_RECONNECT_MAX_RETRIES

    async def _sleep(self, ms: float) -> bool:
        """False when the transport closed while waiting."""
        await self._closed_event.wait(ms / 1000)
        return not self._closed_event.is_set()


async def _close_quietly(response: McpResponse) -> None:
    try:
        await response.close()
    except Exception:
        pass
