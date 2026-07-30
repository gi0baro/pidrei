"""Port of pi's Codex adapter (packages/ai/src/api/openai-codex-responses.ts).

Two transports for one endpoint: a WebSocket (pi's default, `transport: "auto"`)
whose connection is cached per session and carries continuations through
`previous_response_id`, and the SSE POST it falls back to. The fallback ladder,
the per-session state and the debug counters are pi's, mirrored field for field.

The WebSocket itself lives in `utils/websocket.py` (pi reaches for the runtime's
global `WebSocket`, which has no equivalent here); everything about *when* to
connect, reconnect, reuse, keep or drop it is here, as in pi.
"""

import base64
import json
import math
import platform
import re
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, fields
from email.utils import parsedate_to_datetime
from typing import Any, Protocol

import tonio.colored as tonio

from pidrei_ai.api.constrained_sampling import create_grammar_tool_input_properties
from pidrei_ai.api.openai_prompt_cache import clamp_openai_prompt_cache_key
from pidrei_ai.api.openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from pidrei_ai.api.simple_options import build_base_options
from pidrei_ai.registry import clamp_thinking_level
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    OpenAIResponsesCompat,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
    Usage,
)
from pidrei_ai.utils import clock, http, websocket
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.cancel import AbortError, CancelToken
from pidrei_ai.utils.deferred_tools import split_deferred_tools
from pidrei_ai.utils.diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    format_thrown_value,
)
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.session_resources import register_session_resource_cleanup
from pidrei_ai.utils.sse import iterate_sse_messages
from pidrei_ai.utils.uuid import uuidv7


try:
    # pi's `loadNodeZlib()`: absent on runtimes without it, and the SSE body then
    # goes out uncompressed. CPython builds zstd in unless libzstd was missing.
    from compression import zstd
except ImportError:  # pragma: no cover - stdlib built without libzstd
    zstd = None  # type: ignore[assignment]


# --- configuration ------------------------------------------------------------

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
DEFAULT_MAX_RETRIES = 0
BASE_DELAY_MS = 1000
DEFAULT_MAX_RETRY_DELAY_MS = 60_000
DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS = 15_000
# The Codex backend accepts zstd-compressed request bodies on the SSE responses
# endpoint (the same endpoint the official Codex client compresses against).
REQUEST_COMPRESSION_ZSTD_LEVEL = 3
CODEX_TOOL_CALL_PROVIDERS = frozenset(("openai", "openai-codex", "opencode"))
WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE = 1009
WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE = "websocket_connection_limit_reached"
PREVIOUS_RESPONSE_NOT_FOUND_CODE = "previous_response_not_found"

CODEX_RESPONSE_STATUSES = frozenset(
    ("completed", "incomplete", "failed", "cancelled", "queued", "in_progress"),
)

OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
SESSION_WEBSOCKET_CACHE_TTL_MS = 5 * 60 * 1000
SESSION_WEBSOCKET_MAX_AGE_MS = 55 * 60 * 1000

_TERMINAL_RATE_LIMIT_PATTERN = re.compile(
    r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|available balance"
    r"|insufficient_quota|out of budget|quota exceeded|billing",
    re.IGNORECASE,
)
_RETRYABLE_TEXT_PATTERN = re.compile(
    r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
    re.IGNORECASE,
)
_USAGE_LIMIT_CODE_PATTERN = re.compile(
    r"usage_limit_reached|usage_not_included|rate_limit_exceeded",
    re.IGNORECASE,
)


# --- types --------------------------------------------------------------------


@dataclass(slots=True)
class OpenAICodexResponsesOptions(StreamOptions):
    reasoning_effort: str | None = None  # "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
    reasoning_summary: str | None = None  # "auto" | "concise" | "detailed" | "off" | "on"
    service_tier: str | None = None
    text_verbosity: str | None = None  # "low" | "medium" | "high"
    tool_choice: str | None = None  # "auto" | "none" | "required"
    # Pre-built SSE client (test injection / alternative transports).
    client: CodexSSEClient | None = None


class CodexSSEResponseLike(Protocol):
    status: int
    headers: dict[str, str]

    def aiter_bytes(self) -> AsyncIterable[bytes]: ...


class CodexSSEClient(Protocol):
    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_ms: float | None,
        cancel: CancelToken | None,
    ) -> CodexSSEResponseLike: ...


class CodexApiError(Exception):
    """A Codex protocol-level error event (pi: `CodexApiError`)."""

    def __init__(self, message: str, *, code: str | None = None, payload: dict | None = None):
        super().__init__(message)
        self.code = code
        self.payload = payload


class CodexProtocolError(Exception):
    """Malformed Codex framing — invalid SSE or WebSocket JSON."""

    def __init__(self, message: str, *, payload: Any = None):
        super().__init__(message)
        self.payload = payload


class RetryDelayExceededError(Exception):
    pass


class WebSocketCloseError(Exception):
    def __init__(
        self, message: str, *, code: int | None = None, reason: str | None = None, was_clean: bool | None = None
    ):
        super().__init__(message)
        self.code = code
        self.reason = reason
        self.was_clean = was_clean


@dataclass(slots=True)
class OpenAICodexWebSocketDebugStats:
    requests: int = 0
    connections_created: int = 0
    connections_reused: int = 0
    cached_context_requests: int = 0
    store_true_requests: int = 0
    full_context_requests: int = 0
    delta_requests: int = 0
    last_input_items: int = 0
    last_delta_input_items: int | None = None
    last_previous_response_id: str | None = None
    websocket_failures: int = 0
    sse_fallbacks: int = 0
    websocket_fallback_active: bool | None = None
    last_websocket_error: str | None = None


@dataclass(slots=True)
class _CachedWebSocketContinuationState:
    last_request_body: dict
    last_response_id: str
    last_response_items: list[dict]


@dataclass(slots=True)
class _CachedWebSocketConnection:
    socket: Any
    busy: bool
    created_at: int
    idle_timer: CancelToken | None = None
    continuation: _CachedWebSocketContinuationState | None = None


# --- retry helpers ------------------------------------------------------------


def _is_terminal_rate_limit_error(error_text: str) -> bool:
    return _TERMINAL_RATE_LIMIT_PATTERN.search(error_text) is not None


def _is_retryable_error(status: int, error_text: str) -> bool:
    if status == 429 and _is_terminal_rate_limit_error(error_text):
        return False
    if status in (429, 500, 502, 503, 504):
        return True
    return _RETRYABLE_TEXT_PATTERN.search(error_text) is not None


def _get_retry_after_delay_ms(headers: dict[str, str]) -> float | None:
    lowered = {key.lower(): value for key, value in (headers or {}).items()}

    retry_after_ms = lowered.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            millis = float(retry_after_ms)
        except ValueError:
            millis = math.nan
        if math.isfinite(millis):
            return max(0.0, millis)

    retry_after = lowered.get("retry-after")
    if not retry_after:
        return None

    try:
        seconds = float(retry_after)
    except ValueError:
        seconds = math.nan
    if math.isfinite(seconds):
        return max(0.0, seconds * 1000)

    try:
        date = parsedate_to_datetime(retry_after)
    except TypeError, ValueError:
        return None
    return max(0.0, date.timestamp() * 1000 - clock.now_ms())


def _validate_retry_delay_ms(delay_ms: float, options: StreamOptions | None) -> float:
    max_retry_delay_ms = (
        options.max_retry_delay_ms
        if options is not None and options.max_retry_delay_ms is not None
        else DEFAULT_MAX_RETRY_DELAY_MS
    )
    if max_retry_delay_ms > 0 and delay_ms > max_retry_delay_ms:
        raise RetryDelayExceededError(
            f"Server requested {math.ceil(delay_ms / 1000)}s retry delay (max: {math.ceil(max_retry_delay_ms / 1000)}s)"
        )
    return delay_ms


async def _sleep(ms: float, cancel: CancelToken | None = None) -> None:
    """pi's `sleep(ms, signal)`: rejects with its own message when aborted."""
    try:
        await clock.sleep_ms(ms, cancel)
    except AbortError as error:
        raise RuntimeError("Request was aborted") from error


def _normalize_timeout_ms(value: float | None) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value) or value < 0:
        raise RuntimeError(f"Invalid timeoutMs: {value}")
    return math.floor(value)


def _compress_request_body_zstd(body_json: str) -> bytes | None:
    """Returns the zstd-compressed body bytes, or None when unavailable."""
    if zstd is None:
        return None
    try:
        return zstd.compress(body_json.encode(), level=REQUEST_COMPRESSION_ZSTD_LEVEL)
    except Exception:
        return None


# --- SSE transport ------------------------------------------------------------


@dataclass(slots=True)
class _PunkreqCodexResponse:
    status: int
    headers: dict[str, str]
    _response: Any

    def aiter_bytes(self) -> AsyncIterable[bytes]:
        return self._response.iter_bytes()

    async def read_text(self) -> str:
        return (await self._response.read()).decode("utf-8", "replace")

    async def close(self) -> None:
        await self._response.close()


class _PunkreqCodexClient:
    """Default SSE transport: POST the Codex responses endpoint via punkreq."""

    def __init__(self, env: ProviderEnv | None = None):
        self._env = env

    async def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_ms: float | None,
        cancel: CancelToken | None,
    ) -> CodexSSEResponseLike:
        client = http.client_for(url, self._env)
        # pi bounds only the response *head* with `AbortSignal.timeout(timeoutMs)`;
        # punkreq has no head-specific deadline, so this maps to the per-read
        # timeout (as in openai_responses.py) — the head read is the first read.
        response = await client.post(url, content=body, headers=headers, timeout=http.request_timeout(timeout_ms))
        return _PunkreqCodexResponse(
            status=response.status_code,
            headers={key.lower(): value for key, value in dict(response.headers).items()},
            _response=response,
        )


# --- module state -------------------------------------------------------------

_websocket_session_cache: dict[str, _CachedWebSocketConnection] = {}
_websocket_debug_stats: dict[str, OpenAICodexWebSocketDebugStats] = {}
_websocket_sse_fallback_sessions: set[str] = set()
# pi relies on JavaScript's single thread to keep this state consistent; tonio
# runs turns on any worker thread, so the maps take a lock.
_websocket_state_guard = threading.Lock()


def _get_or_create_websocket_debug_stats(session_id: str) -> OpenAICodexWebSocketDebugStats:
    with _websocket_state_guard:
        stats = _websocket_debug_stats.get(session_id)
        if stats is None:
            stats = OpenAICodexWebSocketDebugStats()
            _websocket_debug_stats[session_id] = stats
        return stats


def get_openai_codex_websocket_debug_stats(session_id: str) -> OpenAICodexWebSocketDebugStats | None:
    with _websocket_state_guard:
        stats = _websocket_debug_stats.get(session_id)
        return (
            OpenAICodexWebSocketDebugStats(**{f.name: getattr(stats, f.name) for f in fields(stats)}) if stats else None
        )


def reset_openai_codex_websocket_debug_stats(session_id: str | None = None) -> None:
    with _websocket_state_guard:
        if session_id:
            _websocket_debug_stats.pop(session_id, None)
            _websocket_sse_fallback_sessions.discard(session_id)
            return
        _websocket_debug_stats.clear()
        _websocket_sse_fallback_sessions.clear()


def close_openai_codex_websocket_sessions(session_id: str | None = None) -> None:
    def close_entry(entry: _CachedWebSocketConnection) -> None:
        if entry.idle_timer is not None:
            entry.idle_timer.cancel()
        _close_websocket_silently(entry.socket, 1000, "debug_close")

    with _websocket_state_guard:
        if session_id:
            entries = [_websocket_session_cache.pop(session_id)] if session_id in _websocket_session_cache else []
        else:
            entries = list(_websocket_session_cache.values())
            _websocket_session_cache.clear()
    for entry in entries:
        close_entry(entry)


register_session_resource_cleanup(close_openai_codex_websocket_sessions)


def _is_websocket_sse_fallback_active(session_id: str | None) -> bool:
    if not session_id:
        return False
    with _websocket_state_guard:
        return session_id in _websocket_sse_fallback_sessions


def _record_websocket_sse_fallback(session_id: str | None) -> None:
    if not session_id:
        return
    stats = _get_or_create_websocket_debug_stats(session_id)
    with _websocket_state_guard:
        stats.sse_fallbacks += 1
        stats.websocket_fallback_active = session_id in _websocket_sse_fallback_sessions


def _record_websocket_failure(session_id: str | None, error: Any) -> None:
    if not session_id:
        return
    stats = _get_or_create_websocket_debug_stats(session_id)
    with _websocket_state_guard:
        _websocket_sse_fallback_sessions.add(session_id)
        stats.websocket_failures += 1
        stats.last_websocket_error = format_thrown_value(error)
        stats.websocket_fallback_active = True


# --- main stream function -----------------------------------------------------


def _codex_options(options: StreamOptions | None) -> OpenAICodexResponsesOptions:
    if isinstance(options, OpenAICodexResponsesOptions):
        return options
    if options is None:
        return OpenAICodexResponsesOptions()
    values = {f.name: getattr(options, f.name) for f in fields(StreamOptions)}
    return OpenAICodexResponsesOptions(**values)


def _assert_successful_output(output: AssistantMessage) -> None:
    """pi's `assertSuccessfulOutput`: a done event may only carry stop/length/toolUse."""
    if output.stop_reason == "pending":
        raise RuntimeError("Codex stream ended without a stop reason")
    if output.stop_reason in ("error", "aborted"):
        raise RuntimeError(output.error_message or "An unknown error occurred")


def stream(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    opts = _codex_options(options)
    out_stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            content=[],
            api="openai-codex-responses",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )
        state = _StartState()

        try:
            api_key = opts.api_key
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")

            account_id = _extract_account_id(api_key)
            compat = _get_compat(model)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                context.tools, compat.supports_openai_grammar_tools
            )
            cache_session_id = None if opts.cache_retention == "none" else opts.session_id
            codex_session_id = clamp_openai_prompt_cache_key(cache_session_id)
            body = build_request_body(model, context, opts, codex_session_id, grammar_tool_input_properties)
            next_body = await maybe_call(opts.on_payload, body, model)
            if next_body is not None:
                body = next_body
            websocket_request_id = codex_session_id or uuidv7()
            sse_headers = _build_sse_headers(model.headers, opts.headers, account_id, api_key, codex_session_id)
            websocket_headers = _build_websocket_headers(
                model.headers, opts.headers, account_id, api_key, websocket_request_id
            )
            body_json = json.dumps(body, separators=(",", ":"))
            http_timeout_ms = _normalize_timeout_ms(opts.timeout_ms)
            websocket_connect_timeout_ms = _normalize_timeout_ms(opts.websocket_connect_timeout_ms)
            transport = opts.transport or "auto"
            websocket_disabled_for_session = transport != "sse" and _is_websocket_sse_fallback_active(cache_session_id)
            if websocket_disabled_for_session:
                _record_websocket_sse_fallback(cache_session_id)

            if transport != "sse" and not websocket_disabled_for_session:
                retried_websocket_connection_limit = False
                retried_missing_websocket_continuation = False
                while True:
                    websocket_started = _Flag()
                    try:

                        def _on_start(started=websocket_started) -> None:
                            started.value = True
                            if not state.emitted:
                                state.emitted = True
                                out_stream.push(StartEvent(partial=output))

                        await _process_websocket_stream(
                            _resolve_codex_websocket_url(model.base_url),
                            body,
                            websocket_headers,
                            output,
                            out_stream,
                            model,
                            _on_start,
                            http_timeout_ms,
                            websocket_connect_timeout_ms,
                            cache_session_id,
                            grammar_tool_input_properties,
                            opts,
                        )

                        if opts.cancel is not None and opts.cancel.cancelled:
                            raise RuntimeError("Request was aborted")
                        _assert_successful_output(output)
                        out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
                        out_stream.end()
                        return
                    except Exception as error:
                        aborted = opts.cancel is not None and opts.cancel.cancelled
                        connection_limit_before_start = (
                            not websocket_started.value and _is_websocket_connection_limit_reached_error(error)
                        )
                        previous_response_not_found = _is_previous_response_not_found_error(error)
                        if not aborted and previous_response_not_found and not retried_missing_websocket_continuation:
                            retried_missing_websocket_continuation = True
                            continue
                        if not aborted and connection_limit_before_start and not retried_websocket_connection_limit:
                            retried_websocket_connection_limit = True
                            continue
                        if aborted or (_is_codex_non_transport_error(error) and not connection_limit_before_start):
                            raise
                        # Building the diagnostic renders a traceback, and
                        # `traceback.format_exception` reads the source files to
                        # do it — real filesystem I/O on an error path.
                        diagnostic = await tonio.spawn_blocking(
                            create_assistant_message_diagnostic,
                            "provider_transport_failure",
                            error,
                            {
                                "configuredTransport": transport,
                                "fallbackTransport": None if websocket_started.value else "sse",
                                "eventsEmitted": websocket_started.value,
                                "phase": "after_message_stream_start"
                                if websocket_started.value
                                else "before_message_stream_start",
                                "requestBytes": len(body_json.encode()),
                            },
                        )
                        append_assistant_message_diagnostic(output, diagnostic)
                        _record_websocket_failure(cache_session_id, error)
                        if websocket_started.value:
                            raise
                        _record_websocket_sse_fallback(cache_session_id)
                        break

            # Compress the request body once for the SSE path. The Codex backend
            # decodes Content-Encoding: zstd; the WebSocket transport above sends
            # the uncompressed JSON frame, matching the official Codex client.
            compressed_body = _compress_request_body_zstd(body_json)
            if compressed_body is not None:
                sse_headers["content-encoding"] = "zstd"
            sse_body = compressed_body if compressed_body is not None else body_json.encode()

            client = opts.client if opts.client is not None else _PunkreqCodexClient(opts.env)
            url = _resolve_codex_url(model.base_url)
            response: CodexSSEResponseLike | None = None
            last_error: Exception | None = None
            max_retries = opts.max_retries if opts.max_retries is not None else DEFAULT_MAX_RETRIES

            for attempt in range(max_retries + 1):
                if opts.cancel is not None and opts.cancel.cancelled:
                    raise RuntimeError("Request was aborted")
                try:
                    try:
                        response = await client.post(
                            url,
                            headers=sse_headers,
                            body=sse_body,
                            timeout_ms=http_timeout_ms,
                            cancel=opts.cancel,
                        )
                    except http.RequestTimeout as error:
                        if opts.cancel is None or not opts.cancel.cancelled:
                            raise RuntimeError(
                                f"Codex SSE response headers timed out after {_format_ms(http_timeout_ms)}ms"
                            ) from error
                        raise
                    await maybe_call(
                        opts.on_response, ProviderResponse(status=response.status, headers=response.headers), model
                    )

                    if 200 <= response.status < 300:
                        break

                    error_text = await _read_error_text(response)
                    if attempt < max_retries and _is_retryable_error(response.status, error_text):
                        retry_after_delay_ms = _get_retry_after_delay_ms(response.headers)
                        delay_ms = (
                            BASE_DELAY_MS * 2**attempt
                            if retry_after_delay_ms is None
                            else _validate_retry_delay_ms(retry_after_delay_ms, opts)
                        )
                        await _sleep(delay_ms, opts.cancel)
                        continue

                    info = _parse_error_response(error_text, response.status)
                    raise RuntimeError(info[1] or info[0])
                except Exception as error:
                    if isinstance(error, AbortError) or str(error) == "Request was aborted":
                        raise RuntimeError("Request was aborted") from error
                    last_error = error
                    # Network errors are retryable
                    if (
                        attempt < max_retries
                        and not isinstance(error, RetryDelayExceededError)
                        and "usage limit" not in str(error)
                    ):
                        await _sleep(BASE_DELAY_MS * 2**attempt, opts.cancel)
                        continue
                    raise last_error from None

            if response is None or not 200 <= response.status < 300:
                raise last_error if last_error is not None else RuntimeError("Failed after retries")

            if not state.emitted:
                state.emitted = True
                out_stream.push(StartEvent(partial=output))
            await _process_stream(response, output, out_stream, model, grammar_tool_input_properties, opts)

            if opts.cancel is not None and opts.cancel.cancelled:
                raise RuntimeError("Request was aborted")

            _assert_successful_output(output)
            out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            out_stream.end()
        except Exception as error:
            for block in output.content:
                # Streaming scratch buffers are only used during parsing; never persist them.
                for scratch in ("partial_json", "custom_input"):
                    if hasattr(block, scratch):
                        try:
                            delattr(block, scratch)
                        except AttributeError:
                            pass
            output.stop_reason = "aborted" if opts.cancel is not None and opts.cancel.cancelled else "error"
            output.error_message = format_provider_error(normalize_provider_error(error))
            out_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
            out_stream.end()

    tonio.spawn.without_tracking(_run())
    return out_stream


@dataclass(slots=True)
class _StartState:
    emitted: bool = False


@dataclass(slots=True)
class _Flag:
    value: bool = False


def _format_ms(value: float | None) -> str:
    if value is None:
        return "None"
    return str(int(value)) if float(value).is_integer() else str(value)


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    api_key = options.api_key if options else None
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    opts = _codex_options(base)
    opts.reasoning_effort = reasoning_effort
    return stream(model, context, opts)


# --- request building ---------------------------------------------------------


@dataclass(slots=True)
class _ResolvedCompat:
    supports_strict_mode: bool
    supports_openai_grammar_tools: bool
    supports_tool_search: bool


def _get_compat(model: Model) -> _ResolvedCompat:
    compat = model.compat if isinstance(model.compat, OpenAIResponsesCompat) else None

    def pick(value, default):
        return value if value is not None else default

    return _ResolvedCompat(
        supports_strict_mode=pick(compat.supports_strict_mode if compat else None, True),
        supports_openai_grammar_tools=pick(compat.supports_openai_grammar_tools if compat else None, False),
        supports_tool_search=pick(compat.supports_tool_search if compat else None, False),
    )


def build_request_body(
    model: Model,
    context: Context,
    options: OpenAICodexResponsesOptions | None,
    cache_session_id: str | None,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict:
    compat = _get_compat(model)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )

    immediate_tools, deferred_map = split_deferred_tools(context, compat.supports_tool_search)
    messages = convert_responses_messages(
        model,
        context,
        set(CODEX_TOOL_CALL_PROVIDERS),
        include_system_prompt=False,
        grammar_tool_input_properties=grammar_tool_input_properties,
        deferred_tools=deferred_map,
        tool_options={
            "strict": None,
            "supports_strict_mode": compat.supports_strict_mode,
            "supports_openai_grammar_tools": compat.supports_openai_grammar_tools,
        },
    )

    body: dict[str, Any] = {
        "model": model.id,
        "store": False,
        "stream": True,
        "instructions": context.system_prompt or "You are a helpful assistant.",
        "input": messages,
        "text": {"verbosity": (options.text_verbosity if options else None) or "low"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": (options.tool_choice if options else None) or "auto",
        "parallel_tool_calls": True,
    }
    if cache_session_id is not None:
        body["prompt_cache_key"] = cache_session_id

    if options is not None and options.temperature is not None:
        body["temperature"] = options.temperature

    if options is not None and options.service_tier is not None:
        body["service_tier"] = options.service_tier

    if immediate_tools:
        body["tools"] = convert_responses_tools(
            immediate_tools,
            strict=None,
            supports_strict_mode=compat.supports_strict_mode,
            supports_openai_grammar_tools=compat.supports_openai_grammar_tools,
        )

    if options is not None and options.reasoning_effort is not None:
        mapping = dict(model.thinking_level_map) if model.thinking_level_map is not None else {}
        # pi's `??` chains never yield null here (`null ?? "none"` is "none"), so
        # a mapped-to-None level falls back like an absent one.
        if options.reasoning_effort == "none":
            effort = mapping.get("off") or "none"
        else:
            effort = mapping.get(options.reasoning_effort) or options.reasoning_effort
        body["reasoning"] = {"effort": effort, "summary": options.reasoning_summary or "auto"}

    return body


def _get_service_tier_cost_multiplier(model: Model, service_tier: str | None) -> float:
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.5 if model.id == "gpt-5.5" else 2
    return 1


def _apply_service_tier_pricing(usage: Usage, service_tier: str | None, model: Model) -> None:
    multiplier = _get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1:
        return
    usage.cost.input *= multiplier
    usage.cost.output *= multiplier
    usage.cost.cache_read *= multiplier
    usage.cost.cache_write *= multiplier
    usage.cost.total = usage.cost.input + usage.cost.output + usage.cost.cache_read + usage.cost.cache_write


def _resolve_codex_service_tier(response_service_tier: str | None, request_service_tier: str | None) -> str | None:
    if response_service_tier == "default" and request_service_tier in ("flex", "priority"):
        return request_service_tier
    return response_service_tier if response_service_tier is not None else request_service_tier


def _resolve_codex_url(base_url: str | None = None) -> str:
    raw = base_url if base_url and base_url.strip() else DEFAULT_CODEX_BASE_URL
    normalized = re.sub(r"/+$", "", raw)
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _resolve_codex_websocket_url(base_url: str | None = None) -> str:
    url = _resolve_codex_url(base_url)
    if url.startswith("https:"):
        return f"wss:{url[len('https:') :]}"
    if url.startswith("http:"):
        return f"ws:{url[len('http:') :]}"
    return url


# --- response processing ------------------------------------------------------


async def _process_stream(
    response: CodexSSEResponseLike,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    grammar_tool_input_properties: dict[str, str],
    options: OpenAICodexResponsesOptions | None = None,
) -> None:
    await process_responses_stream(
        _map_codex_events(_parse_sse(response, options.cancel if options else None)),
        output,
        stream,
        model,
        service_tier=options.service_tier if options else None,
        grammar_tool_input_properties=grammar_tool_input_properties,
        resolve_service_tier=_resolve_codex_service_tier,
        apply_service_tier_pricing=lambda usage, tier: _apply_service_tier_pricing(usage, tier, model),
    )


def _is_codex_non_transport_error(error: Any) -> bool:
    return isinstance(error, CodexApiError | CodexProtocolError)


def _is_websocket_connection_limit_reached_error(error: Any) -> bool:
    return isinstance(error, CodexApiError) and error.code == WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE


def _is_previous_response_not_found_error(error: Any) -> bool:
    return isinstance(error, CodexApiError) and error.code == PREVIOUS_RESPONSE_NOT_FOUND_CODE


def _extract_codex_event_error(event: dict) -> tuple[str | None, str | None]:
    nested = event.get("error") if isinstance(event.get("error"), dict) else None
    code = event.get("code") if isinstance(event.get("code"), str) else None
    if code is None and nested is not None and isinstance(nested.get("code"), str):
        code = nested["code"]
    message = event.get("message") if isinstance(event.get("message"), str) else None
    if message is None and nested is not None and isinstance(nested.get("message"), str):
        message = nested["message"]
    return code, message


async def _map_codex_events(events: AsyncIterable[dict]) -> AsyncGenerator[dict]:
    # This generator returns at the terminal event, abandoning its source
    # mid-yield; the source owns the HTTP body, so it is closed explicitly rather
    # than left to the GC (which cannot await — see `utils/http.finish_body`).
    source = aiter(events)
    try:
        async for event in source:
            event_type = event.get("type") if isinstance(event.get("type"), str) else None
            if not event_type:
                continue

            if event_type == "error":
                code, message = _extract_codex_event_error(event)
                raise CodexApiError(
                    f"Codex error: {message or code or json.dumps(event, separators=(',', ':'))}",
                    code=code,
                    payload=event,
                )

            if event_type == "response.failed":
                response = event.get("response") if isinstance(event.get("response"), dict) else None
                error = response.get("error") if response and isinstance(response.get("error"), dict) else None
                code = error.get("code") if error else None
                message = error.get("message") if error else None
                raise CodexApiError(message or "Codex response failed", code=code, payload=event)

            if event_type in ("response.done", "response.completed", "response.incomplete"):
                response = event.get("response")
                normalized_response = response
                if isinstance(response, dict):
                    normalized_response = {**response, "status": _normalize_codex_status(response.get("status"))}
                yield {**event, "type": "response.completed", "response": normalized_response}
                return

            yield event
    finally:
        aclose = getattr(source, "aclose", None)
        if aclose is not None:
            await aclose()


def _normalize_codex_status(status: Any) -> str | None:
    if not isinstance(status, str):
        return None
    return status if status in CODEX_RESPONSE_STATUSES else None


# --- SSE parsing --------------------------------------------------------------


async def _parse_sse(response: CodexSSEResponseLike, cancel: CancelToken | None = None) -> AsyncGenerator[dict]:
    body = response.aiter_bytes()
    ended = False
    try:
        async for sse in iterate_sse_messages(http.cancellable_bytes(body, cancel)):
            # pi skips `[DONE]` and keeps reading (unlike its other adapters,
            # which stop there); the Codex stream ends on `response.completed`.
            data = sse.data.strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except ValueError as cause:
                raise CodexProtocolError(
                    f"Invalid Codex SSE JSON: {format_thrown_value(cause)}", payload=data
                ) from cause
            if isinstance(event, dict):
                yield event
        ended = True
    finally:
        await http.finish_body(body, response, drain=ended)


# --- WebSocket parsing --------------------------------------------------------


def _get_websocket_ready_state(socket: Any) -> int | None:
    ready_state = getattr(socket, "ready_state", None)
    return ready_state if isinstance(ready_state, int) and not isinstance(ready_state, bool) else None


def _is_websocket_reusable(socket: Any) -> bool:
    ready_state = _get_websocket_ready_state(socket)
    # If readyState is unavailable, assume the runtime keeps it open/reusable.
    return ready_state is None or ready_state == websocket.READY_STATE_OPEN


def _forget_entry_locked(session_id: str, entry: _CachedWebSocketConnection) -> None:
    """Drop `entry` from the session cache if it is still the current one."""
    if _websocket_session_cache.get(session_id) is entry:
        del _websocket_session_cache[session_id]


def _is_websocket_session_expired(entry: _CachedWebSocketConnection) -> bool:
    return clock.now_ms() - entry.created_at >= SESSION_WEBSOCKET_MAX_AGE_MS


def _close_websocket_silently(socket: Any, code: int = 1000, reason: str = "done") -> None:
    try:
        socket.close(code, reason)
    except Exception:
        pass


def _schedule_session_websocket_expiry(session_id: str, entry: _CachedWebSocketConnection) -> None:
    """pi's `setTimeout`: close an idle cached socket after the cache TTL."""
    if entry.idle_timer is not None:
        entry.idle_timer.cancel()
    timer = CancelToken()
    entry.idle_timer = timer

    async def _expire() -> None:
        try:
            await clock.sleep_ms(SESSION_WEBSOCKET_CACHE_TTL_MS, timer)
        except AbortError:
            return
        if entry.busy:
            return
        _close_websocket_silently(entry.socket, 1000, "idle_timeout")
        with _websocket_state_guard:
            _forget_entry_locked(session_id, entry)

    tonio.spawn.without_tracking(_expire())


_CONNECT_TIMED_OUT = object()
_CONNECT_CANCELLED = object()


async def _connect_websocket(
    url: str,
    headers: dict[str, str],
    cancel: CancelToken | None = None,
    connect_timeout_ms: float | None = None,
    env: ProviderEnv | None = None,
) -> Any:
    """pi's `connectWebSocket`: connect, bounded by a deadline and the token."""
    ws_headers = {key: value for key, value in headers.items() if key.lower() != "openai-beta"}
    timeout_ms = connect_timeout_ms if connect_timeout_ms is not None else DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS

    if cancel is not None and cancel.cancelled:
        raise RuntimeError("Request was aborted")

    async def _connect() -> Any:
        return await websocket.connect(url, ws_headers, cancel=cancel)

    if timeout_ms <= 0 and cancel is None:
        return await _connect()

    races = [_connect()]
    if timeout_ms > 0:

        async def _timed_out() -> object:
            await tonio.time.sleep(timeout_ms / 1000)
            return _CONNECT_TIMED_OUT

        races.append(_timed_out())
    if cancel is not None:

        async def _aborted() -> object:
            await cancel.wait()
            return _CONNECT_CANCELLED

        races.append(_aborted())

    winner = await tonio.select(*races)
    if winner is _CONNECT_TIMED_OUT:
        raise RuntimeError(f"WebSocket connect timeout after {_format_ms(timeout_ms)}ms")
    if winner is _CONNECT_CANCELLED:
        raise RuntimeError("Request was aborted")
    return winner


@dataclass(slots=True)
class _AcquiredWebSocket:
    socket: Any
    entry: _CachedWebSocketConnection | None
    reused: bool
    release: Any  # Callable[[bool], None] - keep flag


async def _acquire_websocket(
    url: str,
    headers: dict[str, str],
    session_id: str | None,
    cancel: CancelToken | None = None,
    connect_timeout_ms: float | None = None,
    env: ProviderEnv | None = None,
) -> _AcquiredWebSocket:
    if not session_id:
        socket = await _connect_websocket(url, headers, cancel, connect_timeout_ms, env)
        return _AcquiredWebSocket(
            socket=socket,
            entry=None,
            reused=False,
            release=lambda keep=False: _close_websocket_silently(socket),
        )

    # One critical section for the whole cache decision. pi relies on
    # JavaScript's single thread to make check-then-claim atomic; two concurrent
    # turns on one session (pi handles that case explicitly — see the `busy`
    # branch) would otherwise both claim the same socket. Only the closes and the
    # connect happen outside, since neither may run under a thread lock.
    reuse: _CachedWebSocketConnection | None = None
    stale: tuple[Any, str] | None = None
    timer: CancelToken | None = None
    was_busy = False
    with _websocket_state_guard:
        cached = _websocket_session_cache.get(session_id)
        if cached is not None:
            timer, cached.idle_timer = cached.idle_timer, None
            was_busy = cached.busy
            if not cached.busy and _is_websocket_session_expired(cached):
                stale = (cached.socket, "connection_age_limit")
                _forget_entry_locked(session_id, cached)
            elif not cached.busy and _is_websocket_reusable(cached.socket):
                cached.busy = True
                reuse = cached
            elif not cached.busy:
                stale = (cached.socket, "done")
                _forget_entry_locked(session_id, cached)

    if timer is not None:
        timer.cancel()
    if stale is not None:
        _close_websocket_silently(stale[0], 1000, stale[1])

    if reuse is not None:

        def release_cached(keep: bool = False, entry: _CachedWebSocketConnection = reuse) -> None:
            if not keep or not _is_websocket_reusable(entry.socket):
                _close_websocket_silently(entry.socket)
                with _websocket_state_guard:
                    _forget_entry_locked(session_id, entry)
                return
            entry.busy = False
            _schedule_session_websocket_expiry(session_id, entry)

        return _AcquiredWebSocket(socket=reuse.socket, entry=reuse, reused=True, release=release_cached)

    if was_busy:
        # A concurrent turn owns the cached socket: this one gets a one-shot.
        socket = await _connect_websocket(url, headers, cancel, connect_timeout_ms, env)
        return _AcquiredWebSocket(
            socket=socket,
            entry=None,
            reused=False,
            release=lambda keep=False: _close_websocket_silently(socket),
        )

    socket = await _connect_websocket(url, headers, cancel, connect_timeout_ms, env)
    entry = _CachedWebSocketConnection(socket=socket, busy=True, created_at=clock.now_ms())
    with _websocket_state_guard:
        _websocket_session_cache[session_id] = entry

    def release_new(keep: bool = False) -> None:
        if not keep or not _is_websocket_reusable(entry.socket):
            _close_websocket_silently(entry.socket)
            if entry.idle_timer is not None:
                entry.idle_timer.cancel()
            with _websocket_state_guard:
                _forget_entry_locked(session_id, entry)
            return
        entry.busy = False
        _schedule_session_websocket_expiry(session_id, entry)

    return _AcquiredWebSocket(socket=socket, entry=entry, reused=False, release=release_new)


def _extract_websocket_error(event: Any) -> Exception:
    message = getattr(event, "message", None)
    if isinstance(message, str) and message:
        return RuntimeError(message)
    nested = getattr(event, "error", None)
    if isinstance(nested, BaseException) and str(nested):
        return nested if isinstance(nested, Exception) else RuntimeError(str(nested))
    nested_message = getattr(nested, "message", None)
    if isinstance(nested_message, str) and nested_message:
        return RuntimeError(nested_message)
    return RuntimeError("WebSocket error")


def _extract_websocket_close_error(event: Any) -> Exception:
    code = getattr(event, "code", None)
    reason = getattr(event, "reason", None)
    was_clean = getattr(event, "was_clean", None)
    code_text = f" {code}" if isinstance(code, int) and not isinstance(code, bool) else ""
    reason_text = f" {reason}" if isinstance(reason, str) and reason else ""
    if not reason_text and code == WEBSOCKET_MESSAGE_TOO_BIG_CLOSE_CODE:
        reason_text = " message too big"
    return WebSocketCloseError(
        f"WebSocket closed{code_text}{reason_text}".strip(),
        code=code if isinstance(code, int) and not isinstance(code, bool) else None,
        reason=reason if isinstance(reason, str) and reason else None,
        was_clean=was_clean if isinstance(was_clean, bool) else None,
    )


def _decode_websocket_data(data: Any) -> str | None:
    if isinstance(data, str):
        return data
    if isinstance(data, bytes | bytearray | memoryview):
        return bytes(data).decode("utf-8", "replace")
    return None


_WS_IDLE_TIMED_OUT = object()
_WS_CANCELLED = object()

_WEBSOCKET_COMPLETION_TYPES = ("response.completed", "response.done", "response.incomplete")


async def _parse_websocket(
    socket: Any,
    cancel: CancelToken | None = None,
    idle_timeout_ms: float | None = None,
) -> AsyncGenerator[dict]:
    """pi's `parseWebSocket`, over the queued-event socket surface.

    pi registers DOM listeners and drains a queue; here events are already
    queued by the transport (see `utils/websocket.py` on why), so this is the
    same state machine with `receive_event()` in place of the callbacks.
    """
    saw_completion = False
    while True:
        if cancel is not None and cancel.cancelled:
            raise RuntimeError("Request was aborted")

        races: list[Any] = [socket.receive_event()]
        if idle_timeout_ms is not None and idle_timeout_ms > 0:

            async def _idle() -> object:
                await tonio.time.sleep(idle_timeout_ms / 1000)
                return _WS_IDLE_TIMED_OUT

            races.append(_idle())
        if cancel is not None:

            async def _aborted() -> object:
                await cancel.wait()
                return _WS_CANCELLED

            races.append(_aborted())

        event = await tonio.select(*races)
        if event is _WS_IDLE_TIMED_OUT:
            _close_websocket_silently(socket, 1000, "idle_timeout")
            raise RuntimeError(f"WebSocket idle timeout after {_format_ms(idle_timeout_ms)}ms")
        if event is _WS_CANCELLED:
            raise RuntimeError("Request was aborted")

        event_type = getattr(event, "type", None)
        if event_type == "message":
            text = _decode_websocket_data(getattr(event, "data", None))
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except ValueError as cause:
                raise CodexProtocolError(
                    f"Invalid Codex WebSocket JSON: {format_thrown_value(cause)}", payload=text
                ) from cause
            if not isinstance(parsed, dict):
                continue
            if parsed.get("type") in _WEBSOCKET_COMPLETION_TYPES:
                saw_completion = True
            yield parsed
            continue
        if event_type == "error":
            raise _extract_websocket_error(event)
        if event_type == "close":
            if saw_completion:
                return
            raise _extract_websocket_close_error(event)


def _request_body_without_input(body: dict) -> dict:
    return {key: value for key, value in body.items() if key not in ("input", "previous_response_id")}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=False, separators=(",", ":"), default=str)


def _response_inputs_equal(first: list | None, second: list | None) -> bool:
    return _canonical_json(first or []) == _canonical_json(second or [])


def _request_bodies_match_except_input(first: dict, second: dict) -> bool:
    return _canonical_json(_request_body_without_input(first)) == _canonical_json(_request_body_without_input(second))


def _get_cached_websocket_input_delta(
    body: dict,
    continuation: _CachedWebSocketContinuationState,
) -> list | None:
    if not _request_bodies_match_except_input(body, continuation.last_request_body):
        return None

    current_input = body.get("input") or []
    baseline = [*(continuation.last_request_body.get("input") or []), *continuation.last_response_items]
    if len(current_input) < len(baseline):
        return None

    if not _response_inputs_equal(current_input[: len(baseline)], baseline):
        return None

    return current_input[len(baseline) :]


def _build_cached_websocket_request_body(entry: _CachedWebSocketConnection, body: dict) -> dict:
    continuation = entry.continuation
    if continuation is None:
        return body

    delta = _get_cached_websocket_input_delta(body, continuation)
    if delta is None or not continuation.last_response_id:
        entry.continuation = None
        return body

    return {**body, "previous_response_id": continuation.last_response_id, "input": delta}


async def _start_websocket_output_on_first_event(
    events: AsyncIterable[dict],
    on_start,
) -> AsyncGenerator[dict]:
    started = False
    async for event in events:
        if not started:
            started = True
            on_start()
        yield event


async def _process_websocket_stream(
    url: str,
    body: dict,
    headers: dict[str, str],
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Model,
    on_start,
    idle_timeout_ms: float | None,
    websocket_connect_timeout_ms: float | None,
    cache_session_id: str | None,
    grammar_tool_input_properties: dict[str, str],
    options: OpenAICodexResponsesOptions | None = None,
) -> None:
    acquired = await _acquire_websocket(
        url,
        headers,
        cache_session_id,
        options.cancel if options else None,
        websocket_connect_timeout_ms,
        options.env if options else None,
    )
    socket, entry, reused, release = acquired.socket, acquired.entry, acquired.reused, acquired.release
    keep_connection = True
    transport = options.transport if options else None
    use_cached_context = transport in ("websocket-cached", "auto")
    # ChatGPT Codex Responses rejects `store: true` ("Store must be set to false").
    # WebSocket continuation still works via connection-scoped previous_response_id state.
    full_body = body
    request_body = _build_cached_websocket_request_body(entry, full_body) if use_cached_context and entry else full_body
    stats = _get_or_create_websocket_debug_stats(cache_session_id) if cache_session_id else None
    if stats is not None:
        with _websocket_state_guard:
            stats.requests += 1
            if reused:
                stats.connections_reused += 1
            else:
                stats.connections_created += 1
            if use_cached_context:
                stats.cached_context_requests += 1
            if request_body.get("store") is True:
                stats.store_true_requests += 1
            stats.last_input_items = len(request_body.get("input") or [])
            if request_body.get("previous_response_id"):
                stats.delta_requests += 1
                stats.last_delta_input_items = len(request_body.get("input") or [])
                stats.last_previous_response_id = request_body["previous_response_id"]
            else:
                stats.full_context_requests += 1
                stats.last_delta_input_items = None
                stats.last_previous_response_id = None
    try:
        socket.send(json.dumps({"type": "response.create", **request_body}, separators=(",", ":")))
        await process_responses_stream(
            _start_websocket_output_on_first_event(
                _map_codex_events(
                    _parse_websocket(socket, options.cancel if options else None, idle_timeout_ms),
                ),
                on_start,
            ),
            output,
            stream,
            model,
            service_tier=options.service_tier if options else None,
            grammar_tool_input_properties=grammar_tool_input_properties,
            resolve_service_tier=_resolve_codex_service_tier,
            apply_service_tier_pricing=lambda usage, tier: _apply_service_tier_pricing(usage, tier, model),
        )
        if options is not None and options.cancel is not None and options.cancel.cancelled:
            keep_connection = False
        elif use_cached_context and entry is not None and output.response_id:
            response_items = [
                item
                for item in convert_responses_messages(
                    model,
                    Context(messages=[output]),
                    set(CODEX_TOOL_CALL_PROVIDERS),
                    include_system_prompt=False,
                    grammar_tool_input_properties=grammar_tool_input_properties,
                )
                if item.get("type") not in ("function_call_output", "custom_tool_call_output")
            ]
            entry.continuation = _CachedWebSocketContinuationState(
                last_request_body=full_body,
                last_response_id=output.response_id,
                last_response_items=response_items,
            )
    except BaseException:
        if entry is not None:
            entry.continuation = None
        keep_connection = False
        raise
    finally:
        release(keep_connection)


# --- error handling -----------------------------------------------------------


async def _read_error_text(response: CodexSSEResponseLike) -> str:
    read_text = getattr(response, "read_text", None)
    if read_text is not None:
        return await read_text()
    chunks = [chunk async for chunk in response.aiter_bytes()]
    return b"".join(chunks).decode("utf-8", "replace")


def _parse_error_response(raw: str, status: int) -> tuple[str, str | None]:
    """pi's `parseErrorResponse`: returns (message, friendlyMessage)."""
    message = raw or "Request failed"
    friendly_message: str | None = None

    try:
        parsed = json.loads(raw)
    except ValueError:
        return message, friendly_message

    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict):
        code = error.get("code") or error.get("type") or ""
        if (isinstance(code, str) and _USAGE_LIMIT_CODE_PATTERN.search(code)) or status == 429:
            plan_type = error.get("plan_type")
            plan = f" ({plan_type.lower()} plan)" if isinstance(plan_type, str) and plan_type else ""
            resets_at = error.get("resets_at")
            when = ""
            if isinstance(resets_at, int | float) and not isinstance(resets_at, bool):
                mins = max(0, round((resets_at * 1000 - clock.now_ms()) / 60000))
                when = f" Try again in ~{mins} min."
            friendly_message = f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
        message = error.get("message") or friendly_message or message

    return message, friendly_message


# --- auth & headers -----------------------------------------------------------


def _extract_account_id(token: str) -> str:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token")
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
        account_id = (payload.get(JWT_CLAIM_PATH) or {}).get("chatgpt_account_id")
        if not account_id:
            raise ValueError("No account ID in token")
        return account_id
    except Exception as error:
        raise RuntimeError("Failed to extract accountId from token") from error


def _codex_user_agent() -> str:
    # pi: `pi (${os.platform()} ${os.release()}; ${os.arch()})`. Node reports
    # "x64"/"arm64" where platform.machine() reports "x86_64"/"aarch64"; the raw
    # Python value is sent.
    #
    # Deliberately still "pi", unlike the Phase 7 step 1 attribution swap: this
    # pairs with the `originator: pi` header below, and the Codex backend may
    # gate on known originator values. Changing both needs a live Codex account
    # to verify — see PLAN, step 1 follow-up.
    return f"pi ({platform.system().lower()} {platform.release()}; {platform.machine()})"


def _build_base_codex_headers(
    init_headers: dict[str, str] | None,
    additional_headers: ProviderHeaders | None,
    account_id: str,
    token: str,
) -> dict[str, str]:
    headers = dict(init_headers or {})
    for key, value in (additional_headers or {}).items():
        if value is None:
            _delete_header(headers, key)
        else:
            _set_header(headers, key, value)
    _set_header(headers, "Authorization", f"Bearer {token}")
    _set_header(headers, "chatgpt-account-id", account_id)
    _set_header(headers, "originator", "pi")
    _set_header(headers, "User-Agent", _codex_user_agent())
    return headers


def _set_header(headers: dict[str, str], name: str, value: str) -> None:
    """`Headers.set`: case-insensitive replace, keeping the incoming name."""
    _delete_header(headers, name)
    headers[name] = value


def _delete_header(headers: dict[str, str], name: str) -> None:
    lowered = name.lower()
    for key in [key for key in headers if key.lower() == lowered]:
        del headers[key]


def _build_sse_headers(
    init_headers: dict[str, str] | None,
    additional_headers: ProviderHeaders | None,
    account_id: str,
    token: str,
    session_id: str | None = None,
) -> dict[str, str]:
    headers = _build_base_codex_headers(init_headers, additional_headers, account_id, token)
    _set_header(headers, "OpenAI-Beta", "responses=experimental")
    _set_header(headers, "accept", "text/event-stream")
    _set_header(headers, "content-type", "application/json")

    if session_id:
        _set_header(headers, "session-id", session_id)
        _set_header(headers, "x-client-request-id", session_id)

    return headers


def _build_websocket_headers(
    init_headers: dict[str, str] | None,
    additional_headers: ProviderHeaders | None,
    account_id: str,
    token: str,
    request_id: str,
) -> dict[str, str]:
    headers = _build_base_codex_headers(init_headers, additional_headers, account_id, token)
    _delete_header(headers, "accept")
    _delete_header(headers, "content-type")
    _delete_header(headers, "OpenAI-Beta")
    _set_header(headers, "OpenAI-Beta", OPENAI_BETA_RESPONSES_WEBSOCKETS)
    _set_header(headers, "x-client-request-id", request_id)
    _set_header(headers, "session-id", request_id)
    return headers
