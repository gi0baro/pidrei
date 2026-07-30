"""Port of pi's anthropic-messages adapter (packages/ai/src/api/anthropic-messages.ts).

The canonical adapter shape: hand-built request params, hand-parsed SSE (pi
never streams through the SDK), incremental usage/cost accounting, and the
Claude Code "stealth mode" tool-name mapping for OAuth tokens.

Transport: pi constructs an @anthropic-ai/sdk client purely as an HTTP
carrier; pidrei uses punkreq through the seam instead. Tests inject a fake
client via `AnthropicOptions.client` exactly like pi's suites do.
"""

import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, fields
from typing import Any, Literal, Protocol

import tonio.colored as tonio

from pidrei_ai.api.constrained_sampling import resolve_json_schema_strict_sampling
from pidrei_ai.api.github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from pidrei_ai.api.simple_options import adjust_max_tokens_for_thinking, build_base_options, clamp_max_tokens_to_context
from pidrei_ai.api.transform_messages import transform_messages
from pidrei_ai.registry import calculate_cost
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    ProviderEnv,
    ProviderHeaders,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingLevel,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
)
from pidrei_ai.utils import http
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.deferred_tools import split_deferred_tools
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.json_parse import parse_json_with_repair, parse_streaming_json
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates
from pidrei_ai.utils.sse import iterate_sse_messages


ANTHROPIC_VERSION = "2023-06-01"

# Stealth mode: mimic Claude Code's tool naming exactly.
CLAUDE_CODE_VERSION = "2.1.75"

# Claude Code 2.x tool names (canonical casing).
# Source: https://cchistory.mariozechner.at/data/prompts-2.1.11.md
_CLAUDE_CODE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
]
_CC_TOOL_LOOKUP = {name.lower(): name for name in _CLAUDE_CODE_TOOLS}

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"

_ANTHROPIC_MESSAGE_EVENTS = frozenset(
    (
        "message_start",
        "message_delta",
        "message_stop",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    )
)

type AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
type AnthropicThinkingDisplay = Literal["summarized", "omitted"]


def _to_claude_code_name(name: str) -> str:
    return _CC_TOOL_LOOKUP.get(name.lower(), name)


def _from_claude_code_name(name: str, tools: list[Tool] | None) -> str:
    if tools:
        lower_name = name.lower()
        for tool in tools:
            if tool.name.lower() == lower_name:
                return tool.name
    return name


def _resolve_cache_retention(cache_retention: CacheRetention | None, env: ProviderEnv | None) -> CacheRetention:
    """Defaults to "short". pi reads `PI_CACHE_RETENTION` "for backward
    compatibility" with its own older releases; there is no pidrei release to be
    compatible with, so the knob keeps its role under the renamed env var.
    """
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PIDREI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _get_cache_control(
    model: Model,
    cache_retention: CacheRetention | None,
    env: ProviderEnv | None,
) -> dict[str, str] | None:
    retention = _resolve_cache_retention(cache_retention, env)
    if retention == "none":
        return None
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if retention == "long" and _get_compat(model).supports_long_cache_retention:
        cache_control["ttl"] = "1h"
    return cache_control


@dataclass(slots=True)
class _ResolvedCompat:
    supports_eager_tool_input_streaming: bool
    supports_long_cache_retention: bool
    send_session_affinity_headers: bool
    supports_cache_control_on_tools: bool
    supports_temperature: bool
    allow_empty_signature: bool
    supports_strict_tools: bool
    supports_tool_references: bool


def _default_supports_tool_references(model: Model) -> bool:
    """First-party Anthropic models except Haiku and pre-tool-search models."""
    if model.provider != "anthropic" or "haiku" in model.id:
        return False
    version = re.match(r"^claude-(?:opus|sonnet|fable)-(\d+)(?:-(\d+))?(?:-|$)", model.id)
    if not version:
        return False
    major = int(version.group(1))
    minor = int(version.group(2)) if version.group(2) and len(version.group(2)) < 8 else 0
    return major > 4 or (major == 4 and minor >= 5)


def _get_compat(model: Model) -> _ResolvedCompat:
    compat = model.compat if isinstance(model.compat, AnthropicMessagesCompat) else None

    def resolved(value: bool | None, default: bool) -> bool:
        return value if value is not None else default

    return _ResolvedCompat(
        supports_eager_tool_input_streaming=resolved(
            compat.supports_eager_tool_input_streaming if compat else None, True
        ),
        supports_long_cache_retention=resolved(compat.supports_long_cache_retention if compat else None, True),
        send_session_affinity_headers=resolved(compat.send_session_affinity_headers if compat else None, False),
        supports_cache_control_on_tools=resolved(compat.supports_cache_control_on_tools if compat else None, True),
        supports_temperature=resolved(compat.supports_temperature if compat else None, True),
        allow_empty_signature=resolved(compat.allow_empty_signature if compat else None, False),
        supports_strict_tools=resolved(compat.supports_strict_tools if compat else None, False),
        supports_tool_references=resolved(
            compat.supports_tool_references if compat else None, _default_supports_tool_references(model)
        ),
    )


def _force_adaptive_thinking(model: Model) -> bool:
    compat = model.compat if isinstance(model.compat, AnthropicMessagesCompat) else None
    return compat is not None and compat.force_adaptive_thinking is True


def _convert_content_blocks(content: list) -> str | list[dict[str, Any]]:
    """Convert text/image blocks to Anthropic API format."""
    has_images = any(block.type == "image" for block in content)
    if not has_images:
        return sanitize_surrogates("\n".join(block.text for block in content))

    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": block.mime_type, "data": block.data},
                }
            )

    if not any(block["type"] == "text" for block in blocks):
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})
    return blocks


@dataclass(slots=True)
class AnthropicOptions(StreamOptions):
    """Adapter-specific stream options (pi: AnthropicOptions)."""

    # Enable extended thinking. Adaptive models decide when/how much to think;
    # older models use budget-based thinking with thinking_budget_tokens.
    thinking_enabled: bool | None = None
    # Token budget for extended thinking (older models only). Default 1024 when enabled.
    thinking_budget_tokens: int | None = None
    # Effort level for adaptive thinking models.
    effort: AnthropicEffort | None = None
    # "summarized" (default when thinking is enabled) or "omitted".
    thinking_display: AnthropicThinkingDisplay | None = None
    # Request the interleaved-thinking beta for non-adaptive models. Default True.
    interleaved_thinking: bool | None = None
    # Anthropic tool choice: "auto" | "any" | "none" | {"type": "tool", "name": ...}.
    tool_choice: str | dict[str, str] | None = None
    # Pre-built client instance (test injection / alternative transports).
    client: AnthropicClient | None = None


class AnthropicResponseLike(Protocol):
    status: int
    headers: dict[str, str]

    def aiter_bytes(self) -> AsyncIterable[bytes]: ...


class AnthropicClient(Protocol):
    """The adapter's transport contract (pi injects an SDK client here)."""

    async def create(
        self,
        params: dict[str, Any],
        *,
        timeout_ms: float | None,
        cancel: CancelToken | None,
    ) -> AnthropicResponseLike: ...


class AnthropicApiError(Exception):
    """Non-2xx response; carries status/headers for the SDK-mirror retry policy."""

    def __init__(self, status: int, headers: dict[str, str], message: str, error: dict | None = None):
        super().__init__(message)
        self.status = status
        self.headers = headers
        self.error = error


@dataclass(slots=True)
class _PunkreqResponse:
    status: int
    headers: dict[str, str]
    _response: Any

    def aiter_bytes(self) -> AsyncIterable[bytes]:
        return self._response.iter_bytes()

    async def close(self) -> None:
        await self._response.close()


class _PunkreqAnthropicClient:
    """Default transport: POST {base_url}/v1/messages through the punkreq seam."""

    def __init__(self, base_url: str, headers: dict[str, str], env: ProviderEnv | None = None):
        self._url = f"{base_url.rstrip('/')}/v1/messages"
        self._headers = headers
        self._env = env

    async def create(
        self,
        params: dict[str, Any],
        *,
        timeout_ms: float | None,
        cancel: CancelToken | None,
    ) -> AnthropicResponseLike:
        client = http.client_for(self._url, self._env)
        timeout = http.request_timeout(timeout_ms)
        response = await client.post(self._url, json=params, headers=self._headers, timeout=timeout)
        if not 200 <= response.status_code < 300:
            body = (await response.read()).decode("utf-8", "replace")
            raise AnthropicApiError(
                status=response.status_code,
                headers=dict(response.headers),
                message=f"{response.status_code} {_extract_error_message(body)}",
                error=_parse_error_body(body),
            )
        return _PunkreqResponse(status=response.status_code, headers=dict(response.headers), _response=response)


def _parse_error_body(body: str) -> dict | None:
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_error_message(body: str) -> str:
    """Mirror the SDK's error message shape: prefer the API error message text."""
    parsed = _parse_error_body(body)
    if parsed is not None:
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return body


def _merge_headers(*header_sources: ProviderHeaders | None) -> dict[str, str | None]:
    merged: dict[str, str | None] = {}
    for headers in header_sources:
        if headers:
            merged.update(headers)
    return merged


def _has_header(headers: ProviderHeaders | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(key.lower() == expected and value is not None and value.strip() for key, value in headers.items())


def _assert_request_auth(provider: str, api_key: str | None, headers: ProviderHeaders | None) -> None:
    if api_key:
        return
    if (
        _has_header(headers, "authorization")
        or _has_header(headers, "x-api-key")
        or _has_header(headers, "cf-aig-authorization")
    ):
        return
    raise RuntimeError(f"No API key for provider: {provider}")


def _is_oauth_token(api_key: str) -> bool:
    return "sk-ant-oat" in api_key


def _should_use_fine_grained_beta(model: Model, context: Context) -> bool:
    return bool(context.tools) and not _get_compat(model).supports_eager_tool_input_streaming


def _create_client(
    model: Model,
    api_key: str | None,
    interleaved_thinking: bool,
    use_fine_grained_beta: bool,
    options_headers: ProviderHeaders | None,
    dynamic_headers: dict[str, str] | None,
    session_id: str | None,
    env: ProviderEnv | None = None,
) -> tuple[AnthropicClient, bool]:
    """Build the default transport with pi's exact header assembly."""
    # Adaptive thinking models have interleaved thinking built in; skip the beta.
    needs_interleaved_beta = interleaved_thinking and not _force_adaptive_thinking(model)
    beta_features: list[str] = []
    if use_fine_grained_beta:
        beta_features.append(FINE_GRAINED_TOOL_STREAMING_BETA)
    if needs_interleaved_beta:
        beta_features.append(INTERLEAVED_THINKING_BETA)

    base = {
        "accept": "application/json",
        "anthropic-dangerous-direct-browser-access": "true",
        "anthropic-version": ANTHROPIC_VERSION,
    }
    if beta_features:
        base["anthropic-beta"] = ",".join(beta_features)

    # Copilot: Bearer auth, selective betas.
    if model.provider == "github-copilot":
        merged = _merge_headers(
            base,
            {"authorization": f"Bearer {api_key}"} if api_key else None,
            model.headers,
            dynamic_headers,
            options_headers,
        )
        headers = {key: value for key, value in merged.items() if value is not None}
        return _PunkreqAnthropicClient(model.base_url, headers, env), False

    # OAuth: Bearer auth, Claude Code identity headers.
    if api_key and _is_oauth_token(api_key):
        merged = _merge_headers(
            {
                **base,
                "anthropic-beta": ",".join(["claude-code-20250219", "oauth-2025-04-20", *beta_features]),
                "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                "x-app": "cli",
            },
            {"authorization": f"Bearer {api_key}"},
            model.headers,
            options_headers,
        )
        headers = {key: value for key, value in merged.items() if value is not None}
        return _PunkreqAnthropicClient(model.base_url, headers, env), True

    # API key or header-owned auth.
    session_affinity: dict[str, str] = (
        {"x-session-affinity": session_id} if session_id and _get_compat(model).send_session_affinity_headers else {}
    )
    merged = _merge_headers(
        base,
        {"x-api-key": api_key} if api_key else None,
        session_affinity,
        model.headers,
        options_headers,
    )
    headers = {key: value for key, value in merged.items() if value is not None}
    return _PunkreqAnthropicClient(model.base_url, headers, env), False


async def _iterate_anthropic_events(
    response: AnthropicResponseLike,
    cancel: CancelToken | None,
) -> AsyncGenerator[dict[str, Any]]:
    saw_message_start = False
    saw_message_end = False

    body = response.aiter_bytes()
    ended = False
    try:
        async for sse in iterate_sse_messages(http.cancellable_bytes(body, cancel)):
            if sse.event == "error":
                raise RuntimeError(sse.data)

            if (sse.event or "") not in _ANTHROPIC_MESSAGE_EVENTS:
                continue

            try:
                event = parse_json_with_repair(sse.data)
            except Exception as error:
                raw = "\\n".join(sse.raw)
                raise RuntimeError(
                    f"Could not parse Anthropic SSE event {sse.event}: {error}; data={sse.data}; raw={raw}"
                )
            if event.get("type") == "message_start":
                saw_message_start = True
            elif event.get("type") == "message_stop":
                saw_message_end = True
            yield event
        ended = True
    finally:
        await http.finish_body(body, response, drain=ended)

    if saw_message_start and not saw_message_end:
        raise RuntimeError("Anthropic stream ended before message_stop")


def _map_stop_reason(reason: str, stop_details: dict | None) -> tuple[StopReason, str | None]:
    if reason == "end_turn":
        return "stop", None
    if reason == "max_tokens":
        return "length", None
    if reason == "tool_use":
        return "toolUse", None
    if reason == "refusal":
        explanation = stop_details.get("explanation") if isinstance(stop_details, dict) else None
        return "error", explanation or "The model refused to complete the request"
    if reason == "pause_turn":  # Stop is good enough -> resubmit
        return "stop", None
    if reason == "stop_sequence":  # We don't supply stop sequences
        return "stop", None
    if reason == "sensitive":  # Content flagged by safety filters
        return "error", "Provider stopped with: sensitive"
    # Handle unknown stop reasons loudly (the API may add new values).
    raise RuntimeError(f"Unhandled stop reason: {reason}")


def _anthropic_options(options: StreamOptions | None) -> AnthropicOptions:
    if isinstance(options, AnthropicOptions):
        return options
    if options is None:
        return AnthropicOptions()
    values = {field.name: getattr(options, field.name) for field in fields(StreamOptions)}
    return AnthropicOptions(**values)


def stream(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    opts = _anthropic_options(options)
    out_stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            content=[],
            api=model.api,
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )
        # Streaming scratch (pi keeps these on the blocks and strips them later;
        # slotted dataclasses keep them out-of-band instead).
        anthropic_index_to_content: dict[int, int] = {}
        partial_json: dict[int, str] = {}

        try:
            if opts.client is not None:
                client: AnthropicClient = opts.client
                is_oauth = False
            else:
                api_key = opts.api_key
                _assert_request_auth(model.provider, api_key, opts.headers)
                copilot_dynamic_headers: dict[str, str] | None = None
                if model.provider == "github-copilot":
                    copilot_dynamic_headers = build_copilot_dynamic_headers(
                        context.messages, has_copilot_vision_input(context.messages)
                    )
                cache_retention = _resolve_cache_retention(opts.cache_retention, opts.env)
                cache_session_id = None if cache_retention == "none" else opts.session_id
                client, is_oauth = _create_client(
                    model,
                    api_key,
                    opts.interleaved_thinking if opts.interleaved_thinking is not None else True,
                    _should_use_fine_grained_beta(model, context),
                    opts.headers,
                    copilot_dynamic_headers,
                    cache_session_id,
                    opts.env,
                )

            params = _build_params(model, context, is_oauth, opts)
            next_params = await maybe_call(opts.on_payload, params, model)
            if next_params is not None:
                params = next_params

            async def _request():
                return await client.create({**params, "stream": True}, timeout_ms=opts.timeout_ms, cancel=opts.cancel)

            response = await retry_provider_request(
                _request,
                max_retries=opts.max_retries if opts.max_retries is not None else 0,
                max_retry_delay_ms=opts.max_retry_delay_ms,
                cancel=opts.cancel,
            )

            await maybe_call(
                opts.on_response, ProviderResponse(status=response.status, headers=response.headers), model
            )
            out_stream.push(StartEvent(partial=output))

            blocks = output.content

            async for event in _iterate_anthropic_events(response, opts.cancel):
                event_type = event.get("type")
                if event_type == "message_start":
                    message = event.get("message") or {}
                    usage = message.get("usage") or {}
                    output.response_id = message.get("id")
                    # Capture initial usage so input counts survive early aborts.
                    output.usage.input = usage.get("input_tokens") or 0
                    output.usage.output = usage.get("output_tokens") or 0
                    output.usage.cache_read = usage.get("cache_read_input_tokens") or 0
                    output.usage.cache_write = usage.get("cache_creation_input_tokens") or 0
                    cache_creation = usage.get("cache_creation") or {}
                    output.usage.cache_write_1h = cache_creation.get("ephemeral_1h_input_tokens") or 0
                    # Anthropic doesn't provide total_tokens; compute from components.
                    output.usage.total_tokens = (
                        output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
                    )
                    calculate_cost(model, output.usage)
                elif event_type == "content_block_start":
                    content_block = event.get("content_block") or {}
                    block_type = content_block.get("type")
                    index = event.get("index")
                    if block_type == "text":
                        blocks.append(TextContent(text=""))
                        anthropic_index_to_content[index] = len(blocks) - 1
                        out_stream.push(TextStartEvent(content_index=len(blocks) - 1, partial=output))
                    elif block_type == "thinking":
                        blocks.append(ThinkingContent(thinking="", thinking_signature=""))
                        anthropic_index_to_content[index] = len(blocks) - 1
                        out_stream.push(ThinkingStartEvent(content_index=len(blocks) - 1, partial=output))
                    elif block_type == "redacted_thinking":
                        blocks.append(
                            ThinkingContent(
                                thinking="[Reasoning redacted]",
                                thinking_signature=content_block.get("data"),
                                redacted=True,
                            )
                        )
                        anthropic_index_to_content[index] = len(blocks) - 1
                        out_stream.push(ThinkingStartEvent(content_index=len(blocks) - 1, partial=output))
                    elif block_type == "tool_use":
                        name = content_block.get("name", "")
                        blocks.append(
                            ToolCall(
                                id=content_block.get("id", ""),
                                name=_from_claude_code_name(name, context.tools) if is_oauth else name,
                                arguments=content_block.get("input") or {},
                            )
                        )
                        content_index = len(blocks) - 1
                        anthropic_index_to_content[index] = content_index
                        partial_json[content_index] = ""
                        out_stream.push(ToolCallStartEvent(content_index=content_index, partial=output))
                elif event_type == "content_block_delta":
                    delta = event.get("delta") or {}
                    delta_type = delta.get("type")
                    content_index = anthropic_index_to_content.get(event.get("index"))
                    block = blocks[content_index] if content_index is not None else None
                    if delta_type == "text_delta":
                        if block is not None and block.type == "text":
                            block.text += delta.get("text", "")
                            out_stream.push(
                                TextDeltaEvent(content_index=content_index, delta=delta.get("text", ""), partial=output)
                            )
                    elif delta_type == "thinking_delta":
                        if block is not None and block.type == "thinking":
                            block.thinking += delta.get("thinking", "")
                            out_stream.push(
                                ThinkingDeltaEvent(
                                    content_index=content_index, delta=delta.get("thinking", ""), partial=output
                                )
                            )
                    elif delta_type == "input_json_delta":
                        if block is not None and block.type == "toolCall":
                            partial_json[content_index] += delta.get("partial_json", "")
                            block.arguments = parse_streaming_json(partial_json[content_index])
                            out_stream.push(
                                ToolCallDeltaEvent(
                                    content_index=content_index,
                                    delta=delta.get("partial_json", ""),
                                    partial=output,
                                )
                            )
                    elif delta_type == "signature_delta" and block is not None and block.type == "thinking":
                        block.thinking_signature = (block.thinking_signature or "") + delta.get("signature", "")
                elif event_type == "content_block_stop":
                    content_index = anthropic_index_to_content.get(event.get("index"))
                    block = blocks[content_index] if content_index is not None else None
                    if block is not None:
                        if block.type == "text":
                            out_stream.push(
                                TextEndEvent(content_index=content_index, content=block.text, partial=output)
                            )
                        elif block.type == "thinking":
                            out_stream.push(
                                ThinkingEndEvent(content_index=content_index, content=block.thinking, partial=output)
                            )
                        elif block.type == "toolCall":
                            block.arguments = parse_streaming_json(partial_json.pop(content_index, ""))
                            out_stream.push(
                                ToolCallEndEvent(content_index=content_index, tool_call=block, partial=output)
                            )
                elif event_type == "message_delta":
                    delta = event.get("delta") or {}
                    if delta.get("stop_reason"):
                        output.raw_stop_reason = delta["stop_reason"]
                        stop_reason, error_message = _map_stop_reason(delta["stop_reason"], delta.get("stop_details"))
                        output.stop_reason = stop_reason
                        if error_message:
                            output.error_message = error_message
                    # Only update usage fields if present (not null): preserves
                    # message_start input counts when proxies omit them here.
                    usage = event.get("usage")
                    if usage:
                        if usage.get("input_tokens") is not None:
                            output.usage.input = usage["input_tokens"]
                        if usage.get("output_tokens") is not None:
                            output.usage.output = usage["output_tokens"]
                        if usage.get("cache_read_input_tokens") is not None:
                            output.usage.cache_read = usage["cache_read_input_tokens"]
                        if usage.get("cache_creation_input_tokens") is not None:
                            output.usage.cache_write = usage["cache_creation_input_tokens"]
                        # Reasoning tokens ride on output_tokens_details.thinking_tokens
                        # in the final message_delta usage (a subset of output_tokens).
                        details = usage.get("output_tokens_details") or {}
                        if details.get("thinking_tokens") is not None:
                            output.usage.reasoning = details["thinking_tokens"]
                    output.usage.total_tokens = (
                        output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
                    )
                    calculate_cost(model, output.usage)

            if opts.cancel is not None and opts.cancel.cancelled:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Anthropic stream ended without a stop reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            out_stream.end()
        except Exception as error:
            output.stop_reason = "aborted" if opts.cancel is not None and opts.cancel.cancelled else "error"
            output.error_message = str(error) if str(error) else repr(error)
            out_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
            out_stream.end()

    tonio.spawn.without_tracking(_run())
    return out_stream


def _map_thinking_level_to_effort(model: Model, level: ThinkingLevel | None) -> AnthropicEffort:
    """Map a pi thinking level to an Anthropic adaptive-thinking effort."""
    mapped = (model.thinking_level_map or {}).get(level) if level else None
    if isinstance(mapped, str):
        return mapped  # type: ignore[return-value]

    if level in ("minimal", "low"):
        return "low"
    if level == "medium":
        return "medium"
    return "high"


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    _assert_request_auth(model.provider, options.api_key if options else None, options.headers if options else None)

    base = build_base_options(model, context, options, options.api_key if options else None)

    def with_thinking(**extra) -> AnthropicOptions:
        anthropic = _anthropic_options(base)
        for key, value in extra.items():
            setattr(anthropic, key, value)
        return anthropic

    if options is None or not options.reasoning:
        return stream(model, context, with_thinking(thinking_enabled=False))

    # Adaptive-thinking models take an effort level; older models take a budget.
    if _force_adaptive_thinking(model):
        effort = _map_thinking_level_to_effort(model, options.reasoning)
        return stream(model, context, with_thinking(thinking_enabled=True, effort=effort))

    # None means the caller did not request an output cap; the helper uses the
    # model cap (never coerce to 0, or thinking would swallow max_tokens).
    adjusted_max, thinking_budget = adjust_max_tokens_for_thinking(
        base.max_tokens,
        model.max_tokens,
        options.reasoning,
        options.thinking_budgets,
    )
    max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max)

    return stream(
        model,
        context,
        with_thinking(
            max_tokens=max_tokens,
            thinking_enabled=True,
            thinking_budget_tokens=min(thinking_budget, max(0, max_tokens - 1024)),
        ),
    )


def _normalize_tool_call_id(id: str, model: Model, source: AssistantMessage) -> str:
    """Normalize tool call IDs to Anthropic's required pattern and length."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", id)[:64]


def _convert_tool_result(
    msg: ToolResultMessage,
    is_oauth_token: bool,
    deferred_tool_names: set[str],
    loaded_tool_names: set[str],
    normalize_tool_name,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    references: list[dict[str, str]] = []
    for name in msg.added_tool_names or []:
        normalized_name = normalize_tool_name(name)
        if normalized_name not in deferred_tool_names or normalized_name in loaded_tool_names:
            continue
        loaded_tool_names.add(normalized_name)
        references.append(
            {"type": "tool_reference", "tool_name": _to_claude_code_name(name) if is_oauth_token else name}
        )

    converted_content = _convert_content_blocks(msg.content)
    # Anthropic rejects tool references mixed with ordinary tool-result content.
    tool_result = {
        "type": "tool_result",
        "tool_use_id": msg.tool_call_id,
        "content": references if references else converted_content,
        "is_error": msg.is_error,
    }
    if not references:
        sibling_content: list[dict[str, Any]] = []
    elif isinstance(converted_content, str):
        sibling_content = [{"type": "text", "text": converted_content}]
    else:
        sibling_content = converted_content
    return tool_result, sibling_content


def _convert_messages(
    transformed_messages: list[Message],
    is_oauth_token: bool,
    cache_control: dict | None,
    allow_empty_signature: bool,
    deferred_tool_names: set[str],
    normalize_tool_name,
) -> list[dict[str, Any]]:
    params: list[dict[str, Any]] = []
    loaded_tool_names: set[str] = set()

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]

        if msg.role == "user":
            if isinstance(msg.content, str):
                if msg.content.strip():
                    params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                blocks: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        if item.text.strip():
                            blocks.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        blocks.append(
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": item.mime_type, "data": item.data},
                            }
                        )
                if blocks:
                    params.append({"role": "user", "content": blocks})
        elif msg.role == "assistant":
            blocks = []
            for block in msg.content:
                if block.type == "text":
                    if not block.text.strip():
                        continue
                    blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
                elif block.type == "thinking":
                    # Redacted thinking: pass the opaque payload back verbatim.
                    if block.redacted:
                        blocks.append({"type": "redacted_thinking", "data": block.thinking_signature})
                        continue
                    signature = block.thinking_signature
                    has_signature = bool(signature and signature.strip())
                    if not block.thinking.strip() and not has_signature:
                        continue
                    # Missing/empty signature (e.g. aborted stream): convert to
                    # plain text unless the model is marked to accept "".
                    if not has_signature:
                        if allow_empty_signature:
                            blocks.append(
                                {"type": "thinking", "thinking": sanitize_surrogates(block.thinking), "signature": ""}
                            )
                        else:
                            blocks.append({"type": "text", "text": sanitize_surrogates(block.thinking)})
                    else:
                        blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(block.thinking),
                                "signature": signature,
                            }
                        )
                elif block.type == "toolCall":
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": _to_claude_code_name(block.name) if is_oauth_token else block.name,
                            "input": block.arguments if block.arguments is not None else {},
                        }
                    )
            if blocks:
                params.append({"role": "assistant", "content": blocks})
        elif msg.role == "toolResult":
            # Collect consecutive toolResult messages (z.ai Anthropic endpoint).
            tool_results: list[dict[str, Any]] = []
            sibling_content: list[dict[str, Any]] = []
            j = i
            while j < len(transformed_messages) and transformed_messages[j].role == "toolResult":
                tool_result, siblings = _convert_tool_result(
                    transformed_messages[j],  # type: ignore[arg-type]
                    is_oauth_token,
                    deferred_tool_names,
                    loaded_tool_names,
                    normalize_tool_name,
                )
                tool_results.append(tool_result)
                sibling_content.extend(siblings)
                j += 1
            i = j - 1

            # Displaced reference-bearing results must follow every tool_result block.
            params.append({"role": "user", "content": [*tool_results, *sibling_content]})
        i += 1

    # Cache the conversation history via the last user message.
    if cache_control and params:
        last_message = params[-1]
        if last_message["role"] == "user":
            if isinstance(last_message["content"], list):
                if last_message["content"]:
                    last_block = last_message["content"][-1]
                    if last_block.get("type") in ("text", "image", "tool_result"):
                        last_block["cache_control"] = cache_control
            elif isinstance(last_message["content"], str):
                last_message["content"] = [
                    {"type": "text", "text": last_message["content"], "cache_control": cache_control}
                ]

    return params


def _convert_tools(
    tools: list[Tool],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: dict | None,
    defer_loading: bool = False,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools)
        schema = tool.parameters or {}
        legacy_input_schema = {
            "type": "object",
            "properties": schema.get("properties") or {},
            "required": schema.get("required") or [],
        }
        input_schema = {**schema, **legacy_input_schema} if strict is True else legacy_input_schema

        entry: dict[str, Any] = {
            "name": _to_claude_code_name(tool.name) if is_oauth_token else tool.name,
            "description": tool.description,
        }
        if supports_eager_tool_input_streaming:
            entry["eager_input_streaming"] = True
        if strict is True:
            entry["strict"] = True
        entry["input_schema"] = input_schema
        if defer_loading:
            entry["defer_loading"] = True
        if cache_control and index == len(tools) - 1:
            entry["cache_control"] = cache_control
        converted.append(entry)
    return converted


def _build_params(model: Model, context: Context, is_oauth_token: bool, options: AnthropicOptions) -> dict[str, Any]:
    cache_control = _get_cache_control(model, options.cache_retention, options.env)
    compat = _get_compat(model)
    transformed_messages = transform_messages(context.messages, model, _normalize_tool_call_id)
    normalize_tool_name = _to_claude_code_name if is_oauth_token else (lambda name: name)
    immediate_tools, deferred_map = split_deferred_tools(
        Context(messages=transformed_messages, system_prompt=context.system_prompt, tools=context.tools),
        compat.supports_tool_references,
        normalize_tool_name,
    )
    deferred_tools = list(deferred_map.values())
    if not immediate_tools and deferred_tools:
        immediate_tools = deferred_tools
        deferred_tools = []
    deferred_tool_names = {normalize_tool_name(tool.name) for tool in deferred_tools}

    params: dict[str, Any] = {
        "model": model.id,
        "messages": _convert_messages(
            transformed_messages,
            is_oauth_token,
            cache_control,
            compat.allow_empty_signature,
            deferred_tool_names,
            normalize_tool_name,
        ),
        "max_tokens": options.max_tokens if options.max_tokens is not None else model.max_tokens,
        "stream": True,
    }

    # For OAuth tokens, we MUST include Claude Code identity.
    if is_oauth_token:
        params["system"] = [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                **({"cache_control": cache_control} if cache_control else {}),
            }
        ]
        if context.system_prompt:
            params["system"].append(
                {
                    "type": "text",
                    "text": sanitize_surrogates(context.system_prompt),
                    **({"cache_control": cache_control} if cache_control else {}),
                }
            )
    elif context.system_prompt:
        params["system"] = [
            {
                "type": "text",
                "text": sanitize_surrogates(context.system_prompt),
                **({"cache_control": cache_control} if cache_control else {}),
            }
        ]

    # Temperature is incompatible with extended thinking and unsupported on Opus 4.7+.
    if options.temperature is not None and not options.thinking_enabled and compat.supports_temperature:
        params["temperature"] = options.temperature

    if immediate_tools or deferred_tools:
        params["tools"] = [
            *_convert_tools(
                immediate_tools,
                is_oauth_token,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                cache_control if compat.supports_cache_control_on_tools else None,
            ),
            *_convert_tools(
                deferred_tools,
                is_oauth_token,
                compat.supports_eager_tool_input_streaming,
                compat.supports_strict_tools,
                None,
                True,
            ),
        ]

    # Thinking mode: adaptive, budget-based, or explicitly disabled.
    if model.reasoning:
        if options.thinking_enabled:
            # Default "summarized" so Opus 4.7/Mythos Preview behave like older
            # Claude 4 models (whose API default is also "summarized").
            display = options.thinking_display or "summarized"
            if _force_adaptive_thinking(model):
                params["thinking"] = {"type": "adaptive", "display": display}
                if options.effort:
                    params["output_config"] = {"effort": options.effort}
            else:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": options.thinking_budget_tokens or 1024,
                    "display": display,
                }
        elif options.thinking_enabled is False and (model.thinking_level_map or {}).get("off", "") is not None:
            params["thinking"] = {"type": "disabled"}

    if options.metadata:
        user_id = options.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    if options.tool_choice:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    return params
