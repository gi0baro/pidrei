"""Port of pi's openai-completions adapter (packages/ai/src/api/openai-completions.ts).

The compat-matrix adapter that ~20 OpenAI-compatible providers ride on:
URL/provider-based compat auto-detection with per-model overrides, nine
thinking formats, Anthropic-style cache_control passthrough, grammar (custom)
tools, deferred kimi tools, and a chunk state machine tolerant of provider
quirks (usage in choice, reasoning fields, missing ids).

Transport: pi uses the openai SDK purely as an HTTP/SSE carrier; pidrei posts
through the punkreq seam and parses the SSE chunk stream itself (data events,
`[DONE]` terminator, error chunks). Tests inject a fake client via
`OpenAICompletionsOptions.client`.
"""

import json
import re
import time
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, fields
from typing import Any, Protocol

from pidrei_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from pidrei_ai.api.github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
from pidrei_ai.api.openai_prompt_cache import clamp_openai_prompt_cache_key
from pidrei_ai.api.simple_options import MIN_ANSWER_TOKENS, build_base_options, clamp_reasoning
from pidrei_ai.api.transform_messages import transform_messages
from pidrei_ai.registry import calculate_cost, clamp_thinking_level
from pidrei_ai.types import (
    AssistantMessage,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    OpenAICompletionsCompat,
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
    ThinkingBudgets,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from pidrei_ai.utils import http
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.hash import short_hash
from pidrei_ai.utils.json_parse import parse_streaming_json
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates
from pidrei_ai.utils.sse import iterate_sse_messages
from pidrei_ai.utils.user_agent import force_user_agent


@dataclass(slots=True)
class OpenAICompletionsOptions(StreamOptions):
    tool_choice: Any = None
    reasoning_effort: str | None = None  # "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
    # Token budgets per thinking level. Only used when `compat.supports_thinking_token_budget` is set.
    thinking_budgets: ThinkingBudgets | None = None
    # Pre-built client instance (test injection / alternative transports).
    client: OpenAICompletionsClient | None = None


class OpenAIResponseLike(Protocol):
    status: int
    headers: dict[str, str]

    def aiter_bytes(self) -> AsyncIterable[bytes]: ...


class OpenAICompletionsClient(Protocol):
    async def create(
        self,
        params: dict[str, Any],
        *,
        timeout_ms: float | None,
        cancel: CancelToken | None,
    ) -> OpenAIResponseLike: ...


class OpenAIApiError(Exception):
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


def _parse_error_body(body: str) -> dict | None:
    try:
        parsed = json.loads(body)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_error_message(body: str) -> str:
    parsed = _parse_error_body(body)
    if parsed is not None:
        error = parsed.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            return error["message"]
    return body


class _PunkreqOpenAIClient:
    """Default transport: POST {base_url}/chat/completions through the punkreq seam."""

    def __init__(self, base_url: str, headers: dict[str, str], env: ProviderEnv | None = None):
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._headers = headers
        self._env = env

    async def create(
        self,
        params: dict[str, Any],
        *,
        timeout_ms: float | None,
        cancel: CancelToken | None,
    ) -> OpenAIResponseLike:
        client = http.client_for(self._url, self._env)
        timeout = http.request_timeout(timeout_ms)
        response = await client.post(self._url, json=params, headers=self._headers, timeout=timeout)
        if not 200 <= response.status_code < 300:
            body = (await response.read()).decode("utf-8", "replace")
            error_body = _parse_error_body(body)
            error_field = error_body.get("error") if error_body else None
            raise OpenAIApiError(
                status=response.status_code,
                headers=dict(response.headers),
                message=f"{response.status_code} {_extract_error_message(body)}",
                error=error_field if isinstance(error_field, dict) else None,
            )
        return _PunkreqResponse(status=response.status_code, headers=dict(response.headers), _response=response)


# --- compat resolution --------------------------------------------------------


@dataclass(slots=True)
class _ResolvedCompat:
    supports_store: bool
    supports_developer_role: bool
    supports_reasoning_effort: bool
    supports_usage_in_streaming: bool
    supports_finish_reason: bool
    max_tokens_field: str
    requires_tool_result_name: bool
    requires_assistant_after_tool_result: bool
    requires_thinking_as_text: bool
    requires_reasoning_content_on_assistant_messages: bool
    thinking_format: str
    open_router_routing: dict
    vercel_gateway_routing: dict
    chat_template_kwargs: dict
    chat_template_args: dict
    zai_tool_stream: bool
    supports_strict_mode: bool
    supports_openai_grammar_tools: bool
    supports_thinking_token_budget: bool
    cache_control_format: str | None
    send_session_affinity_headers: bool
    deferred_tools_mode: str | None
    session_affinity_format: str
    supports_long_cache_retention: bool


def detect_compat(model: Model) -> _ResolvedCompat:
    """Auto-detect compatibility settings from provider name and baseUrl."""
    provider = model.provider
    base_url = model.base_url

    is_zai = provider in ("zai", "zai-coding-cn") or "api.z.ai" in base_url or "open.bigmodel.cn" in base_url
    is_together = provider == "together" or "api.together.ai" in base_url or "api.together.xyz" in base_url
    is_moonshot = provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    is_cloudflare_ai_gateway = provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url.lower()

    is_non_standard = (
        is_nvidia
        or provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or is_deepseek
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )

    use_max_tokens = (
        "chutes.ai" in base_url
        or is_deepseek
        or is_moonshot
        or is_cloudflare_ai_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_openrouter_developer_role_model = is_openrouter and (
        model.id.startswith("anthropic/") or model.id.startswith("openai/")
    )
    cache_control_format = "anthropic" if provider == "openrouter" and model.id.startswith("anthropic/") else None

    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_openrouter:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    return _ResolvedCompat(
        supports_store=not is_non_standard,
        supports_developer_role=is_openrouter_developer_role_model or (not is_non_standard and not is_openrouter),
        supports_reasoning_effort=(
            not is_grok
            and not is_zai
            and not is_moonshot
            and not is_together
            and not is_cloudflare_ai_gateway
            and not is_nvidia
            and not is_ant_ling
        ),
        supports_usage_in_streaming=True,
        supports_finish_reason=True,
        max_tokens_field="max_tokens" if use_max_tokens else "max_completion_tokens",
        requires_tool_result_name=False,
        requires_assistant_after_tool_result=False,
        requires_thinking_as_text=False,
        requires_reasoning_content_on_assistant_messages=is_deepseek,
        thinking_format=thinking_format,
        open_router_routing={},
        vercel_gateway_routing={},
        chat_template_kwargs={},
        chat_template_args={},
        zai_tool_stream=False,
        supports_strict_mode=not is_moonshot and not is_together and not is_cloudflare_ai_gateway and not is_nvidia,
        supports_openai_grammar_tools=False,
        supports_thinking_token_budget=False,
        cache_control_format=cache_control_format,
        send_session_affinity_headers=False,
        deferred_tools_mode=None,
        session_affinity_format="openrouter" if is_openrouter else "openai",
        supports_long_cache_retention=not (
            is_together or is_cloudflare_workers_ai or is_cloudflare_ai_gateway or is_nvidia or is_ant_ling
        ),
    )


def get_compat(model: Model) -> _ResolvedCompat:
    """Auto-detect from provider/URL then override with explicit model.compat."""
    detected = detect_compat(model)
    compat = model.compat if isinstance(model.compat, OpenAICompletionsCompat) else None
    if compat is None:
        return detected

    def pick(value, fallback):
        return value if value is not None else fallback

    return _ResolvedCompat(
        supports_store=pick(compat.supports_store, detected.supports_store),
        supports_developer_role=pick(compat.supports_developer_role, detected.supports_developer_role),
        supports_reasoning_effort=pick(compat.supports_reasoning_effort, detected.supports_reasoning_effort),
        supports_usage_in_streaming=pick(compat.supports_usage_in_streaming, detected.supports_usage_in_streaming),
        supports_finish_reason=pick(compat.supports_finish_reason, detected.supports_finish_reason),
        max_tokens_field=pick(compat.max_tokens_field, detected.max_tokens_field),
        requires_tool_result_name=pick(compat.requires_tool_result_name, detected.requires_tool_result_name),
        requires_assistant_after_tool_result=pick(
            compat.requires_assistant_after_tool_result, detected.requires_assistant_after_tool_result
        ),
        requires_thinking_as_text=pick(compat.requires_thinking_as_text, detected.requires_thinking_as_text),
        requires_reasoning_content_on_assistant_messages=pick(
            compat.requires_reasoning_content_on_assistant_messages,
            detected.requires_reasoning_content_on_assistant_messages,
        ),
        thinking_format=pick(compat.thinking_format, detected.thinking_format),
        # pi quirk: explicit-compat models fall back to {} here, not to `detected`.
        open_router_routing=dict(compat.open_router_routing) if compat.open_router_routing is not None else {},
        vercel_gateway_routing=(
            dict(compat.vercel_gateway_routing)
            if compat.vercel_gateway_routing is not None
            else detected.vercel_gateway_routing
        ),
        chat_template_kwargs=(
            dict(compat.chat_template_kwargs)
            if compat.chat_template_kwargs is not None
            else detected.chat_template_kwargs
        ),
        chat_template_args=(
            dict(compat.chat_template_args) if compat.chat_template_args is not None else detected.chat_template_args
        ),
        zai_tool_stream=pick(compat.zai_tool_stream, detected.zai_tool_stream),
        supports_strict_mode=pick(compat.supports_strict_mode, detected.supports_strict_mode),
        supports_openai_grammar_tools=pick(
            compat.supports_openai_grammar_tools, detected.supports_openai_grammar_tools
        ),
        supports_thinking_token_budget=pick(
            compat.supports_thinking_token_budget, detected.supports_thinking_token_budget
        ),
        cache_control_format=pick(compat.cache_control_format, detected.cache_control_format),
        send_session_affinity_headers=pick(
            compat.send_session_affinity_headers, detected.send_session_affinity_headers
        ),
        deferred_tools_mode=pick(compat.deferred_tools_mode, detected.deferred_tools_mode),
        session_affinity_format=pick(compat.session_affinity_format, detected.session_affinity_format),
        supports_long_cache_retention=pick(
            compat.supports_long_cache_retention, detected.supports_long_cache_retention
        ),
    )


# --- helpers ------------------------------------------------------------------


def _has_header(headers: ProviderHeaders | None, name: str) -> bool:
    if not headers:
        return False
    expected = name.lower()
    return any(key.lower() == expected and value is not None and value.strip() for key, value in headers.items())


def _get_client_api_key(provider: str, api_key: str | None, headers: ProviderHeaders | None) -> str:
    if api_key:
        return api_key
    if _has_header(headers, "authorization") or _has_header(headers, "cf-aig-authorization"):
        return "unused"
    raise RuntimeError(f"No API key for provider: {provider}")


def _has_tool_history(messages: list[Message]) -> bool:
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant" and any(block.type == "toolCall" for block in msg.content):
            return True
    return False


def _get_deferred_tool_names(messages: list[Message]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        if message.role == "toolResult":
            for name in message.added_tool_names or []:
                names.add(name)
    return names


def _get_tools_by_name(tools: list[Tool] | None, names) -> list[Tool]:
    if not tools:
        return []
    by_name = {tool.name: tool for tool in tools}
    return [by_name[name] for name in names if name in by_name]


def _is_encrypted_reasoning_detail(detail: Any) -> bool:
    return (
        isinstance(detail, dict)
        and detail.get("type") == "reasoning.encrypted"
        and isinstance(detail.get("id"), str)
        and len(detail["id"]) > 0
        and isinstance(detail.get("data"), str)
        and len(detail["data"]) > 0
    )


def _resolve_cache_retention(cache_retention: CacheRetention | None, env: ProviderEnv | None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PIDREI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


_INCLUDE_NULL = object()
_MISSING = object()


def _thinking_map(model: Model) -> dict:
    return dict(model.thinking_level_map) if model.thinking_level_map is not None else {}


def _off_is_not_null(model: Model) -> bool:
    """pi: `model.thinkingLevelMap?.off !== null`."""
    mapping = _thinking_map(model)
    return not ("off" in mapping and mapping["off"] is None)


def _map_effort(model: Model, effort: str) -> str | None:
    """pi: `model.thinkingLevelMap?.[effort] ?? effort` (nullish coalescing)."""
    mapped = _thinking_map(model).get(effort)
    return mapped if mapped is not None else effort


def _map_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    if reason is None:
        return "stop", None
    if reason in ("stop", "end"):
        return "stop", None
    if reason == "length":
        return "length", None
    if reason in ("function_call", "tool_calls"):
        return "toolUse", None
    if reason == "content_filter":
        return "error", "Provider finish_reason: content_filter"
    if reason == "network_error":
        return "error", "Provider finish_reason: network_error"
    return "error", f"Provider finish_reason: {reason}"


def _parse_chunk_usage(raw_usage: dict, model: Model) -> Usage:
    prompt_tokens = raw_usage.get("prompt_tokens") or 0
    prompt_details = raw_usage.get("prompt_tokens_details") or {}
    # pi's `?? ?? ??` chain: providers disagree on placement, and a present zero
    # must win over the next candidate.
    cache_read_tokens = next(
        (
            value
            for value in (
                prompt_details.get("cached_tokens"),
                raw_usage.get("prompt_cache_hit_tokens"),
                raw_usage.get("cached_tokens"),
            )
            if value is not None
        ),
        0,
    )
    cache_write_tokens = prompt_details.get("cache_write_tokens") or 0

    # cached_tokens is cache-read (hits). Providers disagree on placement:
    # OpenAI/OpenRouter use prompt_tokens_details.cached_tokens, DeepSeek uses
    # prompt_cache_hit_tokens, and Kimi documents top-level usage.cached_tokens
    # on the final usage chunk. cache_write_tokens is a separate
    # OpenRouter-compatible write count. Never subtract writes from reads.
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = raw_usage.get("completion_tokens") or 0
    completion_details = raw_usage.get("completion_tokens_details") or {}
    usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read_tokens,
        cache_write=cache_write_tokens,
        reasoning=completion_details.get("reasoning_tokens") or 0,
        total_tokens=input_tokens + output_tokens + cache_read_tokens + cache_write_tokens,
    )
    calculate_cost(model, usage)
    return usage


async def _iterate_chunks(response: OpenAIResponseLike, cancel: CancelToken | None) -> AsyncGenerator[dict]:
    """SSE chunk iteration mirroring the openai SDK stream: data events, a
    `[DONE]` terminator, and error chunks raised as stream errors."""
    body = response.aiter_bytes()
    ended = False
    try:
        async for sse in iterate_sse_messages(body):
            if sse.event == "error":
                raise RuntimeError(sse.data)
            if sse.data == "[DONE]":
                ended = True
                return
            chunk = json.loads(sse.data)
            if isinstance(chunk, dict) and chunk.get("error"):
                error = chunk["error"]
                message = error.get("message") if isinstance(error, dict) else None
                raise RuntimeError(message or json.dumps(error, ensure_ascii=False))
            yield chunk
        ended = True
    finally:
        await http.finish_body(body, response, drain=ended)


# --- client / params ----------------------------------------------------------


def _create_client(
    model: Model,
    context: Context,
    api_key: str,
    options_headers: ProviderHeaders | None,
    session_id: str | None,
    compat: _ResolvedCompat,
    env: ProviderEnv | None = None,
) -> OpenAICompletionsClient:
    headers: dict[str, Any] = dict(model.headers or {})
    if model.provider == "github-copilot":
        headers.update(build_copilot_dynamic_headers(context.messages, has_copilot_vision_input(context.messages)))

    if session_id and compat.send_session_affinity_headers:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
            headers["x-session-affinity"] = session_id

    # Merge options headers last so they can override defaults.
    if options_headers:
        headers.update(options_headers)

    if model.provider == "xai":
        force_user_agent(headers)

    headers["authorization"] = f"Bearer {api_key}"
    final_headers = {key: value for key, value in headers.items() if value is not None}
    return _PunkreqOpenAIClient(model.base_url, final_headers, env)


def _get_compat_cache_control(compat: _ResolvedCompat, cache_retention: CacheRetention) -> dict | None:
    if compat.cache_control_format != "anthropic" or cache_retention == "none":
        return None
    cache_control: dict[str, str] = {"type": "ephemeral"}
    if cache_retention == "long" and compat.supports_long_cache_retention:
        cache_control["ttl"] = "1h"
    return cache_control


def _add_cache_control_to_text_content(message: dict, cache_control: dict) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        if not content:
            return False
        message["content"] = [{"type": "text", "text": content, "cache_control": cache_control}]
        return True

    if not isinstance(content, list):
        return False

    for part in reversed(content):
        if isinstance(part, dict) and part.get("type") == "text":
            part["cache_control"] = cache_control
            return True
    return False


def _apply_anthropic_cache_control(messages: list[dict], tools: list[dict] | None, cache_control: dict) -> None:
    for message in messages:
        if message.get("role") in ("system", "developer"):
            _add_cache_control_to_text_content(message, cache_control)
            break

    if tools:
        tools[-1]["cache_control"] = cache_control

    for message in reversed(messages):
        if message.get("role") in ("user", "assistant", "tool") and _add_cache_control_to_text_content(
            message, cache_control
        ):
            break


def _resolve_chat_template_kwarg_value(model: Model, options: OpenAICompletionsOptions, value: Any) -> Any:
    """Returns the resolved kwarg, `_INCLUDE_NULL` for a literal null, or None to omit."""
    if not isinstance(value, dict):
        return _INCLUDE_NULL if value is None else value

    reasoning_effort = options.reasoning_effort
    if not reasoning_effort and value.get("omitWhenOff"):
        return None
    if value.get("$var") == "thinking.enabled":
        return bool(reasoning_effort)

    mapping = _thinking_map(model)
    lookup_key = reasoning_effort if reasoning_effort else "off"
    mapped_value = mapping.get(lookup_key, _MISSING)
    if mapped_value is _MISSING:
        return reasoning_effort  # may be None -> omitted, mirroring JS undefined
    return mapped_value if isinstance(mapped_value, str) else None


def _build_chat_template_values(model: Model, options: OpenAICompletionsOptions, values: dict[str, Any]) -> dict | None:
    resolved_values: dict[str, Any] = {}
    for key, value in values.items():
        resolved = _resolve_chat_template_kwarg_value(model, options, value)
        if resolved is None:
            continue
        resolved_values[key] = None if resolved is _INCLUDE_NULL else resolved
    return resolved_values if resolved_values else None


def convert_tools(tools: list[Tool], compat: _ResolvedCompat) -> list[dict]:
    converted: list[dict] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, compat.supports_openai_grammar_tools)
        if grammar:
            converted.append(
                {
                    "type": "custom",
                    "custom": {
                        "name": tool.name,
                        "description": tool.description,
                        "format": {
                            "type": "grammar",
                            "grammar": {"syntax": grammar.format, "definition": grammar.definition},
                        },
                    },
                }
            )
            continue

        strict = resolve_json_schema_strict_sampling(tool, compat.supports_strict_mode is not False)
        entry: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": get_json_schema_tool_parameters(tool, strict),
            },
        }
        # Only include strict if the provider supports it; some reject unknown fields.
        if compat.supports_strict_mode is not False:
            entry["function"]["strict"] = strict if strict is not None else False
        converted.append(entry)
    return converted


def convert_messages(
    model: Model,
    context: Context,
    compat: _ResolvedCompat,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> list[dict]:
    grammar_tool_input_properties = grammar_tool_input_properties or {}
    params: list[dict] = []

    def normalize_tool_call_id(id: str) -> str:
        # Pipe-separated IDs from the Responses API: {call_id}|{item_id}. Keep
        # item-level uniqueness while sanitizing to allowed chars, 40-char cap.
        if "|" in id:
            separator_index = id.index("|")
            call_id = re.sub(r"[^a-zA-Z0-9_-]", "_", id[:separator_index])
            item_id = re.sub(r"[^a-zA-Z0-9_-]", "_", id[separator_index + 1 :])
            combined_id = f"{call_id}_{item_id}" if item_id else call_id
            if len(combined_id) <= 40:
                return combined_id
            hashed = short_hash(id)[:8]
            prefix = call_id[: max(1, 40 - len(hashed) - 1)]
            return f"{prefix}_{hashed}"

        if model.provider == "openai":
            return id[:40] if len(id) > 40 else id
        return id

    transformed_messages = transform_messages(
        context.messages, model, lambda id, _model, _source: normalize_tool_call_id(id)
    )

    if context.system_prompt:
        use_developer_role = model.reasoning and compat.supports_developer_role
        role = "developer" if use_developer_role else "system"
        params.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    last_role: str | None = None

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]
        # Some providers don't allow user messages directly after tool results;
        # insert a synthetic assistant message to bridge the gap.
        if compat.requires_assistant_after_tool_result and last_role == "toolResult" and msg.role == "user":
            params.append({"role": "assistant", "content": "I have processed the tool results."})

        if msg.role == "user":
            if isinstance(msg.content, str):
                params.append({"role": "user", "content": sanitize_surrogates(msg.content)})
            else:
                content = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({"type": "text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {"type": "image_url", "image_url": {"url": f"data:{item.mime_type};base64,{item.data}"}}
                        )
                if not content:
                    i += 1
                    continue
                params.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            # Some providers don't accept null content; use empty string instead.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "" if compat.requires_assistant_after_tool_result else None,
            }

            assistant_text_parts = [
                {"type": "text", "text": sanitize_surrogates(block.text)}
                for block in msg.content
                if block.type == "text" and block.text.strip()
            ]
            assistant_text = "".join(part["text"] for part in assistant_text_parts)

            non_empty_thinking = [block for block in msg.content if block.type == "thinking" and block.thinking.strip()]
            if non_empty_thinking:
                if compat.requires_thinking_as_text:
                    # Convert thinking blocks to plain text (no tags to avoid mimicry).
                    thinking_text = "\n\n".join(sanitize_surrogates(block.thinking) for block in non_empty_thinking)
                    assistant_msg["content"] = [{"type": "text", "text": thinking_text}, *assistant_text_parts]
                else:
                    # Assistant content always goes as a plain string (standard Chat
                    # Completions shape; arrays cause some models to mimic the structure).
                    if assistant_text:
                        assistant_msg["content"] = assistant_text

                    signature = non_empty_thinking[0].thinking_signature
                    if model.provider == "opencode-go" and signature == "reasoning":
                        signature = "reasoning_content"
                    if signature:
                        assistant_msg[signature] = "\n".join(block.thinking for block in non_empty_thinking)
            elif assistant_text:
                assistant_msg["content"] = assistant_text

            tool_calls = [block for block in msg.content if block.type == "toolCall"]
            if tool_calls:
                converted_calls = []
                for tc in tool_calls:
                    custom_input_property = grammar_tool_input_properties.get(tc.name)
                    if custom_input_property is not None:
                        converted_calls.append(
                            {
                                "id": tc.id,
                                "type": "custom",
                                "custom": {
                                    "name": tc.name,
                                    "input": sanitize_surrogates(
                                        get_grammar_tool_input(tc.name, tc.arguments, custom_input_property)
                                    ),
                                },
                            }
                        )
                    else:
                        converted_calls.append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                            }
                        )
                assistant_msg["tool_calls"] = converted_calls

                reasoning_details = []
                for tc in tool_calls:
                    if tc.thought_signature:
                        try:
                            reasoning_details.append(json.loads(tc.thought_signature))
                        except ValueError:
                            pass
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details

            if (
                compat.requires_reasoning_content_on_assistant_messages
                and model.reasoning
                and "reasoning_content" not in assistant_msg
            ):
                assistant_msg["reasoning_content"] = ""

            # Skip assistant messages with no content and no tool calls (some
            # providers reject them); handles aborted responses with no content.
            content = assistant_msg["content"]
            has_content = content is not None and len(content) > 0
            if not has_content and "tool_calls" not in assistant_msg:
                i += 1
                continue
            params.append(assistant_msg)
        elif msg.role == "toolResult":
            image_blocks: list[dict] = []
            deferred_tool_names: set[str] = set()
            j = i

            while j < len(transformed_messages) and transformed_messages[j].role == "toolResult":
                tool_msg = transformed_messages[j]

                text_result = "\n".join(block.text for block in tool_msg.content if block.type == "text")
                has_images = any(block.type == "image" for block in tool_msg.content)

                tool_result_text = (
                    text_result if text_result else "(see attached image)" if has_images else "(no tool output)"
                )
                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "content": sanitize_surrogates(tool_result_text),
                    "tool_call_id": tool_msg.tool_call_id,
                }
                if compat.requires_tool_result_name and tool_msg.tool_name:
                    tool_result_msg["name"] = tool_msg.tool_name
                params.append(tool_result_msg)

                if compat.deferred_tools_mode == "kimi":
                    for name in tool_msg.added_tool_names or []:
                        deferred_tool_names.add(name)

                if has_images and "image" in model.input:
                    for block in tool_msg.content:
                        if block.type == "image":
                            image_blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{block.mime_type};base64,{block.data}"},
                                }
                            )
                j += 1

            i = j - 1

            if image_blocks:
                if compat.requires_assistant_after_tool_result:
                    params.append({"role": "assistant", "content": "I have processed the tool results."})
                params.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Attached image(s) from tool result:"}, *image_blocks],
                    }
                )
                last_role = "user"
            else:
                last_role = "toolResult"

            if deferred_tool_names:
                deferred_tools = _get_tools_by_name(context.tools, deferred_tool_names)
                if deferred_tools:
                    # Kimi accepts a system message with tools but no content field.
                    params.append({"role": "system", "tools": convert_tools(deferred_tools, compat)})
            i += 1
            continue

        last_role = msg.role
        i += 1

    return params


def build_params(  # noqa: C901 (mirrors pi's compat ladder)
    model: Model,
    context: Context,
    options: OpenAICompletionsOptions,
    compat: _ResolvedCompat | None = None,
    cache_retention: CacheRetention | None = None,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict:
    compat = compat if compat is not None else get_compat(model)
    cache_retention = (
        cache_retention
        if cache_retention is not None
        else _resolve_cache_retention(options.cache_retention, options.env)
    )
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )

    messages = convert_messages(model, context, compat, grammar_tool_input_properties)
    cache_control = _get_compat_cache_control(compat, cache_retention)

    params: dict[str, Any] = {"model": model.id, "messages": messages, "stream": True}

    if ("api.openai.com" in model.base_url and cache_retention != "none") or (
        cache_retention == "long" and compat.supports_long_cache_retention
    ):
        cache_key = clamp_openai_prompt_cache_key(options.session_id)
        if cache_key is not None:
            params["prompt_cache_key"] = cache_key
    if cache_retention == "long" and compat.supports_long_cache_retention:
        params["prompt_cache_retention"] = "24h"

    if compat.supports_usage_in_streaming is not False:
        params["stream_options"] = {"include_usage": True}

    if compat.supports_store:
        params["store"] = False

    if options.max_tokens:
        params[compat.max_tokens_field] = options.max_tokens

    if options.temperature is not None:
        params["temperature"] = options.temperature

    deferred_tool_names = _get_deferred_tool_names(context.messages) if compat.deferred_tools_mode == "kimi" else set()
    active_tools = [tool for tool in context.tools or [] if tool.name not in deferred_tool_names]
    if active_tools:
        params["tools"] = convert_tools(active_tools, compat)
        if compat.zai_tool_stream:
            params["tool_stream"] = True
    elif _has_tool_history(context.messages):
        # Anthropic (via LiteLLM/proxy) requires tools param when the
        # conversation contains tool_calls/tool_results.
        params["tools"] = []

    if cache_control:
        _apply_anthropic_cache_control(messages, params.get("tools"), cache_control)

    if options.tool_choice:
        params["tool_choice"] = options.tool_choice

    effort = options.reasoning_effort
    mapping = _thinking_map(model)
    if compat.thinking_format == "zai" and model.reasoning:
        params["thinking"] = {"type": "enabled", "clear_thinking": False} if effort else {"type": "disabled"}
        if effort and compat.supports_reasoning_effort:
            mapped_effort = mapping.get(effort, "__missing__")
            resolved = effort if mapped_effort == "__missing__" else mapped_effort
            if isinstance(resolved, str):
                params["reasoning_effort"] = resolved
    elif compat.thinking_format == "qwen" and model.reasoning:
        params["enable_thinking"] = bool(effort)
        if effort and compat.supports_reasoning_effort:
            mapped_effort = mapping.get(effort, "__missing__")
            resolved = effort if mapped_effort == "__missing__" else mapped_effort
            if isinstance(resolved, str):
                params["reasoning_effort"] = resolved
    elif compat.thinking_format == "qwen-chat-template" and model.reasoning:
        params["chat_template_kwargs"] = {"enable_thinking": bool(effort), "preserve_thinking": True}
    elif compat.thinking_format == "chat-template" and model.reasoning:
        chat_template_kwargs = _build_chat_template_values(model, options, compat.chat_template_kwargs)
        if chat_template_kwargs:
            params["chat_template_kwargs"] = chat_template_kwargs
    elif compat.thinking_format == "baseten" and model.reasoning:
        chat_template_args = _build_chat_template_values(model, options, compat.chat_template_args)
        if chat_template_args:
            params["chat_template_args"] = chat_template_args
        if compat.supports_reasoning_effort:
            mapped_effort = mapping.get(effort, "__missing__") if effort else mapping.get("off", "__missing__")
            resolved = effort if mapped_effort == "__missing__" else mapped_effort
            if isinstance(resolved, str):
                params["reasoning_effort"] = resolved
    elif compat.thinking_format == "deepseek" and model.reasoning:
        if effort:
            params["thinking"] = {"type": "enabled"}
        elif _off_is_not_null(model):
            params["thinking"] = {"type": "disabled"}
        if effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = _map_effort(model, effort)
    elif compat.thinking_format == "openrouter" and model.reasoning:
        # OpenRouter normalizes reasoning across providers via a nested object.
        if effort:
            params["reasoning"] = {"effort": _map_effort(model, effort)}
        elif _off_is_not_null(model):
            off_value = mapping.get("off")
            params["reasoning"] = {"effort": off_value if off_value is not None else "none"}
    elif compat.thinking_format == "ant-ling" and model.reasoning and effort:
        mapped = mapping.get(effort)
        if isinstance(mapped, str):
            params["reasoning"] = {"effort": mapped}
    elif compat.thinking_format == "together" and model.reasoning:
        params["reasoning"] = {"enabled": bool(effort)}
        if effort and compat.supports_reasoning_effort:
            params["reasoning_effort"] = _map_effort(model, effort)
    elif compat.thinking_format == "string-thinking" and model.reasoning:
        if effort:
            params["thinking"] = _map_effort(model, effort)
        elif _off_is_not_null(model):
            off_value = mapping.get("off")
            params["thinking"] = off_value if off_value is not None else "none"
    elif effort and model.reasoning and compat.supports_reasoning_effort:
        # OpenAI-style reasoning_effort.
        params["reasoning_effort"] = _map_effort(model, effort)
    elif not effort and model.reasoning and compat.supports_reasoning_effort:
        off_value = mapping.get("off")
        if isinstance(off_value, str):
            params["reasoning_effort"] = off_value

    # vLLM caps reasoning with a top-level thinking_token_budget. Independent of
    # thinking_format: the same server can serve zai, qwen or chat-template models.
    # Reasoning and the answer share max_tokens here, so an uncapped reasoning
    # phase can consume the whole response and leave no answer and no tool call.
    if compat.supports_thinking_token_budget and effort and model.reasoning:
        level = clamp_reasoning(effort)
        budgets = {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, **(options.thinking_budgets or {})}
        ceiling = params.get("max_tokens") or params.get("max_completion_tokens") or model.max_tokens
        # Always leave room for the answer, otherwise the budget recreates the bug it prevents.
        budget = min(budgets[level], max(0, ceiling - MIN_ANSWER_TOKENS))
        if budget > 0:
            params["thinking_token_budget"] = budget

    model_compat = model.compat if isinstance(model.compat, OpenAICompletionsCompat) else None
    if model_compat is not None and model_compat.open_router_routing is not None:
        params["provider"] = model_compat.open_router_routing
    if model_compat is not None and model_compat.vercel_gateway_routing is not None:
        routing = model_compat.vercel_gateway_routing
        gateway_options: dict[str, list[str]] = {}
        if routing.get("only"):
            gateway_options["only"] = routing["only"]
        if routing.get("order"):
            gateway_options["order"] = routing["order"]
        if gateway_options:
            params["providerOptions"] = {"gateway": gateway_options}

    # Last so custom keys override the named request fields.
    if options.sampling_params:
        params.update(options.sampling_params)

    return params


# --- streaming ----------------------------------------------------------------


@dataclass(slots=True)
class _ToolScratch:
    partial_args: str | None = None
    custom_property: str | None = None
    json_buffer: GrammarToolInputJsonBuffer | None = None
    stream_index: int | None = None


def _openai_options(options: StreamOptions | None) -> OpenAICompletionsOptions:
    if isinstance(options, OpenAICompletionsOptions):
        return options
    if options is None:
        return OpenAICompletionsOptions()
    values = {field.name: getattr(options, field.name) for field in fields(StreamOptions)}
    return OpenAICompletionsOptions(**values)


def stream(  # noqa: C901
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    opts = _openai_options(options)
    out_stream = into if into is not None else AssistantMessageEventStream()

    output = AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=int(time.time() * 1000),
    )
    out_stream.partial = output

    async def _run() -> None:  # noqa: C901
        try:
            compat = get_compat(model)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                context.tools, compat.supports_openai_grammar_tools
            )
            if opts.client is not None:
                client: OpenAICompletionsClient = opts.client
            else:
                api_key = _get_client_api_key(model.provider, opts.api_key, opts.headers)
                cache_retention = _resolve_cache_retention(opts.cache_retention, opts.env)
                cache_session_id = None if cache_retention == "none" else opts.session_id
                client = _create_client(model, context, api_key, opts.headers, cache_session_id, compat, opts.env)

            params = build_params(model, context, opts, compat, None, grammar_tool_input_properties)
            next_params = await maybe_call(opts.on_payload, params, model)
            if next_params is not None:
                params = next_params

            async def _request():
                return await client.create(params, timeout_ms=opts.timeout_ms, cancel=opts.cancel)

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
            scratch: dict[int, _ToolScratch] = {}

            def content_index(block) -> int:
                for index, candidate in enumerate(blocks):
                    if candidate is block:
                        return index
                return -1

            text_block: TextContent | None = None
            thinking_block: ThinkingContent | None = None
            has_finish_reason = False
            tool_blocks_by_index: dict[int, ToolCall] = {}
            tool_blocks_by_id: dict[str, ToolCall] = {}
            pending_reasoning_details: dict[str, str] = {}

            def tool_scratch(block: ToolCall) -> _ToolScratch:
                return scratch.setdefault(id(block), _ToolScratch())

            def get_custom_tool_call_input(block: ToolCall) -> str:
                prop = tool_scratch(block).custom_property
                if prop is None:
                    return ""
                value = block.arguments.get(prop)
                return value if isinstance(value, str) else ""

            def append_custom_tool_call_input(block: ToolCall, next_input: str, close: bool) -> str | None:
                entry = tool_scratch(block)
                if entry.custom_property is None or entry.json_buffer is None:
                    return None
                delta = append_grammar_tool_input_json_delta(
                    entry.json_buffer, entry.custom_property, next_input, close
                )
                block.arguments = {entry.custom_property: next_input}
                return delta

            def finish_block(block) -> None:
                index = content_index(block)
                if index == -1:
                    return
                if block.type == "text":
                    out_stream.push(TextEndEvent(content_index=index, content=block.text, partial=output))
                elif block.type == "thinking":
                    out_stream.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=output))
                elif block.type == "toolCall":
                    entry = tool_scratch(block)
                    if entry.custom_property is not None:
                        delta = append_custom_tool_call_input(block, get_custom_tool_call_input(block), True)
                        if delta is not None:
                            out_stream.push(ToolCallDeltaEvent(content_index=index, delta=delta, partial=output))
                    else:
                        block.arguments = parse_streaming_json(entry.partial_args)
                    scratch.pop(id(block), None)
                    out_stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=output))

            def ensure_text_block() -> TextContent:
                nonlocal text_block
                if text_block is None:
                    text_block = TextContent(text="")
                    blocks.append(text_block)
                    out_stream.push(TextStartEvent(content_index=content_index(text_block), partial=output))
                return text_block

            def ensure_thinking_block(thinking_signature: str) -> ThinkingContent:
                nonlocal thinking_block
                if thinking_block is None:
                    thinking_block = ThinkingContent(thinking="", thinking_signature=thinking_signature)
                    blocks.append(thinking_block)
                    out_stream.push(ThinkingStartEvent(content_index=content_index(thinking_block), partial=output))
                return thinking_block

            def apply_pending_reasoning_detail(block: ToolCall) -> None:
                if not block.id:
                    return
                pending = pending_reasoning_details.pop(block.id, None)
                if pending is not None:
                    block.thought_signature = pending

            def ensure_tool_call_block(tool_call: dict) -> ToolCall:
                stream_index = tool_call.get("index") if isinstance(tool_call.get("index"), int) else None
                function = tool_call.get("function") or {}
                custom = tool_call.get("custom")
                name = function.get("name") or (custom or {}).get("name") or ""
                block = tool_blocks_by_index.get(stream_index) if stream_index is not None else None
                if block is None and tool_call.get("id"):
                    block = tool_blocks_by_id.get(tool_call["id"])
                if block is None:
                    # The "input" fallback must not normally be taken: it exists so a
                    # made-up tool name still has somewhere to stash its input.
                    custom_input_property = (
                        grammar_tool_input_properties.get(name, "input")
                        if custom is not None and not tool_call.get("function")
                        else None
                    )
                    has_custom = custom_input_property is not None
                    block = ToolCall(
                        id=tool_call.get("id") or "",
                        name=name,
                        arguments={custom_input_property: ""} if has_custom else {},
                    )
                    entry = tool_scratch(block)
                    entry.partial_args = None if has_custom else ""
                    if has_custom:
                        entry.custom_property = custom_input_property
                        entry.json_buffer = GrammarToolInputJsonBuffer()
                    entry.stream_index = stream_index
                    if stream_index is not None:
                        tool_blocks_by_index[stream_index] = block
                    if tool_call.get("id"):
                        tool_blocks_by_id[tool_call["id"]] = block
                    blocks.append(block)
                    out_stream.push(ToolCallStartEvent(content_index=content_index(block), partial=output))
                entry = tool_scratch(block)
                if stream_index is not None and entry.stream_index is None:
                    entry.stream_index = stream_index
                    tool_blocks_by_index[stream_index] = block
                if tool_call.get("id"):
                    tool_blocks_by_id[tool_call["id"]] = block
                if not block.name and name:
                    block.name = name
                if custom is not None and not tool_call.get("function") and entry.custom_property is None:
                    custom_input_property = grammar_tool_input_properties.get(block.name, "input")
                    block.arguments = {custom_input_property: ""}
                    entry.custom_property = custom_input_property
                    entry.json_buffer = GrammarToolInputJsonBuffer()
                    entry.partial_args = None
                apply_pending_reasoning_detail(block)
                return block

            async for chunk in _iterate_chunks(response, opts.cancel):
                if not isinstance(chunk, dict):
                    continue

                # Each chunk in a streamed completion carries the same id.
                if not output.response_id:
                    output.response_id = chunk.get("id")
                chunk_model = chunk.get("model")
                if (
                    isinstance(chunk_model, str)
                    and chunk_model
                    and chunk_model != model.id
                    and not output.response_model
                ):
                    output.response_model = chunk_model
                if chunk.get("usage"):
                    output.usage = _parse_chunk_usage(chunk["usage"], model)

                choices = chunk.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else None
                if not choice:
                    continue

                # Fallback: some providers (e.g. Moonshot) return usage in choice.usage.
                if not chunk.get("usage") and choice.get("usage"):
                    output.usage = _parse_chunk_usage(choice["usage"], model)

                if choice.get("finish_reason"):
                    output.raw_stop_reason = choice["finish_reason"]
                    stop_reason, error_message = _map_stop_reason(choice["finish_reason"])
                    output.stop_reason = stop_reason
                    if error_message:
                        output.error_message = error_message
                    has_finish_reason = True

                delta = choice.get("delta")
                if delta:
                    content = delta.get("content")
                    if content is not None and len(content) > 0:
                        block = ensure_text_block()
                        block.text += content
                        out_stream.push(
                            TextDeltaEvent(content_index=content_index(block), delta=content, partial=output)
                        )

                    # Reasoning may arrive in reasoning_content (llama.cpp),
                    # reasoning, or reasoning_text; use the first non-empty
                    # field to avoid duplication.
                    found_reasoning_field = None
                    for field_name in ("reasoning_content", "reasoning", "reasoning_text"):
                        value = delta.get(field_name)
                        if isinstance(value, str) and value:
                            found_reasoning_field = field_name
                            break

                    if found_reasoning_field:
                        reasoning_delta = delta[found_reasoning_field]
                        thinking_signature = (
                            "reasoning_content"
                            if model.provider == "opencode-go" and found_reasoning_field == "reasoning"
                            else found_reasoning_field
                        )
                        block = ensure_thinking_block(thinking_signature)
                        block.thinking += reasoning_delta
                        out_stream.push(
                            ThinkingDeltaEvent(
                                content_index=content_index(block), delta=reasoning_delta, partial=output
                            )
                        )

                    if delta.get("tool_calls"):
                        for tool_call in delta["tool_calls"]:
                            block = ensure_tool_call_block(tool_call)
                            if not block.id and tool_call.get("id"):
                                block.id = tool_call["id"]
                                tool_blocks_by_id[tool_call["id"]] = block
                            name = (tool_call.get("function") or {}).get("name") or (tool_call.get("custom") or {}).get(
                                "name"
                            )
                            if not block.name and name:
                                block.name = name

                            call_delta = ""
                            function_arguments = (tool_call.get("function") or {}).get("arguments")
                            custom_input = (tool_call.get("custom") or {}).get("input")
                            entry = tool_scratch(block)
                            if function_arguments:
                                call_delta = function_arguments
                                entry.partial_args = (entry.partial_args or "") + function_arguments
                                block.arguments = parse_streaming_json(entry.partial_args)
                            elif custom_input:
                                next_input = get_custom_tool_call_input(block) + custom_input
                                call_delta = append_custom_tool_call_input(block, next_input, False) or ""
                            out_stream.push(
                                ToolCallDeltaEvent(content_index=content_index(block), delta=call_delta, partial=output)
                            )

                    reasoning_details = delta.get("reasoning_details")
                    if isinstance(reasoning_details, list):
                        for detail in reasoning_details:
                            if _is_encrypted_reasoning_detail(detail):
                                serialized_detail = json.dumps(detail)
                                matching = tool_blocks_by_id.get(detail["id"])
                                if matching is not None:
                                    matching.thought_signature = serialized_detail
                                else:
                                    pending_reasoning_details[detail["id"]] = serialized_detail

            for block in list(blocks):
                finish_block(block)
            if opts.cancel is not None and opts.cancel.cancelled:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "aborted":
                raise RuntimeError("Request was aborted")
            if not has_finish_reason and not compat.supports_finish_reason:
                output.stop_reason = "toolUse" if any(block.type == "toolCall" for block in output.content) else "stop"
            if output.stop_reason == "error":
                raise RuntimeError(output.error_message or "Provider returned an error stop reason")
            if (compat.supports_finish_reason and not has_finish_reason) or output.stop_reason == "pending":
                raise RuntimeError("Stream ended without finish_reason")

            out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            out_stream.end()
        except Exception as error:
            output.stop_reason = "aborted" if opts.cancel is not None and opts.cancel.cancelled else "error"
            output.error_message = format_provider_error(normalize_provider_error(error))
            # OpenRouter attaches extra info in error.metadata.raw; append it
            # only when not already present to avoid double-printing.
            error_field = getattr(error, "error", None)
            raw_metadata = error_field.get("metadata", {}).get("raw") if isinstance(error_field, dict) else None
            if raw_metadata and str(raw_metadata) not in output.error_message:
                output.error_message += f"\n{raw_metadata}"
            out_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
            out_stream.end()

    out_stream.spawn_producer(_run(), opts.cancel)
    return out_stream


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    _get_client_api_key(model.provider, options.api_key if options else None, options.headers if options else None)

    base = build_base_options(model, context, options, options.api_key if options else None)
    clamped_reasoning = clamp_thinking_level(model, options.reasoning) if options and options.reasoning else None
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning
    tool_choice = getattr(options, "tool_choice", None)

    opts = _openai_options(base)
    opts.reasoning_effort = reasoning_effort
    opts.tool_choice = tool_choice
    opts.thinking_budgets = options.thinking_budgets if options else None
    return stream(model, context, opts, into=into)
