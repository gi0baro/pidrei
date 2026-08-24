"""Port of pi's openai-responses adapter (packages/ai/src/api/openai-responses.ts)."""

import json
import time
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass, fields
from typing import Any, Protocol

from pidrei_ai.api.constrained_sampling import create_grammar_tool_input_properties
from pidrei_ai.api.github_copilot_headers import build_copilot_dynamic_headers, has_copilot_vision_input
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
    CacheRetention,
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
from pidrei_ai.utils import http
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.deferred_tools import split_deferred_tools
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_ai.utils.sse import iterate_sse_messages
from pidrei_ai.utils.user_agent import force_user_agent


OPENAI_TOOL_CALL_PROVIDERS = frozenset(("openai", "openai-codex", "opencode"))
# OpenAI Responses rejects max_output_tokens below 16.
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16


@dataclass(slots=True)
class OpenAIResponsesOptions(StreamOptions):
    reasoning_effort: str | None = None  # "minimal" | "low" | "medium" | "high" | "xhigh" | "max"
    reasoning_summary: str | None = None  # "auto" | "detailed" | "concise"
    service_tier: str | None = None
    tool_choice: Any = None
    # Pre-built client instance (test injection / alternative transports).
    client: OpenAIResponsesClient | None = None


class OpenAIResponseLike(Protocol):
    status: int
    headers: dict[str, str]

    def aiter_bytes(self) -> AsyncIterable[bytes]: ...


class OpenAIResponsesClient(Protocol):
    async def create(
        self,
        params: dict[str, Any],
        *,
        timeout_ms: float | None,
        cancel: CancelToken | None,
    ) -> OpenAIResponseLike: ...


class OpenAIApiError(Exception):
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


class _PunkreqResponsesClient:
    """Default transport: POST {base_url}/responses through the punkreq seam."""

    def __init__(self, base_url: str, headers: dict[str, str], env: ProviderEnv | None = None):
        self._url = f"{base_url.rstrip('/')}/responses"
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


async def _iterate_events(response: OpenAIResponseLike, cancel: CancelToken | None) -> AsyncGenerator[dict]:
    body = response.aiter_bytes()
    ended = False
    try:
        async for sse in iterate_sse_messages(body):
            if sse.data == "[DONE]":
                ended = True
                return
            event = json.loads(sse.data)
            if isinstance(event, dict):
                yield event
        ended = True
    finally:
        await http.finish_body(body, response, drain=ended)


# --- compat / options ---------------------------------------------------------


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


def _detect_session_affinity_format(model: Model) -> str:
    return "openrouter" if model.provider == "openrouter" or "openrouter.ai" in model.base_url else "openai"


def _resolve_cache_retention(cache_retention: CacheRetention | None, env: ProviderEnv | None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PIDREI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


@dataclass(slots=True)
class _ResolvedCompat:
    supports_developer_role: bool
    session_affinity_format: str
    supports_long_cache_retention: bool
    supports_strict_mode: bool
    supports_openai_grammar_tools: bool
    supports_additional_tools: bool
    supports_tool_search: bool
    supports_explicit_prompt_cache_mode: bool


def get_compat(model: Model) -> _ResolvedCompat:
    compat = model.compat if isinstance(model.compat, OpenAIResponsesCompat) else None

    def pick(value, default):
        return value if value is not None else default

    return _ResolvedCompat(
        supports_developer_role=pick(compat.supports_developer_role if compat else None, True),
        session_affinity_format=pick(
            compat.session_affinity_format if compat else None, _detect_session_affinity_format(model)
        ),
        supports_long_cache_retention=pick(compat.supports_long_cache_retention if compat else None, True),
        supports_strict_mode=pick(compat.supports_strict_mode if compat else None, False),
        supports_openai_grammar_tools=pick(compat.supports_openai_grammar_tools if compat else None, False),
        supports_additional_tools=pick(compat.supports_additional_tools if compat else None, False),
        supports_tool_search=pick(compat.supports_tool_search if compat else None, False),
        supports_explicit_prompt_cache_mode=pick(compat.supports_explicit_prompt_cache_mode if compat else None, False),
    )


def _format_openai_responses_error(error: Any) -> str:
    return format_provider_error(normalize_provider_error(error), "OpenAI API error")


def _responses_options(options: StreamOptions | None) -> OpenAIResponsesOptions:
    if isinstance(options, OpenAIResponsesOptions):
        return options
    if options is None:
        return OpenAIResponsesOptions()
    values = {field.name: getattr(options, field.name) for field in fields(StreamOptions)}
    return OpenAIResponsesOptions(**values)


# --- client / params ----------------------------------------------------------


def _create_client(
    model: Model,
    context: Context,
    api_key: str,
    options_headers: ProviderHeaders | None,
    session_id: str | None,
    env: ProviderEnv | None = None,
) -> OpenAIResponsesClient:
    compat = get_compat(model)
    headers: dict[str, Any] = dict(model.headers or {})
    if model.provider == "github-copilot":
        headers.update(build_copilot_dynamic_headers(context.messages, has_copilot_vision_input(context.messages)))

    if session_id:
        if compat.session_affinity_format == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.session_affinity_format == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id

    if options_headers:
        headers.update(options_headers)

    if model.provider == "xai":
        force_user_agent(headers)

    headers["authorization"] = f"Bearer {api_key}"
    final_headers = {key: value for key, value in headers.items() if value is not None}
    return _PunkreqResponsesClient(model.base_url, final_headers, env)


def build_params(
    model: Model,
    context: Context,
    options: OpenAIResponsesOptions,
    compat: _ResolvedCompat | None = None,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict:
    compat = compat if compat is not None else get_compat(model)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, compat.supports_openai_grammar_tools
        )

    deferred_tools_mode = (
        "additional-tools"
        if compat.supports_additional_tools
        else ("tool-search" if compat.supports_tool_search else None)
    )
    immediate_tools, deferred_map = split_deferred_tools(context, deferred_tools_mode is not None)
    messages = convert_responses_messages(
        model,
        context,
        set(OPENAI_TOOL_CALL_PROVIDERS),
        grammar_tool_input_properties=grammar_tool_input_properties,
        deferred_tools=deferred_map,
        deferred_tools_mode=deferred_tools_mode,
        tool_options={
            "supports_strict_mode": compat.supports_strict_mode,
            "supports_openai_grammar_tools": compat.supports_openai_grammar_tools,
        },
    )

    cache_retention = _resolve_cache_retention(options.cache_retention, options.env)
    disable_implicit_prompt_cache = cache_retention == "none" and compat.supports_explicit_prompt_cache_mode
    params: dict[str, Any] = {"model": model.id, "input": messages, "stream": True, "store": False}
    if cache_retention != "none":
        cache_key = clamp_openai_prompt_cache_key(options.session_id)
        if cache_key is not None:
            params["prompt_cache_key"] = cache_key
    if cache_retention == "long" and compat.supports_long_cache_retention:
        params["prompt_cache_retention"] = "24h"
    if disable_implicit_prompt_cache:
        params["prompt_cache_options"] = {"mode": "explicit"}

    if options.max_tokens:
        params["max_output_tokens"] = max(options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options.temperature is not None:
        params["temperature"] = options.temperature

    if options.service_tier is not None:
        params["service_tier"] = options.service_tier

    if immediate_tools:
        params["tools"] = convert_responses_tools(
            immediate_tools,
            supports_strict_mode=compat.supports_strict_mode,
            supports_openai_grammar_tools=compat.supports_openai_grammar_tools,
        )

    if options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    if model.reasoning:
        mapping = dict(model.thinking_level_map) if model.thinking_level_map is not None else {}
        if options.reasoning_effort or options.reasoning_summary:
            if options.reasoning_effort:
                mapped = mapping.get(options.reasoning_effort)
                effort = mapped if mapped is not None else options.reasoning_effort
            else:
                effort = "medium"
            params["reasoning"] = {"effort": effort, "summary": options.reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        elif model.provider != "github-copilot" and not ("off" in mapping and mapping["off"] is None):
            off_value = mapping.get("off")
            params["reasoning"] = {"effort": off_value if off_value is not None else "none"}
        if model.provider == "xai":
            params["include"] = ["reasoning.encrypted_content"]

    # Last so custom keys override the named request fields.
    if options.sampling_params:
        params.update(options.sampling_params)

    return params


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


# --- streaming ----------------------------------------------------------------


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    opts = _responses_options(options)
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

    async def _run() -> None:
        try:
            api_key = None
            if opts.client is None:
                api_key = _get_client_api_key(model.provider, opts.api_key, opts.headers)
            cache_retention = _resolve_cache_retention(opts.cache_retention, opts.env)
            cache_session_id = None if cache_retention == "none" else opts.session_id
            compat = get_compat(model)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                context.tools, compat.supports_openai_grammar_tools
            )
            client = (
                opts.client
                if opts.client is not None
                else _create_client(model, context, api_key, opts.headers, cache_session_id, opts.env)
            )
            params = build_params(model, context, opts, compat, grammar_tool_input_properties)
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

            await process_responses_stream(
                _iterate_events(response, opts.cancel),
                output,
                out_stream,
                model,
                service_tier=opts.service_tier,
                grammar_tool_input_properties=grammar_tool_input_properties,
                apply_service_tier_pricing=lambda usage, tier: _apply_service_tier_pricing(usage, tier, model),
            )

            if opts.cancel is not None and opts.cancel.cancelled:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("OpenAI Responses stream ended without a stop reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            out_stream.end()
        except Exception as error:
            output.stop_reason = "aborted" if opts.cancel is not None and opts.cancel.cancelled else "error"
            output.error_message = _format_openai_responses_error(error)
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

    opts = _responses_options(base)
    opts.tool_choice = options.tool_choice if options else None
    opts.reasoning_effort = reasoning_effort
    return stream(model, context, opts, into=into)
