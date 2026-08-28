"""Port of pi's Azure OpenAI Responses adapter (packages/ai/src/api/azure-openai-responses.ts).

pi builds an `AzureOpenAI` client from the `openai` package and calls
`client.responses.create(params)`. Here that client is the small `AzureOpenAI`
class below, which POSTs to `{baseURL}/responses` over the punkreq seam — the
same transport `api/openai_responses.py` already uses, reusing its response
wrapper, its error shape and its SSE iteration. Azure's own differences are the
`api-key` header (rather than a bearer token), the `api-version` query
parameter, and the base-URL normalization pi's spec pins.

The config keys stay the SDK's own camelCase (`apiKey`, `apiVersion`,
`baseURL`, `defaultHeaders`) because that record is what pi's
azure-openai-base-url spec asserts on.
"""

import time
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlencode, urlparse, urlunparse

from pidrei_ai.api.constrained_sampling import create_grammar_tool_input_properties
from pidrei_ai.api.openai_prompt_cache import clamp_openai_prompt_cache_key
from pidrei_ai.api.openai_responses import (
    OpenAIApiError,
    _extract_error_message,
    _iterate_events,
    _parse_error_body,
    _PunkreqResponse,
)
from pidrei_ai.api.openai_responses_shared import (
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from pidrei_ai.api.simple_options import build_base_options
from pidrei_ai.builders import AssistantMessageBuilder, UsageBuilder
from pidrei_ai.registry import clamp_thinking_level
from pidrei_ai.types import (
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StreamOptions,
)
from pidrei_ai.utils import http
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_ai.utils.user_agent import set_default_user_agent


DEFAULT_AZURE_API_VERSION = "v1"
AZURE_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode", "azure-openai-responses"})
# OpenAI Responses rejects max_output_tokens below 16.
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16

_AZURE_HOST_SUFFIXES = (".openai.azure.com", ".cognitiveservices.azure.com", ".ai.azure.com")
_AZURE_ROOT_PATHS = ("", "/", "/openai", "/openai/v1/responses")


@dataclass(slots=True)
class AzureOpenAIResponsesOptions(StreamOptions):
    reasoning_effort: str | None = None
    tool_choice: Any = None
    reasoning_summary: str | None = None
    azure_api_version: str | None = None
    azure_resource_name: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None


class AzureOpenAI:
    """`new AzureOpenAI(config)` for the one call pi makes on it."""

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.responses = _Responses(self)

    def _url(self) -> str:
        base_url = str(self.config.get("baseURL") or "").rstrip("/")
        api_version = self.config.get("apiVersion")
        url = f"{base_url}/responses"
        return f"{url}?{urlencode({'api-version': api_version})}" if api_version else url

    def _headers(self) -> dict[str, str]:
        headers = {key: value for key, value in (self.config.get("defaultHeaders") or {}).items() if value is not None}
        # Azure authenticates with `api-key`, not an Authorization bearer token.
        headers.setdefault("api-key", self.config.get("apiKey") or "")
        return headers


class _Responses:
    def __init__(self, client: AzureOpenAI):
        self._client = client

    async def create(
        self,
        params: dict[str, Any],
        *,
        timeout_ms: float | None = None,
        cancel: CancelToken | None = None,
    ) -> _PunkreqResponse:
        url = self._client._url()
        client = http.client_for(url, self._client.config.get("env"))
        response = await client.post(
            url,
            json=params,
            headers=self._client._headers(),
            timeout=http.request_timeout(timeout_ms),
        )
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


def _parse_deployment_name_map(value: str | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if not value:
        return result
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        parts = trimmed.split("=", 1)
        if len(parts) != 2:
            continue
        model_id, deployment_name = parts
        if not model_id or not deployment_name:
            continue
        result[model_id.strip()] = deployment_name.strip()
    return result


def resolve_deployment_name(model: Model, options: AzureOpenAIResponsesOptions | None = None) -> str:
    if options is not None and options.azure_deployment_name:
        return options.azure_deployment_name
    mapped = _parse_deployment_name_map(
        get_provider_env_value("AZURE_OPENAI_DEPLOYMENT_NAME_MAP", options.env if options else None)
    ).get(model.id)
    return mapped or model.id


def _format_azure_openai_error(error: Any) -> str:
    return format_provider_error(normalize_provider_error(error), "Azure OpenAI API error")


def _azure_options(options: StreamOptions | None) -> AzureOpenAIResponsesOptions:
    if isinstance(options, AzureOpenAIResponsesOptions):
        return options
    if options is None:
        return AzureOpenAIResponsesOptions()
    values = {f.name: getattr(options, f.name) for f in fields(StreamOptions)}
    return AzureOpenAIResponsesOptions(**values)


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    opts = _azure_options(options)
    out_stream = into if into is not None else AssistantMessageEventStream()

    output = AssistantMessageBuilder(
        content=[],
        api="azure-openai-responses",
        provider=model.provider,
        model=model.id,
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=int(time.time() * 1000),
    )
    out_stream.partial = output

    async def _run() -> None:
        deployment_name = resolve_deployment_name(model, opts)

        try:
            api_key = opts.api_key
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")
            client = create_client(model, api_key, opts)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                context.tools, bool(getattr(model.compat, "supports_openai_grammar_tools", None))
            )
            params = build_params(model, context, opts, deployment_name, grammar_tool_input_properties)
            next_params = await maybe_call(opts.on_payload, params, model)
            if next_params is not None:
                params = next_params

            async def _request():
                return await client.responses.create(params, timeout_ms=opts.timeout_ms, cancel=opts.cancel)

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
                grammar_tool_input_properties=grammar_tool_input_properties,
            )

            if opts.cancel is not None and opts.cancel.cancelled:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Azure OpenAI Responses stream ended without a stop reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            out_stream.end()
        except Exception as error:
            output.stop_reason = "aborted" if opts.cancel is not None and opts.cancel.cancelled else "error"
            output.error_message = _format_azure_openai_error(error)
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
    api_key = options.api_key if options else None
    if not api_key:
        raise RuntimeError(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    clamped_reasoning = (
        clamp_thinking_level(model, options.reasoning) if options is not None and options.reasoning else None
    )
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    opts = _azure_options(base)
    opts.tool_choice = options.tool_choice if options else None
    opts.reasoning_effort = reasoning_effort
    return stream(model, context, opts, into=into)


def normalize_azure_base_url(base_url: str) -> str:
    trimmed = base_url.strip().rstrip("/")
    parsed = urlparse(trimmed)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid Azure OpenAI base URL: {base_url}")

    hostname = (parsed.hostname or "").lower()
    is_azure_host = hostname.endswith(_AZURE_HOST_SUFFIXES)
    normalized_path = parsed.path.rstrip("/")

    # Azure hosts need /openai/v1 as the base path so `/deployments/<model>/...`
    # and `?api-version=v1` append correctly.
    if is_azure_host and normalized_path in _AZURE_ROOT_PATHS:
        parsed = parsed._replace(path="/openai/v1", query="")

    return urlunparse(parsed).rstrip("/")


def build_default_base_url(resource_name: str) -> str:
    return f"https://{resource_name}.openai.azure.com/openai/v1"


def resolve_azure_config(model: Model, options: AzureOpenAIResponsesOptions | None = None) -> tuple[str, str]:
    """Returns `(base_url, api_version)`."""
    env = options.env if options else None
    api_version = (
        (options.azure_api_version if options else None)
        or get_provider_env_value("AZURE_OPENAI_API_VERSION", env)
        or DEFAULT_AZURE_API_VERSION
    )

    base_url = (options.azure_base_url.strip() if options and options.azure_base_url else None) or (
        (get_provider_env_value("AZURE_OPENAI_BASE_URL", env) or "").strip() or None
    )
    resource_name = (options.azure_resource_name if options else None) or get_provider_env_value(
        "AZURE_OPENAI_RESOURCE_NAME", env
    )

    resolved_base_url = base_url
    if not resolved_base_url and resource_name:
        resolved_base_url = build_default_base_url(resource_name)
    if not resolved_base_url and model.base_url:
        resolved_base_url = model.base_url
    if not resolved_base_url:
        raise RuntimeError(
            "Azure OpenAI base URL is required. Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, "
            "or pass azure_base_url, azure_resource_name, or model.base_url."
        )

    return normalize_azure_base_url(resolved_base_url), api_version


def create_client(model: Model, api_key: str, options: AzureOpenAIResponsesOptions | None = None) -> AzureOpenAI:
    headers = dict(model.headers or {})
    if options is not None and options.headers:
        headers.update(options.headers)
    set_default_user_agent(headers)

    base_url, api_version = resolve_azure_config(model, options)

    return AzureOpenAI(
        {
            "apiKey": api_key,
            "apiVersion": api_version,
            "dangerouslyAllowBrowser": True,
            "defaultHeaders": headers,
            "baseURL": base_url,
            "env": options.env if options else None,
        }
    )


def build_params(
    model: Model,
    context: Context,
    options: AzureOpenAIResponsesOptions | None,
    deployment_name: str,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict[str, Any]:
    compat = model.compat
    supports_grammar_tools = bool(getattr(compat, "supports_openai_grammar_tools", None))
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(context.tools, supports_grammar_tools)

    messages = convert_responses_messages(
        model,
        context,
        AZURE_TOOL_CALL_PROVIDERS,
        grammar_tool_input_properties=grammar_tool_input_properties,
    )

    params: dict[str, Any] = {
        "model": deployment_name,
        "input": messages,
        "stream": True,
        "prompt_cache_key": clamp_openai_prompt_cache_key(options.session_id if options else None),
        "store": False,
    }

    if options is not None and options.max_tokens:
        params["max_output_tokens"] = max(options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS)

    if options is not None and options.temperature is not None:
        params["temperature"] = options.temperature

    if context.tools:
        supports_strict = getattr(compat, "supports_strict_mode", None)
        params["tools"] = convert_responses_tools(
            context.tools,
            supports_strict_mode=True if supports_strict is None else supports_strict,
            supports_openai_grammar_tools=supports_grammar_tools,
        )
    if options is not None and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    if model.reasoning:
        mapping = dict(model.thinking_level_map) if model.thinking_level_map is not None else {}
        if options is not None and (options.reasoning_effort or options.reasoning_summary):
            if options.reasoning_effort:
                mapped = mapping.get(options.reasoning_effort)
                effort = mapped if mapped is not None else options.reasoning_effort
            else:
                effort = "medium"
            params["reasoning"] = {"effort": effort, "summary": options.reasoning_summary or "auto"}
            params["include"] = ["reasoning.encrypted_content"]
        # pi: `model.thinkingLevelMap?.off !== null` — an explicit null opts the
        # model out entirely, while an absent key falls through to "none".
        elif not ("off" in mapping and mapping["off"] is None):
            off_value = mapping.get("off")
            params["reasoning"] = {"effort": off_value if off_value is not None else "none"}

    # Last so custom keys override the named request fields.
    if options.sampling_params:
        params.update(options.sampling_params)

    return params
