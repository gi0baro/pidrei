"""Port of pi's Amazon Bedrock adapter (packages/ai/src/api/bedrock-converse-stream.ts).

pi's `BedrockRuntimeClient` comes from `@aws-sdk/client-bedrock-runtime`; here it
comes from `api/bedrock_runtime.py`, which keeps the SDK's client/command/
middleware shape but signs with botocore and sends over punkreq. Everything
below — message conversion, endpoint and region resolution, the thinking payload
tables, the custom-header middleware — mirrors pi.

Two runtime-forced differences:

- pi's `typeof process !== "undefined"` browser fallback is not ported (POSIX
  only, as everywhere else); the Node branch is the only branch.
- pi hands proxy support to the SDK by swapping in a `NodeHttpHandler` with
  proxy agents. punkreq already resolves proxies per request through
  `http.client_for(url, env)`, so `config.requestHandler` has no analogue and
  `options.env` is threaded to the client instead.
"""

import base64
import re
import time
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlparse

from pidrei_ai.api.bedrock_runtime import (
    CACHE_POINT_TYPE_DEFAULT,
    CACHE_TTL_ONE_HOUR,
    CONVERSATION_ROLE_ASSISTANT,
    CONVERSATION_ROLE_USER,
    IMAGE_FORMAT_GIF,
    IMAGE_FORMAT_JPEG,
    IMAGE_FORMAT_PNG,
    IMAGE_FORMAT_WEBP,
    STOP_REASON_END_TURN,
    STOP_REASON_MAX_TOKENS,
    STOP_REASON_MODEL_CONTEXT_WINDOW_EXCEEDED,
    STOP_REASON_STOP_SEQUENCE,
    STOP_REASON_TOOL_USE,
    TOOL_RESULT_STATUS_ERROR,
    TOOL_RESULT_STATUS_SUCCESS,
    BedrockRuntimeClient,
    BedrockRuntimeServiceException,
    ConverseStreamCommand,
)
from pidrei_ai.api.constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from pidrei_ai.api.simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
    clamp_reasoning,
)
from pidrei_ai.api.transform_messages import transform_messages
from pidrei_ai.registry import calculate_cost
from pidrei_ai.types import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    CacheRetention,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderEnv,
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
    ThinkingLevel,
    ThinkingStartEvent,
    Tool,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
)
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.diagnostics import append_assistant_message_diagnostic
from pidrei_ai.utils.error_body import normalize_provider_error
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.headers import provider_headers_to_record
from pidrei_ai.utils.json_parse import parse_streaming_json
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates


EMPTY_TEXT_PLACEHOLDER = "<empty>"

_ARN_REGION = re.compile(r"^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):")
_STANDARD_ENDPOINT = re.compile(r"^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$")
_TOOL_CALL_ID_DISALLOWED = re.compile(r"[^a-zA-Z0-9_-]")
_MATCH_SEPARATORS = re.compile(r"[\s_.:]+")


@dataclass(slots=True)
class BedrockToolChoiceTool:
    name: str
    type: str = "tool"


@dataclass(slots=True)
class BedrockOptions(StreamOptions):
    region: str | None = None
    profile: str | None = None
    # "auto" | "any" | "none" | BedrockToolChoiceTool
    tool_choice: Any = None
    # See https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html
    reasoning: ThinkingLevel | None = None
    # Custom token budgets per thinking level. Overrides default budgets.
    thinking_budgets: ThinkingBudgets | None = None
    # Only supported by Claude 4.x models.
    interleaved_thinking: bool | None = None
    # "summarized" (default here) | "omitted". Only applies to Claude models on Bedrock.
    thinking_display: str | None = None
    # Cost-allocation tags attached to the inference request.
    request_metadata: dict[str, str] | None = None
    # Bearer token for Bedrock API key auth; bypasses SigV4 signing.
    bearer_token: str | None = None


def _bedrock_options(options: StreamOptions | None) -> BedrockOptions:
    if isinstance(options, BedrockOptions):
        return options
    if options is None:
        return BedrockOptions()
    values = {f.name: getattr(options, f.name) for f in fields(StreamOptions)}
    return BedrockOptions(**values)


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    opts = _bedrock_options(options)
    out_stream = into if into is not None else AssistantMessageEventStream()

    output = AssistantMessage(
        content=[],
        api="bedrock-converse-stream",
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=int(time.time() * 1000),
    )
    out_stream.partial = output

    async def _run() -> None:
        blocks = output.content
        # pi tags blocks with the provider's contentBlockIndex while streaming and
        # deletes it afterwards; dataclasses have no spare slot, so the mapping
        # lives beside the blocks and is simply dropped at the end.
        block_indices: dict[int, int] = {}
        partial_json: dict[int, str] = {}

        # A profile explicitly configured through pi's auth flow (the `profile`
        # option or scoped `AWS_PROFILE` on the stored credential's env) must win
        # over ambient AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY. The SDK default
        # chain already prefers a configured profile over env keys, but only when
        # `credentials` is not set on the client config. See #6957.
        options_profile = opts.profile or (opts.env or {}).get("AWS_PROFILE")
        config: dict[str, Any] = {
            "profile": options_profile or get_provider_env_value("AWS_PROFILE", opts.env),
        }
        configured_region = _get_configured_bedrock_region(opts)
        has_ambient_configured_profile = bool(get_provider_env_value("AWS_PROFILE"))
        endpoint_region = _get_standard_bedrock_endpoint_region(model.base_url)
        use_explicit_endpoint = _should_use_explicit_bedrock_endpoint(
            model.base_url, configured_region, has_ambient_configured_profile
        )

        # Only pin standard AWS Bedrock runtime endpoints when no region or ambient
        # AWS_PROFILE is configured. This preserves custom endpoints (VPC/proxy)
        # without forcing built-in catalog defaults such as us-east-1 to override
        # AWS_REGION/AWS_PROFILE.
        if use_explicit_endpoint:
            config["endpoint"] = model.base_url

        skip_auth = get_provider_env_value("AWS_BEDROCK_SKIP_AUTH", opts.env) == "1"
        bearer_token = (
            opts.bearer_token or opts.api_key or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", opts.env) or None
        )
        use_bearer_token = bearer_token is not None and not skip_auth

        # Region resolution: ARN-embedded > explicit option > env vars > SDK default chain.
        arn_region_match = _ARN_REGION.match(model.id)
        if arn_region_match:
            config["region"] = arn_region_match.group(1)
        elif configured_region:
            config["region"] = configured_region
        elif endpoint_region and use_explicit_endpoint:
            config["region"] = endpoint_region
        elif not has_ambient_configured_profile:
            config["region"] = "us-east-1"

        # Support proxies that don't need authentication
        if skip_auth:
            config["credentials"] = {
                "accessKeyId": "dummy-access-key",
                "secretAccessKey": "dummy-secret-key",
            }

        credentials = _get_configured_bedrock_credentials(opts.env)
        if not skip_auth and credentials and not options_profile:
            config["credentials"] = credentials

        if use_bearer_token:
            config["token"] = {"token": bearer_token}
            config["authSchemePreference"] = ["httpBearerAuth"]

        # Kept outside the try so the except can still correlate a mid-stream
        # failure: exceptions raised as stream events carry no HTTP metadata of
        # their own.
        response_request_id: str | None = None

        try:
            client = BedrockRuntimeClient(config)
            observed_raw_response = False

            def _mark_observed() -> None:
                nonlocal observed_raw_response
                observed_raw_response = True

            if opts.on_response:
                add_response_headers_middleware(client, opts.on_response, model, _mark_observed)
            custom_headers = provider_headers_to_record(opts.headers)
            if custom_headers:
                add_custom_headers_middleware(client, custom_headers)
            cache_retention = _resolve_cache_retention(opts.cache_retention, opts.env)
            inference_max_tokens = (
                opts.max_tokens
                if opts.max_tokens is not None
                else (model.max_tokens if _is_anthropic_claude_model(model) else None)
            )
            command_input: dict[str, Any] = {
                "modelId": model.id,
                "messages": convert_messages(context, model, cache_retention, opts.env),
                "system": build_system_prompt(context.system_prompt, model, cache_retention, opts.env),
                "inferenceConfig": {
                    **({"maxTokens": inference_max_tokens} if inference_max_tokens is not None else {}),
                    **({"temperature": opts.temperature} if opts.temperature is not None else {}),
                },
                "toolConfig": convert_tool_config(
                    context.tools,
                    opts.tool_choice,
                    bool(getattr(model.compat, "supports_strict_mode", None)),
                ),
                "additionalModelRequestFields": build_additional_model_request_fields(model, opts),
                **({"requestMetadata": opts.request_metadata} if opts.request_metadata is not None else {}),
            }
            next_command_input = await maybe_call(opts.on_payload, command_input, model)
            if next_command_input is not None:
                command_input = next_command_input
            command = ConverseStreamCommand(command_input)

            response = await client.send(command, cancel=opts.cancel)
            response_request_id = _normalize_diagnostic_value(response.metadata.request_id)
            if not observed_raw_response and response.metadata.http_status_code is not None:
                response_headers: dict[str, str] = {}
                if response.metadata.request_id:
                    response_headers["x-amzn-requestid"] = response.metadata.request_id
                await maybe_call(
                    opts.on_response,
                    ProviderResponse(status=response.metadata.http_status_code, headers=response_headers),
                    model,
                )

            async for item in response.stream:
                if "messageStart" in item:
                    if item["messageStart"].get("role") != CONVERSATION_ROLE_ASSISTANT:
                        raise RuntimeError("Unexpected assistant message start but got user message start instead")
                    out_stream.push(StartEvent(partial=output))
                elif "contentBlockStart" in item:
                    _handle_content_block_start(
                        item["contentBlockStart"], blocks, block_indices, partial_json, output, out_stream
                    )
                elif "contentBlockDelta" in item:
                    _handle_content_block_delta(
                        item["contentBlockDelta"], blocks, block_indices, partial_json, output, out_stream
                    )
                elif "contentBlockStop" in item:
                    _handle_content_block_stop(
                        item["contentBlockStop"], blocks, block_indices, partial_json, output, out_stream
                    )
                elif "messageStop" in item:
                    output.raw_stop_reason = item["messageStop"].get("stopReason")
                    stop_reason, error_message = map_stop_reason(item["messageStop"].get("stopReason"))
                    output.stop_reason = stop_reason
                    if error_message:
                        output.error_message = error_message
                elif "metadata" in item:
                    _handle_metadata(item["metadata"], model, output)

            if opts.cancel is not None and opts.cancel.cancelled:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Bedrock stream ended without a stop reason")
            if output.stop_reason in ("error", "aborted"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            out_stream.end()
        except Exception as error:
            output.stop_reason = "aborted" if opts.cancel is not None and opts.cancel.cancelled else "error"
            output.error_message = format_bedrock_error(error)
            if output.stop_reason == "error":
                _append_bedrock_failure_diagnostic(output, error, response_request_id)
            out_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
            out_stream.end()

    out_stream.spawn_producer(_run(), opts.cancel)
    return out_stream


# Human-readable prefixes for Bedrock SDK exception names. The downstream retry
# logic matches patterns like `server.?error` and `service.?unavailable`, so the
# legacy prefix format is preserved rather than the raw SDK exception name.
BEDROCK_ERROR_PREFIXES = {
    "InternalServerException": "Internal server error",
    "ModelStreamErrorException": "Model stream error",
    "ValidationException": "Validation error",
    "ThrottlingException": "Throttling error",
    "ServiceUnavailableException": "Service unavailable",
}

# Some models reject the account/profile's configured Bedrock data retention mode.
BEDROCK_DATA_RETENTION_DOCS_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html"


def format_bedrock_error(error: Any) -> str:
    """Format a Bedrock error with a human-readable prefix."""
    norm = normalize_provider_error(error)
    # Surface the raw HTTP body (with status) when the SDK did not fold it into
    # the message; otherwise fall back to the message. This is what stops a
    # gateway 403 from collapsing to `Unknown: UnknownError`.
    core = (
        f"{norm.status}: {norm.body}"
        if not norm.message_carries_body and norm.status is not None and norm.body is not None
        else norm.message
    )
    data_retention_hint = (
        f" See {BEDROCK_DATA_RETENTION_DOCS_URL} for supported data retention modes."
        if re.search(r"data retention mode", core, re.IGNORECASE)
        else ""
    )
    if isinstance(error, BedrockRuntimeServiceException):
        prefix = BEDROCK_ERROR_PREFIXES.get(error.name, error.name)
        return f"{prefix}: {core}{data_retention_hint}"
    return f"{core}{data_retention_hint}"


# Over-long header values are dropped rather than truncated: a truncated request
# id is not a request id.
MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS = 200


def _normalize_diagnostic_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if len(trimmed) == 0 or len(trimmed) > MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS:
        return None
    return trimmed


def _extract_bedrock_error_code(error: Any) -> str | None:
    """Modeled Bedrock error codes all end in `Exception`, unlike transport
    names such as `TimeoutError`; the SDK's `Unknown`/`UnknownError` fallbacks
    are excluded the same way. pi additionally sees modeled mid-stream
    exceptions as bare object literals (no code at all); pidrei's runtime raises
    them as `BedrockRuntimeServiceException`, so their code is available here —
    a deliberate, strictly-richer divergence of the hand-rolled runtime.
    """
    if not isinstance(error, Exception):
        return None
    name = getattr(error, "name", None)
    if not isinstance(name, str) or not name.endswith("Exception"):
        return None
    return _normalize_diagnostic_value(name)


def _append_bedrock_failure_diagnostic(output: AssistantMessage, error: Any, fallback_request_id: str | None) -> None:
    """Structured metadata alongside `error_message`, which stays byte-identical
    because `is_retryable_assistant_error` matches against it. Unknown fields
    are omitted, never guessed. `details` only, as the raised value's shape is
    not guaranteed.
    """
    details: dict[str, Any] = {}

    status = getattr(error, "status", None)
    if isinstance(status, int) and not isinstance(status, bool):
        details["status"] = status

    error_code = _extract_bedrock_error_code(error)
    if error_code is not None:
        details["errorCode"] = error_code

    request_id = _normalize_diagnostic_value(getattr(error, "request_id", None))
    if request_id is None:
        request_id = fallback_request_id
    if request_id is not None:
        details["requestId"] = request_id

    if not details:
        return

    append_assistant_message_diagnostic(
        output,
        AssistantMessageDiagnostic(type="bedrock_response_failure", timestamp=int(time.time() * 1000), details=details),
    )


# Header keys that must never be overwritten by caller-supplied headers.
# `host` and `x-amz-*` participate in the SigV4 canonical request; `authorization`
# is owned by SigV4 or the bearer-token path.
RESERVED_HEADER_EXACT = frozenset({"authorization", "host"})


def is_reserved_header(key: str) -> bool:
    lower = key.lower()
    return lower.startswith("x-amz-") or lower in RESERVED_HEADER_EXACT


def add_custom_headers_middleware(client: BedrockRuntimeClient, headers: dict[str, str]) -> None:
    """Attach caller headers via a `build`-step middleware.

    The build step runs after request serialisation but before SigV4 signing, so
    injected headers are covered by the signature. Reserved SigV4/auth headers
    are silently skipped; all others override any existing same-named header.
    """

    def middleware(next_handler):
        async def handle(args):
            request = getattr(args, "request", None)
            request_headers = getattr(request, "headers", None)
            if isinstance(request_headers, dict):
                for key, value in headers.items():
                    if not is_reserved_header(key):
                        request_headers[key] = value
            return await next_handler(args)

        return handle

    client.middleware_stack.add(middleware, step="build", name="pidrei-ai-custom-headers", priority="low")


def _is_smithy_http_response(response: Any) -> bool:
    return isinstance(getattr(response, "status_code", None), int) and getattr(response, "headers", None) is not None


def _to_provider_response(response: Any) -> ProviderResponse | None:
    if not _is_smithy_http_response(response):
        return None
    return ProviderResponse(status=response.status_code, headers=dict(response.headers))


def add_response_headers_middleware(client, on_response, model: Model, on_observed) -> None:
    """Report the raw response headers via a `deserialize`-step middleware.

    Bedrock's modelled `metadata` only preserves selected HTTP metadata (for
    example `request_id`), so custom gateway headers are otherwise lost before
    callers see `on_response`. Capture the raw response at the deserialize step,
    after it arrives but before the event stream is consumed.
    """

    def middleware(next_handler):
        async def handle(args):
            result = await next_handler(args)
            provider_response = _to_provider_response(getattr(result, "response", None))
            if provider_response is not None:
                on_observed()
                await maybe_call(on_response, provider_response, model)
            return result

        return handle

    client.middleware_stack.add(middleware, step="deserialize", name="pidrei-ai-response-headers")


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    base = build_base_options(model, context, options, None)
    if options is None or not options.reasoning:
        opts = _bedrock_options(base)
        opts.reasoning = None
        return stream(model, context, opts, into=into)

    if _is_anthropic_claude_model(model):
        if _supports_adaptive_thinking(model.id, model.name):
            opts = _bedrock_options(base)
            opts.reasoning = options.reasoning
            opts.thinking_budgets = options.thinking_budgets
            return stream(model, context, opts, into=into)

        # None means the caller did not request an output cap; let the helper use
        # the model cap. Do not coerce to 0, or the thinking budget would become
        # the entire max_tokens value.
        adjusted_max_tokens, adjusted_budget = adjust_max_tokens_for_thinking(
            base.max_tokens, model.max_tokens, options.reasoning, options.thinking_budgets
        )
        max_tokens = clamp_max_tokens_to_context(model, context, adjusted_max_tokens)

        level = clamp_reasoning(options.reasoning)
        budgets = _copy_thinking_budgets(options.thinking_budgets)
        setattr(budgets, level, min(adjusted_budget, max(0, max_tokens - 1024)))

        opts = _bedrock_options(base)
        opts.max_tokens = max_tokens
        opts.reasoning = options.reasoning
        opts.thinking_budgets = budgets
        return stream(model, context, opts, into=into)

    opts = _bedrock_options(base)
    opts.reasoning = options.reasoning
    opts.thinking_budgets = options.thinking_budgets
    return stream(model, context, opts, into=into)


def _copy_thinking_budgets(budgets: ThinkingBudgets | None) -> ThinkingBudgets:
    if budgets is None:
        return ThinkingBudgets()
    return ThinkingBudgets(minimal=budgets.minimal, low=budgets.low, medium=budgets.medium, high=budgets.high)


def _handle_content_block_start(
    event: dict, blocks: list, block_indices: dict, partial_json: dict, output: AssistantMessage, out_stream
) -> None:
    index = event.get("contentBlockIndex")
    start = event.get("start") or {}

    if start.get("toolUse"):
        block = ToolCall(
            id=start["toolUse"].get("toolUseId") or "", name=start["toolUse"].get("name") or "", arguments={}
        )
        output.content.append(block)
        position = len(blocks) - 1
        block_indices[index] = position
        partial_json[position] = ""
        out_stream.push(ToolCallStartEvent(content_index=position, partial=output))


def _handle_content_block_delta(
    event: dict, blocks: list, block_indices: dict, partial_json: dict, output: AssistantMessage, out_stream
) -> None:
    content_block_index = event.get("contentBlockIndex")
    delta = event.get("delta") or {}
    index = block_indices.get(content_block_index, -1)
    block = blocks[index] if index >= 0 else None

    if delta.get("text") is not None:
        # No contentBlockStart arrives for text blocks, so create one on first delta.
        if block is None:
            block = TextContent(text="")
            output.content.append(block)
            index = len(blocks) - 1
            block_indices[content_block_index] = index
            out_stream.push(TextStartEvent(content_index=index, partial=output))
        if block.type == "text":
            block.text += delta["text"]
            out_stream.push(TextDeltaEvent(content_index=index, delta=delta["text"], partial=output))
    elif delta.get("toolUse") and block is not None and block.type == "toolCall":
        partial_json[index] = partial_json.get(index, "") + (delta["toolUse"].get("input") or "")
        block.arguments = parse_streaming_json(partial_json[index])
        out_stream.push(
            ToolCallDeltaEvent(content_index=index, delta=delta["toolUse"].get("input") or "", partial=output)
        )
    elif delta.get("reasoningContent"):
        thinking_block = block
        thinking_index = index

        if thinking_block is None:
            thinking_block = ThinkingContent(thinking="", thinking_signature="")
            output.content.append(thinking_block)
            thinking_index = len(blocks) - 1
            block_indices[content_block_index] = thinking_index
            out_stream.push(ThinkingStartEvent(content_index=thinking_index, partial=output))

        if thinking_block.type == "thinking":
            reasoning = delta["reasoningContent"]
            if reasoning.get("text"):
                thinking_block.thinking += reasoning["text"]
                out_stream.push(
                    ThinkingDeltaEvent(content_index=thinking_index, delta=reasoning["text"], partial=output)
                )
            if reasoning.get("signature"):
                thinking_block.thinking_signature = (thinking_block.thinking_signature or "") + reasoning["signature"]


def _handle_metadata(event: dict, model: Model, output: AssistantMessage) -> None:
    usage = event.get("usage")
    if usage:
        output.usage.input = usage.get("inputTokens") or 0
        output.usage.output = usage.get("outputTokens") or 0
        output.usage.cache_read = usage.get("cacheReadInputTokens") or 0
        output.usage.cache_write = usage.get("cacheWriteInputTokens") or 0
        output.usage.total_tokens = usage.get("totalTokens") or (output.usage.input + output.usage.output)
        calculate_cost(model, output.usage)


def _handle_content_block_stop(
    event: dict, blocks: list, block_indices: dict, partial_json: dict, output: AssistantMessage, out_stream
) -> None:
    index = block_indices.pop(event.get("contentBlockIndex"), -1)
    if index < 0:
        return
    block = blocks[index]

    if block.type == "text":
        out_stream.push(TextEndEvent(content_index=index, content=block.text, partial=output))
    elif block.type == "thinking":
        out_stream.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=output))
    elif block.type == "toolCall":
        block.arguments = parse_streaming_json(partial_json.pop(index, ""))
        out_stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=output))


def _get_model_match_candidates(model_id: str, model_name: str | None = None) -> list[str]:
    """Both id and name, so application inference profiles (whose ARNs omit the
    model name) still match."""
    values = [model_id, model_name] if model_name else [model_id]
    candidates: list[str] = []
    for value in values:
        lower = value.lower()
        candidates.extend((lower, _MATCH_SEPARATORS.sub("-", lower)))
    return candidates


def _supports_adaptive_thinking(model_id: str, model_name: str | None = None) -> bool:
    candidates = _get_model_match_candidates(model_id, model_name)
    return any(
        marker in s
        for s in candidates
        for marker in ("opus-4-6", "opus-4-7", "opus-4-8", "opus-5", "sonnet-4-6", "sonnet-5", "fable-5")
    )


def _supports_native_xhigh_effort(model: Model) -> bool:
    candidates = _get_model_match_candidates(model.id, model.name)
    return any(marker in s for s in candidates for marker in ("opus-4-7", "opus-4-8", "opus-5", "sonnet-5", "fable-5"))


def _map_thinking_level_to_effort(model: Model, level: ThinkingLevel | None) -> str:
    if level == "xhigh" and _supports_native_xhigh_effort(model):
        return "xhigh"

    mapped = getattr(model.thinking_level_map, level, None) if (model.thinking_level_map and level) else None
    if isinstance(mapped, str):
        return mapped

    match level:
        case "minimal" | "low":
            return "low"
        case "medium":
            return "medium"
        case "high":
            return "high"
        case _:
            return "high"


def _resolve_cache_retention(cache_retention: CacheRetention | None, env: ProviderEnv | None = None) -> CacheRetention:
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PIDREI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def _is_anthropic_claude_model(model: Model) -> bool:
    id = model.id.lower()
    name = (model.name or "").lower()
    return (
        "anthropic.claude" in id
        or "anthropic/claude" in id
        or "anthropic.claude" in name
        or "anthropic/claude" in name
        or "claude" in name
    )


def _supports_prompt_caching(model: Model, env: ProviderEnv | None = None) -> bool:
    candidates = _get_model_match_candidates(model.id, model.name)

    has_claude_ref = any("claude" in s for s in candidates)
    if not has_claude_ref:
        # Application inference profiles don't carry the model name in the ARN.
        return get_provider_env_value("AWS_BEDROCK_FORCE_CACHE", env) == "1"
    if any("fable-5" in s or "opus-5" in s or "sonnet-5" in s for s in candidates):
        return True
    if any("-4-" in s for s in candidates):
        return True
    if any("claude-3-7-sonnet" in s for s in candidates):
        return True
    return any("claude-3-5-haiku" in s for s in candidates)


def _supports_thinking_signature(model: Model) -> bool:
    """Only Anthropic Claude models accept `reasoningContent…signature`; others
    reject it outright."""
    return _is_anthropic_claude_model(model)


def build_system_prompt(
    system_prompt: str | None, model: Model, cache_retention: CacheRetention, env: ProviderEnv | None = None
) -> list[dict[str, Any]] | None:
    if not system_prompt:
        return None

    blocks: list[dict[str, Any]] = [{"text": sanitize_surrogates(system_prompt)}]

    if cache_retention != "none" and _supports_prompt_caching(model, env):
        blocks.append({"cachePoint": _cache_point(cache_retention)})

    return blocks


def _cache_point(cache_retention: CacheRetention) -> dict[str, Any]:
    return {
        "type": CACHE_POINT_TYPE_DEFAULT,
        **({"ttl": CACHE_TTL_ONE_HOUR} if cache_retention == "long" else {}),
    }


def _normalize_tool_call_id(id: str, _model: Model, _source: AssistantMessage) -> str:
    return _TOOL_CALL_ID_DISALLOWED.sub("_", id)[:64]


def _create_non_blank_text_block(text: str) -> dict[str, Any] | None:
    sanitized = sanitize_surrogates(text)
    return None if sanitized.strip() == "" else {"text": sanitized}


def _create_required_text_block(text: str) -> dict[str, Any]:
    return _create_non_blank_text_block(text) or {"text": EMPTY_TEXT_PLACEHOLDER}


def _sanitize_bedrock_document(value: Any) -> Any:
    if isinstance(value, list):
        return [_sanitize_bedrock_document(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize_bedrock_document(nested) for key, nested in value.items() if len(key) > 0}
    return value


def _convert_tool_result_content(content: list) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for c in content:
        if c.type == "image":
            result.append({"image": create_image_block(c.mime_type, c.data)})
        else:
            text_block = _create_non_blank_text_block(c.text)
            if text_block:
                result.append(text_block)
    if len(result) == 0:
        result.append({"text": EMPTY_TEXT_PLACEHOLDER})
    return result


def convert_messages(
    context: Context, model: Model, cache_retention: CacheRetention, env: ProviderEnv | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    transformed_messages = transform_messages(context.messages, model, _normalize_tool_call_id)

    i = 0
    while i < len(transformed_messages):
        m = transformed_messages[i]

        if m.role == "user":
            content: list[dict[str, Any]] = []
            if isinstance(m.content, str):
                content.append(_create_required_text_block(m.content))
            else:
                for c in m.content:
                    if c.type == "text":
                        text_block = _create_non_blank_text_block(c.text)
                        if text_block:
                            content.append(text_block)
                    elif c.type == "image":
                        content.append({"image": create_image_block(c.mime_type, c.data)})
                    else:
                        continue
                if len(content) == 0:
                    content.append({"text": EMPTY_TEXT_PLACEHOLDER})
            result.append({"role": CONVERSATION_ROLE_USER, "content": content})

        elif m.role == "assistant":
            # Bedrock rejects messages with empty content arrays (e.g. from aborted requests).
            if len(m.content) == 0:
                i += 1
                continue
            content_blocks: list[dict[str, Any]] = []
            for c in m.content:
                if c.type == "text":
                    text_block = _create_non_blank_text_block(c.text)
                    if not text_block:
                        continue
                    content_blocks.append(text_block)
                elif c.type == "toolCall":
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": c.id,
                                "name": c.name,
                                "input": _sanitize_bedrock_document(c.arguments),
                            }
                        }
                    )
                elif c.type == "thinking":
                    thinking = sanitize_surrogates(c.thinking)
                    if thinking.strip() == "":
                        continue
                    if _supports_thinking_signature(model):
                        # Signatures arrive after thinking deltas. A partial or
                        # externally persisted message without one would be
                        # rejected on replay, so fall back to plain text.
                        if not c.thinking_signature or c.thinking_signature.strip() == "":
                            content_blocks.append({"text": thinking})
                        else:
                            content_blocks.append(
                                {
                                    "reasoningContent": {
                                        "reasoningText": {
                                            "text": thinking,
                                            "signature": c.thinking_signature,
                                        }
                                    }
                                }
                            )
                    else:
                        content_blocks.append({"reasoningContent": {"reasoningText": {"text": thinking}}})
                else:
                    continue
            if len(content_blocks) == 0:
                i += 1
                continue
            result.append({"role": CONVERSATION_ROLE_ASSISTANT, "content": content_blocks})

        elif m.role == "toolResult":
            # Bedrock requires all tool results to be in one user message, so
            # consecutive toolResult messages are collected together.
            tool_results: list[dict[str, Any]] = [
                {
                    "toolResult": {
                        "toolUseId": m.tool_call_id,
                        "content": _convert_tool_result_content(m.content),
                        "status": TOOL_RESULT_STATUS_ERROR if m.is_error else TOOL_RESULT_STATUS_SUCCESS,
                    }
                }
            ]

            j = i + 1
            while j < len(transformed_messages) and transformed_messages[j].role == "toolResult":
                next_msg = transformed_messages[j]
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": next_msg.tool_call_id,
                            "content": _convert_tool_result_content(next_msg.content),
                            "status": TOOL_RESULT_STATUS_ERROR if next_msg.is_error else TOOL_RESULT_STATUS_SUCCESS,
                        }
                    }
                )
                j += 1

            i = j - 1
            result.append({"role": CONVERSATION_ROLE_USER, "content": tool_results})

        i += 1

    # Cache point on the last user message for supported Claude models.
    if cache_retention != "none" and _supports_prompt_caching(model, env) and len(result) > 0:
        last_message = result[-1]
        if last_message["role"] == CONVERSATION_ROLE_USER and last_message.get("content"):
            last_message["content"].append({"cachePoint": _cache_point(cache_retention)})

    return result


def convert_tool_config(
    tools: list[Tool] | None, tool_choice: Any, supports_strict_mode: bool
) -> dict[str, Any] | None:
    if not tools:
        return None
    if tool_choice == "none":
        return None

    bedrock_tools: list[dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        bedrock_tools.append(
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": get_json_schema_tool_parameters(tool, strict)},
                    **({"strict": True} if strict is True else {}),
                }
            }
        )

    bedrock_tool_choice: dict[str, Any] | None = None
    if tool_choice == "auto":
        bedrock_tool_choice = {"auto": {}}
    elif tool_choice == "any":
        bedrock_tool_choice = {"any": {}}
    elif isinstance(tool_choice, BedrockToolChoiceTool):
        bedrock_tool_choice = {"tool": {"name": tool_choice.name}}

    return {"tools": bedrock_tools, "toolChoice": bedrock_tool_choice}


def map_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    match reason:
        case s if s in (STOP_REASON_END_TURN, STOP_REASON_STOP_SEQUENCE):
            return "stop", None
        case s if s in (STOP_REASON_MAX_TOKENS, STOP_REASON_MODEL_CONTEXT_WINDOW_EXCEEDED):
            return "length", None
        case s if s == STOP_REASON_TOOL_USE:
            return "toolUse", None
        case _:
            return ("error", f"Provider stopped with: {reason}") if reason else ("error", None)


def _get_configured_bedrock_region(options: BedrockOptions) -> str | None:
    return (
        options.region
        or get_provider_env_value("AWS_REGION", options.env)
        or get_provider_env_value("AWS_DEFAULT_REGION", options.env)
        or None
    )


def _get_configured_bedrock_credentials(env: ProviderEnv | None = None) -> dict[str, str] | None:
    access_key_id = get_provider_env_value("AWS_ACCESS_KEY_ID", env)
    secret_access_key = get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
    if not access_key_id or not secret_access_key:
        return None
    session_token = get_provider_env_value("AWS_SESSION_TOKEN", env)
    return {
        "accessKeyId": access_key_id,
        "secretAccessKey": secret_access_key,
        **({"sessionToken": session_token} if session_token else {}),
    }


def _get_standard_bedrock_endpoint_region(base_url: str | None) -> str | None:
    if not base_url:
        return None

    hostname = urlparse(base_url).hostname
    if not hostname:
        return None
    match = _STANDARD_ENDPOINT.match(hostname.lower())
    return match.group(1) if match else None


def _should_use_explicit_bedrock_endpoint(
    base_url: str, configured_region: str | None, has_ambient_configured_profile: bool
) -> bool:
    endpoint_region = _get_standard_bedrock_endpoint_region(base_url)
    if not endpoint_region:
        return True
    return not configured_region and not has_ambient_configured_profile


def _is_gov_cloud_bedrock_target(model: Model, options: BedrockOptions) -> bool:
    region = _get_configured_bedrock_region(options)
    if region and region.lower().startswith("us-gov-"):
        return True

    model_id = model.id.lower()
    return model_id.startswith(("us-gov.", "arn:aws-us-gov:"))


def build_additional_model_request_fields(model: Model, options: BedrockOptions) -> dict[str, Any] | None:
    if not options.reasoning or not model.reasoning:
        return None

    if not _is_anthropic_claude_model(model):
        return None

    # GovCloud Bedrock currently rejects the Claude thinking.display field.
    display = (
        None
        if _is_gov_cloud_bedrock_target(model, options)
        else (options.thinking_display if options.thinking_display is not None else "summarized")
    )

    if _supports_adaptive_thinking(model.id, model.name):
        return {
            "thinking": {"type": "adaptive", **({"display": display} if display is not None else {})},
            "output_config": {"effort": _map_thinking_level_to_effort(model, options.reasoning)},
        }

    default_budgets = {
        "minimal": 1024,
        "low": 2048,
        "medium": 8192,
        "high": 16384,
        # Budget-based Claude clamps extended levels to high.
        "xhigh": 16384,
        "max": 16384,
    }
    level = "high" if options.reasoning in ("xhigh", "max") else options.reasoning
    custom = getattr(options.thinking_budgets, level, None) if options.thinking_budgets else None
    budget = custom if custom is not None else default_budgets[options.reasoning]

    result: dict[str, Any] = {
        "thinking": {
            "type": "enabled",
            "budget_tokens": budget,
            **({"display": display} if display is not None else {}),
        }
    }
    if options.interleaved_thinking is None or options.interleaved_thinking:
        result["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]
    return result


_IMAGE_FORMATS = {
    "image/jpeg": IMAGE_FORMAT_JPEG,
    "image/jpg": IMAGE_FORMAT_JPEG,
    "image/png": IMAGE_FORMAT_PNG,
    "image/gif": IMAGE_FORMAT_GIF,
    "image/webp": IMAGE_FORMAT_WEBP,
}


def create_image_block(mime_type: str, data: str) -> dict[str, Any]:
    format = _IMAGE_FORMATS.get(mime_type)
    if format is None:
        raise ValueError(f"Unknown image type: {mime_type}")
    return {"source": {"bytes": base64.b64decode(data)}, "format": format}
