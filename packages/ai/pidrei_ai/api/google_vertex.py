"""Port of pi's Google Vertex AI adapter (packages/ai/src/api/google-vertex.ts).

pi's client comes from `@google/genai`, which also brings Application Default
Credentials with it; here the client is `api/google_client.py` and ADC is
`auth/google_adc.py`. The streaming state machine mirrors pi's — which is its own
copy of the Gemini adapter's, duplication included, so a future pi diff lands on
the same two files it does upstream.

Note the two deliberate differences from `google_generative_ai.py`, both pi's:
this adapter has no Gemma branch in its thinking-level mapping, and no
`2.5-flash-lite` row in its budget table.
"""

import itertools
import json
import re
import time
from dataclasses import dataclass, fields
from typing import Any
from urllib.parse import urlparse

import tonio.colored as tonio

from pidrei_ai.api.google_client import RESOURCE_SCOPE_COLLECTION, GoogleGenAI
from pidrei_ai.api.google_shared import (
    GoogleThinkingLevel,
    convert_messages,
    convert_tools,
    is_thinking_part,
    map_stop_reason,
    resolve_google_function_calling_mode,
    retain_thought_signature,
    supports_google_strict_tool_sampling,
)
from pidrei_ai.api.simple_options import build_base_options
from pidrei_ai.registry import calculate_cost, clamp_thinking_level
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Model,
    ProviderEnv,
    ProviderHeaders,
    SimpleStreamOptions,
    StartEvent,
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
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    Usage,
    UsageCost,
)
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.error_body import format_provider_error, normalize_provider_error
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.headers import provider_headers_to_record
from pidrei_ai.utils.provider_env import get_provider_env_value
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates


API_VERSION = "v1"
GCP_VERTEX_CREDENTIALS_MARKER = "gcp-vertex-credentials"

_GEMINI_3_PRO = re.compile(r"gemini-3(?:\.\d+)?-pro")
_GEMINI_3_FLASH = re.compile(r"gemini-3(?:\.\d+)?-flash")
_PLACEHOLDER_API_KEY = re.compile(r"^<[^>]+>$")
_API_VERSION_SEGMENT = re.compile(r"^v\d+(?:beta\d*)?$")
_API_VERSION_IN_PATH = re.compile(r"(?:^|/)v\d+(?:beta\d*)?(?:/|$)")


@dataclass(slots=True)
class GoogleVertexThinking:
    enabled: bool
    budget_tokens: int | None = None  # -1 for dynamic, 0 to disable
    level: GoogleThinkingLevel | None = None


@dataclass(slots=True)
class GoogleVertexOptions(StreamOptions):
    tool_choice: str | None = None  # "auto" | "none" | "any"
    thinking: GoogleVertexThinking | None = None
    project: str | None = None
    location: str | None = None


# Counter for generating unique tool call IDs. pi's `++toolCallCounter` is safe
# because JavaScript is single-threaded; `count()` keeps it safe here.
_tool_call_counter = itertools.count(1)


def _vertex_options(options: StreamOptions | None) -> GoogleVertexOptions:
    if isinstance(options, GoogleVertexOptions):
        return options
    if options is None:
        return GoogleVertexOptions()
    values = {f.name: getattr(options, f.name) for f in fields(StreamOptions)}
    return GoogleVertexOptions(**values)


def stream(model: Model, context: Context, options: StreamOptions | None = None) -> AssistantMessageEventStream:
    opts = _vertex_options(options)
    out_stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            content=[],
            api="google-vertex",
            provider=model.provider,
            model=model.id,
            usage=Usage(),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )

        try:
            api_key = resolve_api_key(opts)
            # Create the client using either a Vertex API key, if provided, or ADC
            # with project and location
            client = (
                create_client_with_api_key(model, api_key, opts.headers)
                if api_key
                else create_client(model, resolve_project(opts), resolve_location(opts), opts.headers, opts.env)
            )
            params = build_params(model, context, opts)
            next_params = await maybe_call(opts.on_payload, params, model)
            if next_params is not None:
                params = next_params
            google_stream = client.generate_content_stream(params, env=opts.env, cancel=opts.cancel)

            out_stream.push(StartEvent(partial=output))
            current_block: TextContent | ThinkingContent | None = None
            blocks = output.content

            def block_index() -> int:
                return len(blocks) - 1

            async for chunk in google_stream:
                # Vertex uses the same GenerateContentResponse shape as Gemini;
                # responseId is an output-only identifier for each response.
                if not output.response_id:
                    output.response_id = chunk.get("responseId")
                candidates = chunk.get("candidates") or []
                candidate = candidates[0] if candidates else None
                parts = ((candidate or {}).get("content") or {}).get("parts")
                if parts:
                    for part in parts:
                        text = part.get("text")
                        if text is not None:
                            is_thinking = is_thinking_part(part)
                            if (
                                current_block is None
                                or (is_thinking and current_block.type != "thinking")
                                or (not is_thinking and current_block.type != "text")
                            ):
                                if current_block is not None:
                                    if current_block.type == "text":
                                        out_stream.push(
                                            TextEndEvent(
                                                content_index=len(blocks) - 1,
                                                content=current_block.text,
                                                partial=output,
                                            )
                                        )
                                    else:
                                        out_stream.push(
                                            ThinkingEndEvent(
                                                content_index=block_index(),
                                                content=current_block.thinking,
                                                partial=output,
                                            )
                                        )
                                if is_thinking:
                                    current_block = ThinkingContent(thinking="", thinking_signature=None)
                                    output.content.append(current_block)
                                    out_stream.push(ThinkingStartEvent(content_index=block_index(), partial=output))
                                else:
                                    current_block = TextContent(text="")
                                    output.content.append(current_block)
                                    out_stream.push(TextStartEvent(content_index=block_index(), partial=output))
                            if current_block.type == "thinking":
                                current_block.thinking += text
                                current_block.thinking_signature = retain_thought_signature(
                                    current_block.thinking_signature, part.get("thoughtSignature")
                                )
                                out_stream.push(
                                    ThinkingDeltaEvent(content_index=block_index(), delta=text, partial=output)
                                )
                            else:
                                current_block.text += text
                                current_block.text_signature = retain_thought_signature(
                                    current_block.text_signature, part.get("thoughtSignature")
                                )
                                out_stream.push(TextDeltaEvent(content_index=block_index(), delta=text, partial=output))

                        function_call = part.get("functionCall")
                        if function_call:
                            if current_block is not None:
                                if current_block.type == "text":
                                    out_stream.push(
                                        TextEndEvent(
                                            content_index=block_index(),
                                            content=current_block.text,
                                            partial=output,
                                        )
                                    )
                                else:
                                    out_stream.push(
                                        ThinkingEndEvent(
                                            content_index=block_index(),
                                            content=current_block.thinking,
                                            partial=output,
                                        )
                                    )
                                current_block = None

                            provided_id = function_call.get("id")
                            needs_new_id = not provided_id or any(
                                b.type == "toolCall" and b.id == provided_id for b in output.content
                            )
                            tool_call_id = (
                                f"{function_call.get('name')}_{int(time.time() * 1000)}_{next(_tool_call_counter)}"
                                if needs_new_id
                                else provided_id
                            )

                            tool_call = ToolCall(
                                id=tool_call_id,
                                name=function_call.get("name") or "",
                                arguments=function_call.get("args") or {},
                                thought_signature=part.get("thoughtSignature") or None,
                            )

                            output.content.append(tool_call)
                            out_stream.push(ToolCallStartEvent(content_index=block_index(), partial=output))
                            out_stream.push(
                                ToolCallDeltaEvent(
                                    content_index=block_index(),
                                    delta=json.dumps(tool_call.arguments, separators=(",", ":")),
                                    partial=output,
                                )
                            )
                            out_stream.push(
                                ToolCallEndEvent(content_index=block_index(), tool_call=tool_call, partial=output)
                            )

                finish_reason = (candidate or {}).get("finishReason")
                if finish_reason:
                    output.raw_stop_reason = finish_reason
                    output.stop_reason = map_stop_reason(finish_reason)
                    if any(b.type == "toolCall" for b in output.content):
                        output.stop_reason = "toolUse"

                usage_metadata = chunk.get("usageMetadata")
                if usage_metadata:
                    output.usage = _usage_from_metadata(usage_metadata)
                    calculate_cost(model, output.usage)

            if current_block is not None:
                if current_block.type == "text":
                    out_stream.push(
                        TextEndEvent(content_index=block_index(), content=current_block.text, partial=output)
                    )
                else:
                    out_stream.push(
                        ThinkingEndEvent(content_index=block_index(), content=current_block.thinking, partial=output)
                    )

            if opts.cancel is not None and opts.cancel.cancelled:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Google Vertex stream ended without a finish reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(
                    f"Provider stopped with: {output.raw_stop_reason}"
                    if output.raw_stop_reason
                    else "An unknown error occurred"
                )

            out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            out_stream.end()
        except Exception as error:
            output.stop_reason = "aborted" if opts.cancel is not None and opts.cancel.cancelled else "error"
            output.error_message = format_provider_error(normalize_provider_error(error))
            out_stream.push(ErrorEvent(reason=output.stop_reason, error=output))
            out_stream.end()

    tonio.spawn.without_tracking(_run())
    return out_stream


def _usage_from_metadata(metadata: dict[str, Any]) -> Usage:
    cached = metadata.get("cachedContentTokenCount") or 0
    thoughts = metadata.get("thoughtsTokenCount") or 0
    return Usage(
        input=(metadata.get("promptTokenCount") or 0) - cached,
        output=(metadata.get("candidatesTokenCount") or 0) + thoughts,
        cache_read=cached,
        cache_write=0,
        reasoning=thoughts,
        total_tokens=metadata.get("totalTokenCount") or 0,
        cost=UsageCost(),
    )


def stream_simple(
    model: Model, context: Context, options: SimpleStreamOptions | None = None
) -> AssistantMessageEventStream:
    base = build_base_options(model, context, options, None)
    if options is None or not options.reasoning:
        return stream(model, context, _with_thinking(base, GoogleVertexThinking(enabled=False)))

    clamped_reasoning = clamp_thinking_level(model, options.reasoning)
    effort = "high" if clamped_reasoning == "off" else clamped_reasoning

    if _is_gemini_3_pro_model(model) or _is_gemini_3_flash_model(model):
        return stream(
            model,
            context,
            _with_thinking(base, GoogleVertexThinking(enabled=True, level=_get_gemini_3_thinking_level(effort, model))),
        )

    return stream(
        model,
        context,
        _with_thinking(
            base,
            GoogleVertexThinking(
                enabled=True, budget_tokens=_get_google_budget(model, effort, options.thinking_budgets)
            ),
        ),
    )


def _with_thinking(base: StreamOptions, thinking: GoogleVertexThinking) -> GoogleVertexOptions:
    opts = _vertex_options(base)
    opts.thinking = thinking
    # No project/location here on purpose: `build_base_options` carries only the
    # shared `StreamOptions` fields, so `stream_simple` reaches Vertex without
    # them and `resolve_project`/`resolve_location` fall back to the environment.
    # pi's `buildBaseOptions` drops them the same way.
    return opts


def create_client(
    model: Model,
    project: str,
    location: str,
    options_headers: ProviderHeaders | None = None,
    env: ProviderEnv | None = None,
) -> GoogleGenAI:
    google_auth_options = build_google_auth_options(env)
    return GoogleGenAI(
        {
            "vertexai": True,
            "project": project,
            "location": location,
            "apiVersion": API_VERSION,
            **({"googleAuthOptions": google_auth_options} if google_auth_options else {}),
            "httpOptions": build_http_options(model, options_headers),
        }
    )


def create_client_with_api_key(
    model: Model, api_key: str, options_headers: ProviderHeaders | None = None
) -> GoogleGenAI:
    return GoogleGenAI(
        {
            "vertexai": True,
            "apiKey": api_key,
            "apiVersion": API_VERSION,
            "httpOptions": build_http_options(model, options_headers),
        }
    )


def build_http_options(model: Model, options_headers: ProviderHeaders | None = None) -> dict[str, Any] | None:
    http_options: dict[str, Any] = {}
    base_url = resolve_custom_base_url(model.base_url)
    if base_url:
        http_options["baseUrl"] = base_url
        http_options["baseUrlResourceScope"] = RESOURCE_SCOPE_COLLECTION
        if base_url_includes_api_version(base_url):
            http_options["apiVersion"] = ""

    headers = provider_headers_to_record({**(model.headers or {}), **(options_headers or {})})
    if headers:
        http_options["headers"] = headers

    return http_options if http_options else None


def resolve_custom_base_url(base_url: str) -> str | None:
    trimmed = (base_url or "").strip()
    if not trimmed or "{location}" in trimmed:
        return None
    return trimmed


def base_url_includes_api_version(base_url: str) -> bool:
    parsed = urlparse(base_url)
    if parsed.scheme and parsed.netloc:
        return any(_API_VERSION_SEGMENT.match(part) for part in parsed.path.split("/"))
    return _API_VERSION_IN_PATH.search(base_url) is not None


def build_google_auth_options(env: ProviderEnv | None = None) -> dict[str, str] | None:
    key_filename = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
    return {"keyFilename": key_filename} if key_filename else None


def resolve_api_key(options: GoogleVertexOptions | None = None) -> str | None:
    api_key = (options.api_key or "").strip() if options is not None else ""
    if not api_key or api_key == GCP_VERTEX_CREDENTIALS_MARKER or _is_placeholder_api_key(api_key):
        return None
    return api_key


def _is_placeholder_api_key(api_key: str) -> bool:
    return _PLACEHOLDER_API_KEY.match(api_key) is not None


def resolve_project(options: GoogleVertexOptions | None = None) -> str:
    env = options.env if options is not None else None
    project = (
        (options.project if options is not None else None)
        or get_provider_env_value("GOOGLE_CLOUD_PROJECT", env)
        or get_provider_env_value("GCLOUD_PROJECT", env)
    )
    if not project:
        raise RuntimeError(
            "Vertex AI requires a project ID. Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def resolve_location(options: GoogleVertexOptions | None = None) -> str:
    env = options.env if options is not None else None
    location = (options.location if options is not None else None) or get_provider_env_value(
        "GOOGLE_CLOUD_LOCATION", env
    )
    if not location:
        raise RuntimeError("Vertex AI requires a location. Set GOOGLE_CLOUD_LOCATION or pass location in options.")
    return location


def build_params(model: Model, context: Context, options: GoogleVertexOptions | None = None) -> dict[str, Any]:
    options = options if options is not None else GoogleVertexOptions()
    contents = convert_messages(model, context)

    generation_config: dict[str, Any] = {}
    if options.temperature is not None:
        generation_config["temperature"] = options.temperature
    if options.max_tokens is not None:
        generation_config["maxOutputTokens"] = options.max_tokens

    function_calling_mode = (
        resolve_google_function_calling_mode(
            context.tools, options.tool_choice, supports_google_strict_tool_sampling(model.id)
        )
        if context.tools
        else None
    )
    config: dict[str, Any] = {
        **generation_config,
        **({"systemInstruction": sanitize_surrogates(context.system_prompt)} if context.system_prompt else {}),
        **({"tools": convert_tools(context.tools)} if context.tools else {}),
        **(
            {"toolConfig": {"functionCallingConfig": {"mode": function_calling_mode}}}
            if function_calling_mode is not None
            else {}
        ),
    }

    if options.thinking is not None and options.thinking.enabled and model.reasoning:
        thinking_config: dict[str, Any] = {"includeThoughts": True}
        if options.thinking.level is not None:
            thinking_config["thinkingLevel"] = options.thinking.level
        elif options.thinking.budget_tokens is not None:
            thinking_config["thinkingBudget"] = options.thinking.budget_tokens
        config["thinkingConfig"] = thinking_config
    elif model.reasoning and options.thinking is not None and not options.thinking.enabled:
        config["thinkingConfig"] = _get_disabled_thinking_config(model)

    if options.cancel is not None and options.cancel.cancelled:
        raise RuntimeError("Request aborted")

    return {"model": model.id, "contents": contents, "config": config}


def _is_gemini_3_pro_model(model: Model) -> bool:
    return _GEMINI_3_PRO.search(model.id.lower()) is not None


def _is_gemini_3_flash_model(model: Model) -> bool:
    id = model.id.lower()
    return _GEMINI_3_FLASH.search(id) is not None or id == "gemini-flash-latest" or id == "gemini-flash-lite-latest"


def _get_disabled_thinking_config(model: Model) -> dict[str, Any]:
    # Google docs: Gemini 3.1 Pro cannot disable thinking, and Gemini 3 Flash / Flash-Lite
    # do not support full thinking-off either. For Gemini 3 models, use the lowest supported
    # thinkingLevel without includeThoughts so hidden thinking remains invisible to pidrei.
    if _is_gemini_3_pro_model(model):
        return {"thinkingLevel": "LOW"}
    if _is_gemini_3_flash_model(model):
        return {"thinkingLevel": "MINIMAL"}

    # Gemini 2.x supports disabling via thinkingBudget = 0.
    return {"thinkingBudget": 0}


def _get_gemini_3_thinking_level(effort: str, model: Model) -> GoogleThinkingLevel:
    if _is_gemini_3_pro_model(model):
        if effort in ("minimal", "low"):
            return "LOW"
        return "HIGH"
    match effort:
        case "minimal":
            return "MINIMAL"
        case "low":
            return "LOW"
        case "medium":
            return "MEDIUM"
        case _:
            return "HIGH"


def _get_google_budget(model: Model, effort: str, custom_budgets: ThinkingBudgets | None = None) -> int:
    if custom_budgets is not None and getattr(custom_budgets, effort, None) is not None:
        return getattr(custom_budgets, effort)

    if "2.5-pro" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 32768}[effort]

    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[effort]

    return -1
