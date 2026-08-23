"""Port of pi's Google Generative AI adapter (packages/ai/src/api/google-generative-ai.ts).

pi's `GoogleGenAI` client comes from `@google/genai`; here it comes from
`api/google_client.py`, which is that SDK's request path hand-rolled over the
punkreq seam. Everything else — the streaming state machine, the thinking-level
and budget tables, the disabled-thinking configs — mirrors pi.
"""

import itertools
import json
import re
import time
from dataclasses import dataclass, fields
from typing import Any

from pidrei_ai.api.google_client import GoogleGenAI
from pidrei_ai.api.google_shared import (
    GoogleThinkingLevel,
    convert_messages,
    convert_tools,
    is_thinking_part,
    map_stop_reason,
    resolve_google_function_calling_mode,
    retain_thought_signature,
    retry_google_request,
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
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates


_GEMMA_4 = re.compile(r"gemma-?4")
_GEMINI_3_PRO = re.compile(r"gemini-3(?:\.\d+)?-pro")
_GEMINI_3_FLASH = re.compile(r"gemini-3(?:\.\d+)?-flash")


@dataclass(slots=True)
class GoogleThinking:
    enabled: bool
    budget_tokens: int | None = None  # -1 for dynamic, 0 to disable
    level: GoogleThinkingLevel | None = None


@dataclass(slots=True)
class GoogleOptions(StreamOptions):
    tool_choice: str | None = None  # "auto" | "none" | "any"
    thinking: GoogleThinking | None = None


# Counter for generating unique tool call IDs. pi's `++toolCallCounter` is safe
# because JavaScript is single-threaded; `count()` keeps it safe here, where two
# turns really can interleave.
_tool_call_counter = itertools.count(1)


def _google_options(options: StreamOptions | None) -> GoogleOptions:
    if isinstance(options, GoogleOptions):
        return options
    if options is None:
        return GoogleOptions()
    values = {f.name: getattr(options, f.name) for f in fields(StreamOptions)}
    return GoogleOptions(**values)


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    opts = _google_options(options)
    out_stream = into if into is not None else AssistantMessageEventStream()

    output = AssistantMessage(
        content=[],
        api="google-generative-ai",
        provider=model.provider,
        model=model.id,
        usage=Usage(),
        stop_reason="pending",
        timestamp=int(time.time() * 1000),
    )
    out_stream.partial = output

    async def _run() -> None:
        try:
            api_key = opts.api_key
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")
            client = create_client(model, api_key, opts.headers)
            params = build_params(model, context, opts)
            next_params = await maybe_call(opts.on_payload, params, model)
            if next_params is not None:
                params = next_params
            google_stream = await retry_google_request(
                lambda: client.generate_content_stream(params, env=opts.env, cancel=opts.cancel), opts
            )

            out_stream.push(StartEvent(partial=output))
            current_block: TextContent | ThinkingContent | None = None
            blocks = output.content

            def block_index() -> int:
                return len(blocks) - 1

            async for chunk in google_stream:
                # `responseId` is documented as an output-only field identifying each
                # response. Keep the first non-empty one from the stream.
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

                            # Generate unique ID if not provided or if it's a duplicate
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
                    if any(b.type == "toolCall" for b in output.content) and output.stop_reason == "stop":
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
                raise RuntimeError("Google stream ended without a finish reason")
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

    out_stream.spawn_producer(_run(), opts.cancel)
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
    if options is None or not options.reasoning:
        return stream(model, context, _with_thinking(base, GoogleThinking(enabled=False)), into=into)

    clamped_reasoning = clamp_thinking_level(model, options.reasoning)
    effort = "high" if clamped_reasoning == "off" else clamped_reasoning

    if _is_gemini_3_pro_model(model) or _is_gemini_3_flash_model(model) or _is_gemma_4_model(model):
        return stream(
            model,
            context,
            _with_thinking(base, GoogleThinking(enabled=True, level=_get_thinking_level(effort, model))),
            into=into,
        )

    return stream(
        model,
        context,
        _with_thinking(
            base,
            GoogleThinking(enabled=True, budget_tokens=_get_google_budget(model, effort, options.thinking_budgets)),
        ),
        into=into,
    )


def _with_thinking(base: StreamOptions, thinking: GoogleThinking) -> GoogleOptions:
    opts = _google_options(base)
    opts.thinking = thinking
    return opts


def create_client(model: Model, api_key: str | None = None, options_headers: ProviderHeaders | None = None):
    http_options: dict[str, Any] = {}
    if model.base_url:
        http_options["baseUrl"] = model.base_url
        http_options["apiVersion"] = ""  # baseUrl already includes version path, don't append
    headers = provider_headers_to_record({**(model.headers or {}), **(options_headers or {})})
    if headers:
        http_options["headers"] = headers

    return GoogleGenAI({"apiKey": api_key, "httpOptions": http_options if http_options else None})


def build_params(model: Model, context: Context, options: GoogleOptions | None = None) -> dict[str, Any]:
    options = options if options is not None else GoogleOptions()
    contents = convert_messages(model, context)

    generation_config: dict[str, Any] = {}
    if options.temperature is not None:
        generation_config["temperature"] = options.temperature
    if options.max_tokens is not None:
        generation_config["maxOutputTokens"] = options.max_tokens

    supports_strict_mode = supports_google_strict_tool_sampling(model.id)
    function_calling_mode = (
        resolve_google_function_calling_mode(context.tools, options.tool_choice, supports_strict_mode)
        if context.tools
        else None
    )
    config: dict[str, Any] = {
        **generation_config,
        **({"systemInstruction": sanitize_surrogates(context.system_prompt)} if context.system_prompt else {}),
        **({"tools": convert_tools(context.tools, False, supports_strict_mode)} if context.tools else {}),
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


def _is_gemma_4_model(model: Model) -> bool:
    return _GEMMA_4.search(model.id.lower()) is not None


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
    if _is_gemma_4_model(model):
        return {"thinkingLevel": "MINIMAL"}

    # Gemini 2.x supports disabling via thinkingBudget = 0.
    return {"thinkingBudget": 0}


def _get_thinking_level(effort: str, model: Model) -> GoogleThinkingLevel:
    if _is_gemini_3_pro_model(model):
        if effort in ("minimal", "low"):
            return "LOW"
        return "HIGH"
    if _is_gemma_4_model(model):
        if effort in ("minimal", "low"):
            return "MINIMAL"
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

    if "2.5-flash-lite" in model.id:
        return {"minimal": 512, "low": 2048, "medium": 8192, "high": 24576}[effort]

    if "2.5-flash" in model.id:
        return {"minimal": 128, "low": 2048, "medium": 8192, "high": 24576}[effort]

    return -1
