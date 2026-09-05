"""Port of pi's Mistral adapter (packages/ai/src/api/mistral-conversations.ts).

Since 0.84.2 pi speaks to the native Mistral Chat Completions endpoint itself
(no `@mistralai/mistralai` SDK): it POSTs `{baseUrl}/v1/chat/completions` with a
bearer token and remaps the SDK-style camelCase request model to the API's
snake_case wire form with explicit per-structure key tables. The payload stays
camelCase (`maxTokens`, `promptMode`, `reasoningEffort`, `promptCacheKey`)
because that is the object pi's `onPayload` hook — and its
mistral-reasoning-mode spec — sees; conversion happens at the transport
boundary. `parameters` and `arguments` carry caller-controlled JSON whose keys
are never touched.

Deviations (established adapter conventions): pi's per-request `fetch`
injection is a client-injection seam here (`MistralOptions.client`), and pi's
whole-request `AbortSignal.timeout(timeoutMs ?? 60_000)` maps to the punkreq
read timeout via `http.request_timeout` (a legitimately long stream must not
be cut off; see that helper's docstring).
"""

import json
import time
from dataclasses import dataclass, fields
from typing import Any

from pidrei_ai.api.constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from pidrei_ai.api.simple_options import build_base_options
from pidrei_ai.api.transform_messages import transform_messages
from pidrei_ai.builders import (
    AssistantMessageBuilder,
    TextContentBuilder,
    ThinkingContentBuilder,
    ToolCallBuilder,
    UsageBuilder,
)
from pidrei_ai.registry import calculate_cost, clamp_thinking_level
from pidrei_ai.types import (
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
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
    ThinkingStartEvent,
    Tool,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from pidrei_ai.utils import http
from pidrei_ai.utils.callbacks import maybe_call
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.hash import short_hash
from pidrei_ai.utils.json_parse import parse_streaming_json
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates
from pidrei_ai.utils.sse import iterate_sse_messages
from pidrei_ai.utils.user_agent import get_user_agent


MISTRAL_TOOL_CALL_ID_LENGTH = 9
MAX_MISTRAL_ERROR_BODY_CHARS = 4000


class MistralHttpError(Exception):
    def __init__(self, status_code: int, body: str, status_text: str = ""):
        super().__init__(status_text or f"Request failed with status {status_code}")
        self.status_code = status_code
        self.body = body


class _PunkreqMistralClient:
    """Default transport: POST the wire payload through the punkreq seam."""

    def __init__(self, env=None):
        self._env = env

    async def post_chat_completions(
        self,
        url: str,
        wire_payload: dict[str, Any],
        *,
        headers: dict[str, str],
        timeout_ms: float | None,
        cancel: CancelToken | None,
    ):
        client = http.client_for(url, self._env)
        return await client.post(url, json=wire_payload, headers=headers, timeout=http.request_timeout(timeout_ms))


async def request_mistral_stream(model: Model, payload: dict[str, Any], api_key: str, options: MistralOptions):
    url = f"{(model.base_url or 'https://api.mistral.ai').rstrip('/')}/v1/chat/completions"
    headers = build_mistral_headers(model, api_key, options)
    client = options.client if options.client is not None else _PunkreqMistralClient(env=options.env)
    response = await client.post_chat_completions(
        url,
        to_mistral_wire_payload(payload),
        headers=headers,
        timeout_ms=options.timeout_ms,
        cancel=options.cancel,
    )

    await maybe_call(
        options.on_response, ProviderResponse(status=response.status_code, headers=dict(response.headers)), model
    )

    if not 200 <= response.status_code < 300:
        body = (await response.read()).decode("utf-8", "replace")
        raise MistralHttpError(response.status_code, body)
    return _iterate_completion_events(response, options.cancel)


def build_mistral_headers(model: Model, api_key: str, options: MistralOptions | None = None) -> dict[str, str]:
    """pi builds a case-insensitive `Headers`; lowercase keys stand in for that."""
    headers = {
        "user-agent": get_user_agent(),
        "accept": "text/event-stream",
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    _apply_mistral_header_overrides(headers, model.headers)
    _apply_mistral_header_overrides(headers, options.headers if options is not None else None)

    has_explicit_affinity = _has_mistral_header_override(model.headers, "x-affinity") or _has_mistral_header_override(
        options.headers if options is not None else None, "x-affinity"
    )
    if _should_use_prompt_caching(options) and not has_explicit_affinity:
        headers["x-affinity"] = options.session_id

    return headers


def _apply_mistral_header_overrides(headers: dict[str, str], overrides: dict[str, str | None] | None) -> None:
    if not overrides:
        return
    for name, value in overrides.items():
        if value is None:
            headers.pop(name.lower(), None)
        else:
            headers[name.lower()] = value


def _has_mistral_header_override(overrides: dict[str, str | None] | None, target: str) -> bool:
    return bool(overrides) and any(name.lower() == target for name in overrides)


_TOP_LEVEL_WIRE_KEYS = (
    ("topP", "top_p"),
    ("maxTokens", "max_tokens"),
    ("randomSeed", "random_seed"),
    ("responseFormat", "response_format"),
    ("toolChoice", "tool_choice"),
    ("presencePenalty", "presence_penalty"),
    ("frequencyPenalty", "frequency_penalty"),
    ("parallelToolCalls", "parallel_tool_calls"),
    ("reasoningEffort", "reasoning_effort"),
    ("promptMode", "prompt_mode"),
    ("promptCacheKey", "prompt_cache_key"),
    ("safePrompt", "safe_prompt"),
)

_CONTENT_CHUNK_WIRE_KEYS = (
    ("imageUrl", "image_url"),
    ("documentUrl", "document_url"),
    ("documentName", "document_name"),
    ("fileId", "file_id"),
    ("referenceIds", "reference_ids"),
    ("inputAudio", "input_audio"),
)


def to_mistral_wire_payload(payload: dict[str, Any]) -> dict[str, Any]:
    wire_payload = dict(payload)
    for source, target in _TOP_LEVEL_WIRE_KEYS:
        _remap_mistral_property(wire_payload, source, target)
    wire_payload["messages"] = [to_mistral_wire_message(message) for message in payload.get("messages") or []]

    response_format = wire_payload.get("response_format")
    if isinstance(response_format, dict):
        wire_response_format = dict(response_format)
        _remap_mistral_property(wire_response_format, "jsonSchema", "json_schema")
        json_schema = wire_response_format.get("json_schema")
        if isinstance(json_schema, dict):
            wire_json_schema = dict(json_schema)
            _remap_mistral_property(wire_json_schema, "schemaDefinition", "schema")
            wire_response_format["json_schema"] = wire_json_schema
        wire_payload["response_format"] = wire_response_format

    return wire_payload


def to_mistral_wire_message(message: dict[str, Any]) -> dict[str, Any]:
    wire_message = dict(message)
    _remap_mistral_property(wire_message, "toolCalls", "tool_calls")
    _remap_mistral_property(wire_message, "toolCallId", "tool_call_id")
    if isinstance(message.get("content"), list):
        wire_message["content"] = [to_mistral_wire_content_chunk(chunk) for chunk in message["content"]]
    return wire_message


def to_mistral_wire_content_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    wire_chunk = dict(chunk)
    for source, target in _CONTENT_CHUNK_WIRE_KEYS:
        _remap_mistral_property(wire_chunk, source, target)
    return wire_chunk


def _remap_mistral_property(record: dict[str, Any], source: str, target: str) -> None:
    if source not in record:
        return
    record[target] = record.pop(source)


async def _iterate_completion_events(response: Any, cancel: CancelToken | None):
    body = response.aiter_bytes() if hasattr(response, "aiter_bytes") else response.iter_bytes()
    ended = False
    try:
        async for message in iterate_sse_messages(body):
            if message.data == "[DONE]":
                ended = True
                return
            chunk = json.loads(message.data)
            if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
                raise RuntimeError("Invalid Mistral streaming event")  # noqa: TRY004 (mirrors pi's plain Error)
            yield chunk
        ended = True
    finally:
        await http.finish_body(body, response, drain=ended, cancel=cancel)


@dataclass(slots=True)
class MistralOptions(StreamOptions):
    # "auto" | "none" | "any" | "required" | {"type": "function", "function": {"name": ...}}
    tool_choice: Any = None
    prompt_mode: str | None = None  # "reasoning"
    reasoning_effort: str | None = None  # "none" | "high"
    # Transport seam (pi injects a per-request `fetch` here).
    client: Any = None


def _mistral_options(options: StreamOptions | None) -> MistralOptions:
    if isinstance(options, MistralOptions):
        return options
    if options is None:
        return MistralOptions()
    values = {f.name: getattr(options, f.name) for f in fields(StreamOptions)}
    return MistralOptions(**values)


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
    *,
    into: AssistantMessageEventStream | None = None,
) -> AssistantMessageEventStream:
    opts = _mistral_options(options)
    out_stream = into if into is not None else AssistantMessageEventStream()

    output = create_output(model)
    out_stream.partial = output

    async def _run() -> None:
        try:
            api_key = opts.api_key
            if not api_key:
                raise RuntimeError(f"No API key for provider: {model.provider}")

            normalize = _create_mistral_tool_call_id_normalizer()
            transformed_messages = transform_messages(
                context.messages, model, lambda id, _model, _source: normalize(id)
            )

            payload = build_chat_payload(model, context, transformed_messages, opts)
            next_payload = await maybe_call(opts.on_payload, payload, model)
            if next_payload is not None:
                payload = next_payload
            mistral_stream = await request_mistral_stream(model, payload, api_key, opts)
            out_stream.push(StartEvent(partial=output))
            await consume_chat_stream(model, output, out_stream, mistral_stream)

            if opts.cancel is not None and opts.cancel.cancelled:
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Mistral stream ended without a finish reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            out_stream.push(DoneEvent(reason=output.stop_reason, message=output))
            out_stream.end()
        except Exception as error:
            output.stop_reason = "aborted" if opts.cancel is not None and opts.cancel.cancelled else "error"
            output.error_message = format_mistral_error(error)
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
        raise Exception(f"No API key for provider: {model.provider}")

    base = build_base_options(model, context, options, api_key)
    clamped_reasoning = (
        clamp_thinking_level(model, options.reasoning) if options is not None and options.reasoning else None
    )
    reasoning = None if clamped_reasoning == "off" else clamped_reasoning
    should_use_reasoning = model.reasoning and reasoning is not None

    opts = _mistral_options(base)
    opts.tool_choice = options.tool_choice if options else None
    opts.prompt_mode = "reasoning" if should_use_reasoning and _uses_prompt_mode_reasoning(model) else None
    opts.reasoning_effort = (
        _map_reasoning_effort(model, reasoning) if should_use_reasoning and _uses_reasoning_effort(model) else None
    )
    return stream(model, context, opts, into=into)


def create_output(model: Model) -> AssistantMessageBuilder:
    return AssistantMessageBuilder(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.id,
        usage=UsageBuilder(),
        stop_reason="pending",
        timestamp=int(time.time() * 1000),
    )


def _create_mistral_tool_call_id_normalizer():
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def normalize(id: str) -> str:
        existing = id_map.get(id)
        if existing:
            return existing

        attempt = 0
        while True:
            candidate = derive_mistral_tool_call_id(id, attempt)
            owner = reverse_map.get(candidate)
            if not owner or owner == id:
                id_map[id] = candidate
                reverse_map[candidate] = id
                return candidate
            attempt += 1

    return normalize


def derive_mistral_tool_call_id(id: str, attempt: int) -> str:
    normalized = "".join(char for char in id if char.isascii() and char.isalnum())
    if attempt == 0 and len(normalized) == MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or id
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    hashed = short_hash(seed)
    return "".join(char for char in hashed if char.isascii() and char.isalnum())[:MISTRAL_TOOL_CALL_ID_LENGTH]


def format_mistral_error(error: Any) -> str:
    if isinstance(error, BaseException):
        status_code = getattr(error, "status_code", None)
        body = getattr(error, "body", None)
        body_text = body.strip() if isinstance(body, str) else None
        if isinstance(status_code, int) and body_text:
            return f"Mistral API error ({status_code}): {truncate_error_text(body_text, MAX_MISTRAL_ERROR_BODY_CHARS)}"
        if isinstance(status_code, int):
            return f"Mistral API error ({status_code}): {error}"
        return str(error)
    return safe_json_stringify(error)


def truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"))
    except Exception:
        return str(value)


def build_chat_payload(
    model: Model, context: Context, messages: list[Message], options: MistralOptions | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.id,
        "stream": True,
        "messages": to_chat_messages(messages, "image" in model.input),
    }

    if context.tools:
        payload["tools"] = to_function_tools(context.tools)
    if options is not None and options.temperature is not None:
        payload["temperature"] = options.temperature
    if options is not None and options.max_tokens is not None:
        payload["maxTokens"] = options.max_tokens
    if options is not None and options.tool_choice:
        payload["toolChoice"] = map_tool_choice(options.tool_choice)
    if options is not None and options.prompt_mode:
        payload["promptMode"] = options.prompt_mode
    if options is not None and options.reasoning_effort:
        payload["reasoningEffort"] = options.reasoning_effort
    if _should_use_prompt_caching(options):
        payload["promptCacheKey"] = options.session_id

    if context.system_prompt:
        payload["messages"].insert(0, {"role": "system", "content": sanitize_surrogates(context.system_prompt)})

    return payload


def _should_use_prompt_caching(options: MistralOptions | None) -> bool:
    return options is not None and options.cache_retention != "none" and bool(options.session_id)


def _get_mistral_cached_prompt_tokens(usage: dict, prompt_tokens: int) -> int:
    details = (
        usage.get("promptTokensDetails")
        or usage.get("prompt_tokens_details")
        or usage.get("promptTokenDetails")
        or usage.get("prompt_token_details")
        or {}
    )
    raw = (
        (details.get("cachedTokens") if isinstance(details, dict) else None)
        or (details.get("cached_tokens") if isinstance(details, dict) else None)
        or usage.get("numCachedTokens")
        or usage.get("num_cached_tokens")
        or 0
    )
    cached = raw if isinstance(raw, (int, float)) and not isinstance(raw, bool) else 0
    return min(prompt_tokens, max(0, int(cached)))


async def consume_chat_stream(model: Model, output: AssistantMessageBuilder, out_stream, mistral_stream) -> None:
    current_block: TextContent | ThinkingContent | None = None
    blocks = output.content
    tool_blocks_by_key: dict[str | int, int] = {}
    partial_args: dict[int, str] = {}

    def block_index() -> int:
        return len(blocks) - 1

    def finish_current_block(block) -> None:
        if not block:
            return
        if block.type == "text":
            out_stream.push(TextEndEvent(content_index=block_index(), content=block.text, partial=output))
        elif block.type == "thinking":
            out_stream.push(ThinkingEndEvent(content_index=block_index(), content=block.thinking, partial=output))

    def start_text_block(text_delta: str) -> TextContentBuilder:
        nonlocal current_block
        finish_current_block(current_block)
        current_block = TextContentBuilder(text="")
        output.content.append(current_block)
        out_stream.push(TextStartEvent(content_index=block_index(), partial=output))
        return current_block

    async for chunk in mistral_stream:
        # Mistral's streamed chunk carries an id; keep the first non-empty one.
        if not output.response_id:
            output.response_id = chunk.get("id")

        usage = chunk.get("usage")
        if usage:
            prompt_tokens = usage.get("prompt_tokens") or 0
            cached_prompt_tokens = _get_mistral_cached_prompt_tokens(usage, prompt_tokens)
            output.usage.input = max(0, prompt_tokens - cached_prompt_tokens)
            output.usage.output = usage.get("completion_tokens") or 0
            output.usage.cache_read = cached_prompt_tokens
            output.usage.cache_write = 0
            output.usage.total_tokens = (
                usage.get("total_tokens")
                or output.usage.input + output.usage.output + output.usage.cache_read + output.usage.cache_write
            )
            calculate_cost(model, output.usage)

        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]

        finish_reason = choice.get("finish_reason")
        if finish_reason:
            output.raw_stop_reason = finish_reason
            stop_reason, error_message = map_chat_stop_reason(finish_reason)
            output.stop_reason = stop_reason
            if error_message:
                output.error_message = error_message

        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content is not None:
            content_items = [content] if isinstance(content, str) else content
            for item in content_items:
                if isinstance(item, str):
                    text_delta = sanitize_surrogates(item)
                    if current_block is None or current_block.type != "text":
                        current_block = start_text_block(text_delta)
                    current_block.text += text_delta
                    out_stream.push(TextDeltaEvent(content_index=block_index(), delta=text_delta, partial=output))
                    continue

                if item.get("type") == "thinking":
                    delta_text = "".join(part.get("text") or "" for part in item.get("thinking") or [])
                    thinking_delta = sanitize_surrogates(delta_text)
                    if not thinking_delta:
                        continue
                    if current_block is None or current_block.type != "thinking":
                        finish_current_block(current_block)
                        current_block = ThinkingContentBuilder(thinking="")
                        output.content.append(current_block)
                        out_stream.push(ThinkingStartEvent(content_index=block_index(), partial=output))
                    current_block.thinking += thinking_delta
                    out_stream.push(
                        ThinkingDeltaEvent(content_index=block_index(), delta=thinking_delta, partial=output)
                    )
                    continue

                if item.get("type") == "text":
                    text_delta = sanitize_surrogates(item.get("text", ""))
                    if current_block is None or current_block.type != "text":
                        current_block = start_text_block(text_delta)
                    current_block.text += text_delta
                    out_stream.push(TextDeltaEvent(content_index=block_index(), delta=text_delta, partial=output))

        tool_calls = delta.get("tool_calls") or []
        for tool_call in tool_calls:
            if current_block is not None:
                finish_current_block(current_block)
                current_block = None
            index = tool_call.get("index") or 0
            raw_id = tool_call.get("id")
            call_id = raw_id if raw_id and raw_id != "null" else derive_mistral_tool_call_id(f"toolcall:{index}", 0)
            key = tool_call["index"] if tool_call.get("index") is not None else call_id
            existing_index = tool_blocks_by_key.get(key)
            block = None

            if existing_index is not None:
                existing = output.content[existing_index]
                if existing.type == "toolCall":
                    block = existing

            if block is None:
                function = tool_call.get("function") or {}
                block = ToolCallBuilder(id=call_id, name=function.get("name") or "", arguments={})
                output.content.append(block)
                tool_blocks_by_key[key] = len(output.content) - 1
                partial_args[len(output.content) - 1] = ""
                out_stream.push(ToolCallStartEvent(content_index=len(output.content) - 1, partial=output))

            raw_arguments = (tool_call.get("function") or {}).get("arguments")
            args_delta = (
                raw_arguments
                if isinstance(raw_arguments, str)
                else json.dumps(raw_arguments or {}, separators=(",", ":"))
            )
            position = tool_blocks_by_key[key]
            partial_args[position] = partial_args.get(position, "") + args_delta
            block.arguments = parse_streaming_json(partial_args[position])
            out_stream.push(ToolCallDeltaEvent(content_index=position, delta=args_delta, partial=output))

    finish_current_block(current_block)
    for index in tool_blocks_by_key.values():
        block = output.content[index]
        if block.type != "toolCall":
            continue
        block.arguments = parse_streaming_json(partial_args.get(index, ""))
        out_stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=output))


def to_function_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    result = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, True)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": get_json_schema_tool_parameters(tool, strict),
                    "strict": strict if strict is not None else False,
                },
            }
        )
    return result


def to_chat_messages(messages: list[Message], supports_images: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                result.append({"role": "user", "content": sanitize_surrogates(msg.content)})
                continue
            had_images = any(item.type == "image" for item in msg.content)
            content: list[dict[str, Any]] = []
            for item in msg.content:
                if item.type == "text":
                    content.append({"type": "text", "text": sanitize_surrogates(item.text)})
                elif supports_images:
                    content.append({"type": "image_url", "imageUrl": f"data:{item.mime_type};base64,{item.data}"})
            if content:
                result.append({"role": "user", "content": content})
                continue
            if had_images and not supports_images:
                result.append({"role": "user", "content": "(image omitted: model does not support images)"})
            continue

        if msg.role == "assistant":
            content_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in msg.content:
                if block.type == "text":
                    if block.text.strip():
                        content_parts.append({"type": "text", "text": sanitize_surrogates(block.text)})
                    continue
                if block.type == "thinking":
                    if block.thinking.strip():
                        content_parts.append(
                            {
                                "type": "thinking",
                                "thinking": [{"type": "text", "text": sanitize_surrogates(block.thinking)}],
                            }
                        )
                    continue
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.arguments or {}, separators=(",", ":")),
                        },
                        "index": 0,
                    }
                )

            assistant_message: dict[str, Any] = {"role": "assistant", "prefix": False}
            if content_parts:
                assistant_message["content"] = content_parts
            if tool_calls:
                assistant_message["toolCalls"] = tool_calls
            if content_parts or tool_calls:
                result.append(assistant_message)
            continue

        text_result = "\n".join(sanitize_surrogates(part.text) for part in msg.content if part.type == "text")
        has_images = any(part.type == "image" for part in msg.content)
        tool_content: list[dict[str, Any]] = [
            {"type": "text", "text": build_tool_result_text(text_result, has_images, supports_images, msg.is_error)}
        ]
        if supports_images:
            for part in msg.content:
                if part.type == "image":
                    tool_content.append({"type": "image_url", "imageUrl": f"data:{part.mime_type};base64,{part.data}"})
        result.append(
            {
                "role": "tool",
                "toolCallId": msg.tool_call_id,
                "name": msg.tool_name,
                "content": tool_content,
            }
        )

    return result


def build_tool_result_text(text: str, has_images: bool, supports_images: bool, is_error: bool) -> str:
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""

    if trimmed:
        image_suffix = (
            "\n[tool image omitted: model does not support images]" if has_images and not supports_images else ""
        )
        return f"{error_prefix}{trimmed}{image_suffix}"

    if has_images:
        if supports_images:
            return "[tool error] (see attached image)" if is_error else "(see attached image)"
        return (
            "[tool error] (image omitted: model does not support images)"
            if is_error
            else "(image omitted: model does not support images)"
        )

    return "[tool error] (no tool output)" if is_error else "(no tool output)"


def _uses_reasoning_effort(model: Model) -> bool:
    return model.id in ("mistral-small-2603", "mistral-small-latest", "mistral-medium-3.5")


def _uses_prompt_mode_reasoning(model: Model) -> bool:
    return model.reasoning and not _uses_reasoning_effort(model)


def _map_reasoning_effort(model: Model, level: str) -> str:
    mapping = dict(model.thinking_level_map) if model.thinking_level_map is not None else {}
    mapped = mapping.get(level)
    return mapped if mapped is not None else "high"


def map_tool_choice(choice: Any) -> Any:
    if not choice:
        return None
    if choice in ("auto", "none", "any", "required"):
        return choice
    return {"type": "function", "function": {"name": choice["function"]["name"]}}


def map_chat_stop_reason(reason: str | None) -> tuple[StopReason, str | None]:
    if reason is None:
        return "stop", None
    match reason:
        case "stop":
            return "stop", None
        case "length" | "model_length":
            return "length", None
        case "tool_calls":
            return "toolUse", None
        case "error":
            return "error", "Provider stopped with: error"
        case _:
            return "error", f"Provider stopped with: {reason}"
