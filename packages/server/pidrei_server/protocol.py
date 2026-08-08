"""pi-ai → protocol bridge (port of pi server `protocol.ts`).

pi enforces the mapping with compile-time `Assert<ExactKeys<...>>` type
checks; Python can't, so the same enumerations run as import-time dataclass
field assertions — a pidrei_ai field addition fails here just as a pi-ai
field addition fails compilation upstream. Provider replay metadata,
diagnostics, cache-write retention splits, model transport settings, pricing
tiers, and deferred-tool availability remain intentionally server-side.

Python-flavoured edges of the lossy `sanitize_protocol_details`: non-finite
floats render as Python's `str()` (`nan`/`inf`), datetimes as `isoformat()`,
out-of-safe-range ints (the bigint analogue) as decimal strings, and
callables are dropped like JS functions.
"""

import dataclasses
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, get_args

from pidrei_ai.registry import get_supported_thinking_levels
from pidrei_ai.types import (
    AssistantMessage,
    ImageContent,
    Model,
    ModelCost,
    ModelThinkingLevel,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage as AiUsage,
    UsageCost,
    UserMessage,
)
from pidrei_protocol import (
    JsonValue,
    ModelMetadata,
    ThinkingLevel,
    TranscriptItem,
    Usage,
)


_MAX_SAFE_INTEGER = 2**53 - 1


def _assert_exact_fields(cls: type, expected: set[str]) -> None:
    actual = {field.name for field in dataclasses.fields(cls)}
    if actual != expected:
        raise AssertionError(
            f"{cls.__name__} fields drifted from the protocol bridge: "
            f"unmapped={sorted(actual - expected)} missing={sorted(expected - actual)}"
        )


def _literal_values(alias: Any) -> set[str]:
    return set(get_args(alias.__value__))


if _literal_values(ModelThinkingLevel) != _literal_values(ThinkingLevel):
    raise AssertionError("pidrei_ai thinking levels no longer match protocol thinking levels")
# Enumerate mapped and intentionally omitted pidrei_ai fields so additions fail import here.
_assert_exact_fields(TextContent, {"type", "text", "text_signature"})
_assert_exact_fields(ThinkingContent, {"type", "thinking", "thinking_signature", "redacted"})
_assert_exact_fields(ImageContent, {"type", "data", "mime_type"})
_assert_exact_fields(ToolCall, {"type", "id", "name", "arguments", "thought_signature"})
_assert_exact_fields(
    AiUsage, {"input", "output", "cache_read", "cache_write", "cache_write_1h", "reasoning", "total_tokens", "cost"}
)
_assert_exact_fields(UsageCost, {"input", "output", "cache_read", "cache_write", "total"})
# Provider replay metadata, diagnostics, cache-write retention splits, model
# transport settings, model sampling defaults, pricing tiers, and deferred-tool
# availability remain intentionally server-side.
_assert_exact_fields(
    Model,
    {
        "id",
        "name",
        "api",
        "provider",
        "base_url",
        "reasoning",
        "thinking_level_map",
        "input",
        "cost",
        "context_window",
        "max_tokens",
        "sampling_params",
        "headers",
        "compat",
    },
)
_assert_exact_fields(ModelCost, {"input", "output", "cache_read", "cache_write", "tiers"})
_assert_exact_fields(UserMessage, {"role", "content", "timestamp"})
_assert_exact_fields(
    AssistantMessage,
    {
        "role",
        "content",
        "api",
        "provider",
        "model",
        "response_model",
        "response_id",
        "diagnostics",
        "usage",
        "stop_reason",
        "deferred",
        "error_message",
        "raw_stop_reason",
        "timestamp",
    },
)
_assert_exact_fields(
    ToolResultMessage,
    {
        "role",
        "tool_call_id",
        "tool_name",
        "content",
        "details",
        "usage",
        "added_tool_names",
        "is_error",
        "timestamp",
    },
)


@dataclass(slots=True, frozen=True)
class AssistantTranscriptOptions:
    id: str


@dataclass(slots=True, frozen=True)
class UserTranscriptOptions:
    id: str


@dataclass(slots=True, frozen=True)
class ToolTranscriptOptions:
    id: str
    call: ToolCall


def _non_negative_integer(value: float | None) -> int | None:
    if value is None or isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return max(0, math.floor(value))


def _non_negative_number(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return 0
    return max(0, value)


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or len(value) == 0:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _timestamp(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0 or value > _MAX_SAFE_INTEGER:
        raise TypeError("Protocol timestamps must be non-negative integers")
    return value


def to_protocol_json_value(value: object, seen: set[int] | None = None) -> JsonValue:
    """Validate and copy a value from an execution boundary into the protocol's JSON-compatible subset."""
    if seen is None:
        seen = set()
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Protocol JSON numbers must be finite")
        return value
    if isinstance(value, dict) and type(value) is not dict:
        raise TypeError("Protocol JSON objects must be plain objects")
    if not isinstance(value, list | dict):
        raise TypeError(f"Unsupported protocol JSON value: {type(value).__name__}")
    if id(value) in seen:
        raise TypeError("Protocol JSON values must not contain circular references")
    seen.add(id(value))
    try:
        if isinstance(value, list):
            return [to_protocol_json_value(entry, seen) for entry in value]
        return {key: to_protocol_json_value(entry, seen) for key, entry in value.items()}
    finally:
        seen.discard(id(value))


_OMITTED = object()


def sanitize_protocol_details(value: object, seen: set[int] | None = None) -> JsonValue | None:
    """Lossily sanitize diagnostic tool details that must not affect execution semantics."""
    result = _sanitize_details(value, seen if seen is not None else set())
    return None if result is _OMITTED else result


def _sanitize_details(value: object, seen: set[int]) -> Any:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int):
        return value if abs(value) <= _MAX_SAFE_INTEGER else str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if callable(value):
        return _OMITTED
    if isinstance(value, datetime):
        return value.isoformat()
    if not isinstance(value, list | dict):
        return str(value)
    if id(value) in seen:
        return "[Circular]"
    seen.add(id(value))
    try:
        if isinstance(value, list):
            return [None if (entry := _sanitize_details(item, seen)) is _OMITTED else entry for item in value]
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = _sanitize_details(item, seen)
            if normalized is not _OMITTED:
                result[str(key)] = normalized
        return result
    finally:
        seen.discard(id(value))


def to_protocol_usage(usage: AiUsage | None) -> Usage | None:
    if usage is None:
        return None
    reasoning = _non_negative_integer(usage.reasoning)
    result: Usage = {
        "input": _non_negative_integer(usage.input) or 0,
        "output": _non_negative_integer(usage.output) or 0,
        "cacheRead": _non_negative_integer(usage.cache_read) or 0,
        "cacheWrite": _non_negative_integer(usage.cache_write) or 0,
        **({} if reasoning is None else {"reasoning": reasoning}),
        "totalTokens": _non_negative_integer(usage.total_tokens) or 0,
        "cost": {
            "input": _non_negative_number(usage.cost.input),
            "output": _non_negative_number(usage.cost.output),
            "cacheRead": _non_negative_number(usage.cost.cache_read),
            "cacheWrite": _non_negative_number(usage.cost.cache_write),
            "total": _non_negative_number(usage.cost.total),
        },
    }
    return result


def to_protocol_model_metadata(model: Model, authenticated: bool) -> ModelMetadata:
    result: ModelMetadata = {
        "provider": _identifier(model.provider, "Model provider"),
        "id": _identifier(model.id, "Model id"),
        "name": _identifier(model.name, "Model name"),
        "api": _identifier(model.api, "Model API"),
        "reasoning": model.reasoning,
        "input": list(model.input),
        "contextWindow": max(1, math.floor(model.context_window)),
        "maxTokens": max(1, math.floor(model.max_tokens)),
        "cost": {
            "input": _non_negative_number(model.cost.input),
            "output": _non_negative_number(model.cost.output),
            "cacheRead": _non_negative_number(model.cost.cache_read),
            "cacheWrite": _non_negative_number(model.cost.cache_write),
        },
        "supportedThinkingLevels": get_supported_thinking_levels(model),
        "authenticated": authenticated,
    }
    return result


def _to_protocol_user_content(content: str | list[TextContent | ImageContent]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextContent):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            parts.append({"type": "image", "data": part.data, "mimeType": part.mime_type})
        else:
            raise TypeError(f"Unsupported user content: {type(part).__name__}")
    return parts


def to_protocol_user_message(message: UserMessage, options: UserTranscriptOptions) -> TranscriptItem:
    result: TranscriptItem = {
        "id": _identifier(options.id, "Transcript item id"),
        "role": "user",
        "content": _to_protocol_user_content(message.content),
        "timestamp": _timestamp(message.timestamp),
    }
    return result


def _to_protocol_assistant_content(message: AssistantMessage) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in message.content:
        if isinstance(part, TextContent):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ThinkingContent):
            # pidrei_ai `redacted` is a plain bool (never absent), so the
            # schema-optional key is always present.
            parts.append({"type": "thinking", "thinking": part.thinking, "redacted": part.redacted})
        elif isinstance(part, ToolCall):
            parts.append(
                {
                    "type": "toolCall",
                    "toolCallId": _identifier(part.id, "Tool call id"),
                    "toolName": _identifier(part.name, "Tool call name"),
                    "input": to_protocol_json_value(part.arguments),
                }
            )
        else:
            raise TypeError(f"Unsupported assistant content: {type(part).__name__}")
    return parts


def to_protocol_assistant_message(message: AssistantMessage, options: AssistantTranscriptOptions) -> TranscriptItem:
    usage = to_protocol_usage(message.usage)
    common: dict[str, Any] = {
        "id": _identifier(options.id, "Transcript item id"),
        "role": "assistant",
        "content": _to_protocol_assistant_content(message),
        "model": {
            "provider": _identifier(message.provider, "Assistant provider"),
            "id": _identifier(message.model, "Assistant model"),
        },
        **(
            {}
            if message.response_model is None
            else {"responseModel": _identifier(message.response_model, "Assistant response model")}
        ),
        **({"usage": usage} if usage is not None else {}),
        "timestamp": _timestamp(message.timestamp),
    }
    match message.stop_reason:
        case "pending":
            return {**common, "status": "streaming"}
        case "stop" | "length" | "toolUse":
            return {**common, "status": "complete", "stopReason": message.stop_reason}
        case "deferred":
            raise TypeError("Deferred assistant messages are not supported by protocol v1")
        case "error":
            if message.error_message is not None and len(message.error_message) == 0:
                raise TypeError("Assistant error messages must not be empty")
            return {
                **common,
                "status": "error",
                "stopReason": "error",
                **({} if message.error_message is None else {"errorMessage": message.error_message}),
            }
        case "aborted":
            return {
                **common,
                "status": "aborted",
                "stopReason": "aborted",
                **({} if message.error_message is None else {"errorMessage": message.error_message}),
            }
        case _:
            raise TypeError(f"Unsupported assistant stop reason: {message.stop_reason}")


def _to_protocol_tool_content(content: list[TextContent | ImageContent]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for part in content:
        if isinstance(part, TextContent):
            parts.append({"type": "text", "text": part.text})
        elif isinstance(part, ImageContent):
            parts.append({"type": "image", "data": part.data, "mimeType": part.mime_type})
        else:
            raise TypeError(f"Unsupported tool content: {type(part).__name__}")
    return parts


def to_protocol_tool_result_message(message: ToolResultMessage, options: ToolTranscriptOptions) -> TranscriptItem:
    call_id = _identifier(options.call.id, "Tool call id")
    call_name = _identifier(options.call.name, "Tool call name")
    if _identifier(message.tool_call_id, "Tool result call id") != call_id:
        raise TypeError(f"Tool result {message.tool_call_id} does not match tool call {call_id}")
    if _identifier(message.tool_name, "Tool result name") != call_name:
        raise TypeError(f"Tool result {message.tool_name} does not match tool call {call_name}")
    details = sanitize_protocol_details(message.details) if message.details is not None else None
    usage = to_protocol_usage(message.usage)
    common: dict[str, Any] = {
        "id": _identifier(options.id, "Transcript item id"),
        "role": "tool",
        "toolCallId": call_id,
        "toolName": call_name,
        "input": to_protocol_json_value(options.call.arguments),
        "content": _to_protocol_tool_content(message.content),
        **({} if details is None else {"details": details}),
        **({"usage": usage} if usage is not None else {}),
        "timestamp": _timestamp(message.timestamp),
    }
    if message.is_error:
        return {**common, "status": "error", "isError": True}
    return {**common, "status": "complete", "isError": False}
