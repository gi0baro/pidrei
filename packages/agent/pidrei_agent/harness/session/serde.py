"""Message-level (de)serialization to pi's camelCase wire shape.

pi persists raw JS objects with `JSON.stringify`, so wire payloads carry
camelCase keys and omit `undefined` fields. This module maps the port's
snake_case message/usage dataclasses to and from that exact shape with
explicit per-type field maps — mechanical key conversion would corrupt
user-owned dicts (`arguments`, `details`, `data`, `metadata`), which pass
through with their keys untouched. `details` values that are port-side
*dataclasses* (pi's are plain JSON objects) are converted to camelCase dicts
on the way out, or `json.dumps` would refuse them; they parse back as dicts,
which every consumer already accepts (renderers probe dict-or-attribute).

Unknown message roles round-trip as their raw parsed dicts, exactly like pi's
blind cast. Both the v4 session codec (`session/jsonl/codec.py`) and the
coding-agent's own session store build on these converters.
"""

import dataclasses
from typing import Any

from pidrei_ai.types import (
    AssistantMessage,
    AssistantMessageDiagnostic,
    DeferredHandle,
    DiagnosticErrorInfo,
    ImageContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UsageCost,
    UserMessage,
)

from ..messages import BashExecutionMessage, BranchSummaryMessage, CompactionSummaryMessage, CustomMessage


def _put(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        target[key] = value


# --- usage --------------------------------------------------------------------


def serialize_usage(usage: Any) -> Any:
    if usage is None or isinstance(usage, dict):
        return usage
    data: dict[str, Any] = {
        "input": usage.input,
        "output": usage.output,
        "cacheRead": usage.cache_read,
        "cacheWrite": usage.cache_write,
    }
    _put(data, "cacheWrite1h", usage.cache_write_1h)
    _put(data, "reasoning", usage.reasoning)
    data["totalTokens"] = usage.total_tokens
    cost = usage.cost
    data["cost"] = {
        "input": cost.input,
        "output": cost.output,
        "cacheRead": cost.cache_read,
        "cacheWrite": cost.cache_write,
        "total": cost.total,
    }
    return data


def parse_usage(data: Any) -> Usage | None:
    if not isinstance(data, dict):
        return None
    cost = data.get("cost") or {}
    return Usage(
        input=data.get("input", 0),
        output=data.get("output", 0),
        cache_read=data.get("cacheRead", 0),
        cache_write=data.get("cacheWrite", 0),
        cache_write_1h=data.get("cacheWrite1h"),
        reasoning=data.get("reasoning"),
        total_tokens=data.get("totalTokens", 0),
        cost=UsageCost(
            input=cost.get("input", 0.0),
            output=cost.get("output", 0.0),
            cache_read=cost.get("cacheRead", 0.0),
            cache_write=cost.get("cacheWrite", 0.0),
            total=cost.get("total", 0.0),
        ),
    )


# --- content blocks ------------------------------------------------------------


def serialize_content_block(block: Any) -> Any:
    if isinstance(block, dict):
        return block
    block_type = getattr(block, "type", None)
    if block_type == "text":
        data = {"type": "text", "text": block.text}
        _put(data, "textSignature", block.text_signature)
        return data
    if block_type == "thinking":
        data = {"type": "thinking", "thinking": block.thinking}
        _put(data, "thinkingSignature", block.thinking_signature)
        if block.redacted:
            data["redacted"] = True
        return data
    if block_type == "toolCall":
        data = {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments}
        _put(data, "thoughtSignature", block.thought_signature)
        return data
    if block_type == "image":
        return {"type": "image", "data": block.data, "mimeType": block.mime_type}
    return block


def parse_content_block(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    block_type = data.get("type")
    if block_type == "text":
        return TextContent(text=data.get("text", ""), text_signature=data.get("textSignature"))
    if block_type == "thinking":
        return ThinkingContent(
            thinking=data.get("thinking", ""),
            thinking_signature=data.get("thinkingSignature"),
            redacted=bool(data.get("redacted", False)),
        )
    if block_type == "toolCall":
        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments") or {},
            thought_signature=data.get("thoughtSignature"),
        )
    if block_type == "image":
        return ImageContent(data=data.get("data", ""), mime_type=data.get("mimeType", ""))
    return data


def serialize_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [serialize_content_block(block) for block in content]
    return content


def _parse_content(content: Any) -> Any:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return [parse_content_block(block) for block in content]
    return content


# --- messages -------------------------------------------------------------------


def serialize_message(message: Any) -> Any:
    if message is None or isinstance(message, dict):
        return message
    role = getattr(message, "role", None)
    if role == "user":
        return {"role": "user", "content": serialize_content(message.content), "timestamp": message.timestamp}
    if role == "assistant":
        data: dict[str, Any] = {
            "role": "assistant",
            "content": serialize_content(message.content),
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "usage": serialize_usage(message.usage),
            "stopReason": message.stop_reason,
            "timestamp": message.timestamp,
        }
        _put(data, "responseModel", message.response_model)
        _put(data, "responseId", message.response_id)
        if message.diagnostics is not None:
            data["diagnostics"] = [_serialize_diagnostic(diagnostic) for diagnostic in message.diagnostics]
        _put(data, "errorMessage", message.error_message)
        _put(data, "rawStopReason", message.raw_stop_reason)
        _put(data, "deferred", _serialize_deferred_handle(message.deferred))
        return data
    if role == "toolResult":
        data = {
            "role": "toolResult",
            "toolCallId": message.tool_call_id,
            "toolName": message.tool_name,
            "content": serialize_content(message.content),
        }
        _put(data, "details", to_wire_value(message.details))
        _put(data, "usage", serialize_usage(message.usage))
        if message.added_tool_names:
            data["addedToolNames"] = message.added_tool_names
        data["isError"] = message.is_error
        data["timestamp"] = message.timestamp
        return data
    if role == "custom":
        data = {"role": "custom", "customType": message.custom_type, "content": serialize_content(message.content)}
        data["display"] = message.display
        _put(data, "details", to_wire_value(message.details))
        data["timestamp"] = message.timestamp
        return data
    if role == "bashExecution":
        data = {"role": "bashExecution", "command": message.command, "output": message.output}
        _put(data, "exitCode", message.exit_code)
        data["cancelled"] = message.cancelled
        data["truncated"] = message.truncated
        _put(data, "fullOutputPath", message.full_output_path)
        data["timestamp"] = message.timestamp
        _put(data, "excludeFromContext", message.exclude_from_context)
        return data
    if role == "branchSummary":
        return {
            "role": "branchSummary",
            "summary": message.summary,
            "fromId": message.from_id,
            "timestamp": message.timestamp,
        }
    if role == "compactionSummary":
        return {
            "role": "compactionSummary",
            "summary": message.summary,
            "tokensBefore": message.tokens_before,
            "timestamp": message.timestamp,
        }
    return message


def parse_message(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    role = data.get("role")
    if role == "user":
        return UserMessage(content=_parse_content(data.get("content")), timestamp=data.get("timestamp", 0))
    if role == "assistant":
        return AssistantMessage(
            content=_parse_content(data.get("content")) or [],
            api=data.get("api", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            usage=parse_usage(data.get("usage")) or Usage(),
            stop_reason=data.get("stopReason", "stop"),
            timestamp=data.get("timestamp", 0),
            response_model=data.get("responseModel"),
            response_id=data.get("responseId"),
            diagnostics=(
                [_parse_diagnostic(diagnostic) for diagnostic in data["diagnostics"]]
                if isinstance(data.get("diagnostics"), list)
                else None
            ),
            error_message=data.get("errorMessage"),
            raw_stop_reason=data.get("rawStopReason"),
            deferred=_parse_deferred_handle(data.get("deferred")),
        )
    if role == "toolResult":
        return ToolResultMessage(
            tool_call_id=data.get("toolCallId", ""),
            tool_name=data.get("toolName", ""),
            content=_parse_content(data.get("content")) or [],
            is_error=bool(data.get("isError", False)),
            timestamp=data.get("timestamp", 0),
            details=data.get("details"),
            usage=parse_usage(data.get("usage")),
            added_tool_names=data.get("addedToolNames"),
        )
    if role == "custom":
        return CustomMessage(
            custom_type=data.get("customType", ""),
            content=_parse_content(data.get("content")),
            display=bool(data.get("display", False)),
            details=data.get("details"),
            timestamp=data.get("timestamp", 0),
        )
    if role == "bashExecution":
        return BashExecutionMessage(
            command=data.get("command", ""),
            output=data.get("output", ""),
            exit_code=data.get("exitCode"),
            cancelled=bool(data.get("cancelled", False)),
            truncated=bool(data.get("truncated", False)),
            full_output_path=data.get("fullOutputPath"),
            timestamp=data.get("timestamp", 0),
            exclude_from_context=data.get("excludeFromContext"),
        )
    if role == "branchSummary":
        return BranchSummaryMessage(
            summary=data.get("summary", ""), from_id=data.get("fromId", ""), timestamp=data.get("timestamp", 0)
        )
    if role == "compactionSummary":
        return CompactionSummaryMessage(
            summary=data.get("summary", ""),
            tokens_before=data.get("tokensBefore", 0),
            timestamp=data.get("timestamp", 0),
        )
    return data


def _serialize_deferred_handle(handle: Any) -> Any:
    if handle is None or isinstance(handle, dict):
        return handle
    data: dict[str, Any] = {
        "provider": handle.provider,
        "modelId": handle.model_id,
        "api": handle.api,
        "id": handle.id,
    }
    _put(data, "expiresAt", handle.expires_at)
    _put(data, "pollAfterMs", handle.poll_after_ms)
    _put(data, "data", handle.data)
    return data


def _parse_deferred_handle(data: Any) -> DeferredHandle | None:
    if not isinstance(data, dict):
        return None
    return DeferredHandle(
        provider=data.get("provider", ""),
        model_id=data.get("modelId", ""),
        api=data.get("api", ""),
        id=data.get("id", ""),
        expires_at=data.get("expiresAt"),
        poll_after_ms=data.get("pollAfterMs"),
        data=data.get("data"),
    )


def _serialize_diagnostic(diagnostic: Any) -> Any:
    if isinstance(diagnostic, dict):
        return diagnostic
    data: dict[str, Any] = {"type": diagnostic.type, "timestamp": diagnostic.timestamp}
    if diagnostic.error is not None:
        error: dict[str, Any] = {"message": diagnostic.error.message}
        _put(error, "name", diagnostic.error.name)
        _put(error, "stack", diagnostic.error.stack)
        _put(error, "code", diagnostic.error.code)
        data["error"] = error
    _put(data, "details", diagnostic.details)
    return data


def _parse_diagnostic(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    error = data.get("error")
    return AssistantMessageDiagnostic(
        type=data.get("type", ""),
        timestamp=data.get("timestamp", 0),
        error=(
            DiagnosticErrorInfo(
                message=error.get("message", ""),
                name=error.get("name"),
                stack=error.get("stack"),
                code=error.get("code"),
            )
            if isinstance(error, dict)
            else None
        ),
        details=data.get("details"),
    )


# --- details --------------------------------------------------------------------


def _camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def to_wire_value(value: Any) -> Any:
    """pi's `details` are plain JSON objects; the port's tools produce
    dataclasses. Convert those to pi's wire shape (camelCase keys, None
    dropped, like JSON.stringify over a JS object). Dicts keep their keys —
    they are user- or wire-owned already."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        wire: dict[str, Any] = {}
        for field in dataclasses.fields(value):
            item = getattr(value, field.name)
            if item is None:
                continue
            wire[_camel(field.name)] = to_wire_value(item)
        return wire
    if isinstance(value, dict):
        return {key: to_wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_wire_value(item) for item in value]
    return value


def _serialize_details(details: Any) -> Any:
    read_files = getattr(details, "read_files", None)
    modified_files = getattr(details, "modified_files", None)
    if isinstance(read_files, list) and isinstance(modified_files, list):
        return {"readFiles": read_files, "modifiedFiles": modified_files}
    return to_wire_value(details)
