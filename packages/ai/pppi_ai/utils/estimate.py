"""Port of pi's token estimation heuristics (packages/ai/src/utils/estimate.ts).

Estimates use ~4 chars/token and a flat per-image cost; when a recent
assistant message carries provider-reported usage that still describes the
current prefix, that usage is trusted and only trailing messages are estimated.
"""

import json
from dataclasses import dataclass
from typing import Any

from pppi_ai.types import Context, Message, Tool, Usage


CHARS_PER_TOKEN = 4
ESTIMATED_IMAGE_CHARS = 4800


@dataclass(slots=True)
class ContextUsageEstimate:
    # Estimated total context tokens.
    tokens: int
    # Tokens reported by the most recent applicable assistant usage block.
    usage_tokens: int
    # Estimated tokens after the most recent applicable assistant usage block.
    trailing_tokens: int
    # Index of the applicable message that provided usage, or None when none exists.
    last_usage_index: int | None


def calculate_context_tokens(usage: Usage) -> int:
    return usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write


def _safe_json_stringify(value: Any) -> str:
    try:
        # Compact separators and raw non-ASCII mirror JSON.stringify's output size.
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "[unserializable]"


def _tool_json_shape(tool: Tool) -> dict[str, Any]:
    # The JSON.stringify shape of a pi Tool object (undefined keys omitted).
    shape: dict[str, Any] = {"name": tool.name, "description": tool.description, "parameters": tool.parameters}
    if tool.constrained_sampling is not None:
        if tool.constrained_sampling is False:
            shape["constrainedSampling"] = False
        elif tool.constrained_sampling.type == "json_schema":
            shape["constrainedSampling"] = {"type": "json_schema", "strict": tool.constrained_sampling.strict}
        else:
            shape["constrainedSampling"] = {"type": "grammar", "variants": dict(tool.constrained_sampling.variants)}
    return shape


def _estimate_text_and_image_content_chars(content: str | list) -> int:
    if isinstance(content, str):
        return len(content)

    chars = 0
    for block in content:
        chars += len(block.text) if block.type == "text" else ESTIMATED_IMAGE_CHARS
    return chars


def estimate_text_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def estimate_text_and_image_content_tokens(content: str | list) -> int:
    return -(-_estimate_text_and_image_content_chars(content) // CHARS_PER_TOKEN)


def estimate_message_tokens(message: Message) -> int:
    if message.role in ("user", "toolResult"):
        return estimate_text_and_image_content_tokens(message.content)

    chars = 0
    for block in message.content:
        if block.type == "text":
            chars += len(block.text)
        elif block.type == "thinking":
            chars += len(block.thinking)
        else:
            chars += len(block.name) + len(_safe_json_stringify(block.arguments))
    return -(-chars // CHARS_PER_TOKEN)


def _get_last_assistant_usage_info(messages: list[Message]) -> tuple[Usage, int] | None:
    latest_prefix_timestamp = float("-inf")
    usage_info: tuple[Usage, int] | None = None

    for index, message in enumerate(messages):
        if message.role == "assistant":
            # A newer prefix message inserted after this response (e.g. a compaction
            # summary) means its usage cannot describe the current prefix.
            usage_applies_to_prefix = message.timestamp >= latest_prefix_timestamp
            if (
                usage_applies_to_prefix
                and message.stop_reason not in ("aborted", "error")
                and calculate_context_tokens(message.usage) > 0
            ):
                usage_info = (message.usage, index)
        latest_prefix_timestamp = max(latest_prefix_timestamp, message.timestamp)

    return usage_info


def _estimate_messages(messages: list[Message]) -> ContextUsageEstimate:
    usage_info = _get_last_assistant_usage_info(messages)
    if usage_info is not None:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing_tokens = sum(estimate_message_tokens(message) for message in messages[index + 1 :])
        return ContextUsageEstimate(
            tokens=usage_tokens + trailing_tokens,
            usage_tokens=usage_tokens,
            trailing_tokens=trailing_tokens,
            last_usage_index=index,
        )

    tokens = sum(estimate_message_tokens(message) for message in messages)
    return ContextUsageEstimate(tokens=tokens, usage_tokens=0, trailing_tokens=tokens, last_usage_index=None)


def _estimate_tools_tokens(tools: list[Tool] | None) -> int:
    if not tools:
        return 0
    return estimate_text_tokens(_safe_json_stringify([_tool_json_shape(tool) for tool in tools]))


def estimate_context_tokens(context: Context | list[Message]) -> ContextUsageEstimate:
    if isinstance(context, list):
        return _estimate_messages(context)

    estimate = _estimate_messages(context.messages)
    if estimate.last_usage_index is not None:
        added_names = {
            name
            for message in context.messages[estimate.last_usage_index + 1 :]
            if message.role == "toolResult"
            for name in (message.added_tool_names or [])
        }
        added_tools = [tool for tool in (context.tools or []) if tool.name in added_names]
        added_tool_tokens = _estimate_tools_tokens(added_tools or None)
        return ContextUsageEstimate(
            tokens=estimate.tokens + added_tool_tokens,
            usage_tokens=estimate.usage_tokens,
            trailing_tokens=estimate.trailing_tokens + added_tool_tokens,
            last_usage_index=estimate.last_usage_index,
        )

    prefix_tokens = (
        estimate_text_tokens(context.system_prompt) if context.system_prompt else 0
    ) + _estimate_tools_tokens(context.tools)

    return ContextUsageEstimate(
        tokens=estimate.tokens + prefix_tokens,
        usage_tokens=estimate.usage_tokens,
        trailing_tokens=estimate.trailing_tokens + prefix_tokens,
        last_usage_index=estimate.last_usage_index,
    )
