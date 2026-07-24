"""Port of pi's faux test provider (packages/ai/src/providers/faux.ts).

Partial for now: the message/content helpers used by test suites. The full
`register_faux_provider` streaming provider lands with the models registry
(PLAN.md Phase 1).
"""

import secrets
import time

from pidrei_ai.types import AssistantMessage, StopReason, TextContent, ThinkingContent, ToolCall, Usage


DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"

type FauxContentBlock = TextContent | ThinkingContent | ToolCall


def _random_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(4)}"


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(name: str, arguments: dict, *, id: str | None = None) -> ToolCall:
    return ToolCall(id=id if id is not None else _random_id("tool"), name=name, arguments=arguments)


def _normalize_content(content: str | FauxContentBlock | list[FauxContentBlock]) -> list[FauxContentBlock]:
    if isinstance(content, str):
        return [faux_text(content)]
    return content if isinstance(content, list) else [content]


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: StopReason = "stop",
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=_normalize_content(content),
        api=DEFAULT_API,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL_ID,
        usage=Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp if timestamp is not None else int(time.time() * 1000),
    )
