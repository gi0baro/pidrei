"""Shared helpers for session/agent-session test mirrors (pi test/utilities.ts)."""

import json
import time
from typing import Any

from pidrei_ai.types import AssistantMessage, TextContent, ToolResultMessage, Usage, UsageCost, UserMessage


def now_ms() -> int:
    return int(time.time() * 1000)


def user_msg(text: str) -> UserMessage:
    return UserMessage(content=text, timestamp=now_ms())


def assistant_msg(text: str, **overrides: Any) -> AssistantMessage:
    defaults: dict[str, Any] = {
        "content": [TextContent(text=text)],
        "api": "anthropic-messages",
        "provider": "anthropic",
        "model": "test",
        "usage": Usage(input=1, output=1, cache_read=0, cache_write=0, total_tokens=2, cost=UsageCost()),
        "stop_reason": "stop",
        "timestamp": now_ms(),
    }
    defaults.update(overrides)
    return AssistantMessage(**defaults)


def read_session_file_roles(file: str) -> list[str]:
    """One label per record of a session JSONL file: the message role for message
    entries, otherwise the entry type (e.g. "session", "model_change")."""
    with open(file, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle.read().strip().split("\n")]
    return [(record.get("message") or {}).get("role") or record["type"] for record in records]


def tool_result_msg(text: str, *, usage: Usage | None = None, tool_call_id: str = "call-1") -> ToolResultMessage:
    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="nested-model",
        content=[TextContent(text=text)],
        is_error=False,
        usage=usage,
        timestamp=now_ms(),
    )


def make_usage() -> Usage:
    return Usage(
        input=10,
        output=20,
        cache_read=30,
        cache_write=40,
        total_tokens=100,
        cost=UsageCost(input=0.1, output=0.2, cache_read=0.3, cache_write=0.4, total=1),
    )
