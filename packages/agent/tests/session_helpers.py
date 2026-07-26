"""Shared session test helpers (mirror of pi agent/test/harness/session-test-utils.ts)."""

import tempfile
import time

from pidrei_ai.types import AssistantMessage, TextContent, Usage, UserMessage


def create_user_message(text: str) -> UserMessage:
    return UserMessage(content=[TextContent(text=text)], timestamp=int(time.time() * 1000))


def create_assistant_message(text: str) -> AssistantMessage:
    return AssistantMessage(
        content=[TextContent(text=text)],
        api="anthropic-messages",
        provider="anthropic",
        model="claude-sonnet-4-5",
        usage=Usage(),
        stop_reason="stop",
        timestamp=int(time.time() * 1000),
    )


def create_temp_dir() -> str:
    return tempfile.mkdtemp(prefix="pidrei-agent-session-")
