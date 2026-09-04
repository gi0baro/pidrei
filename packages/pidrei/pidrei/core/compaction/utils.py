"""Mirror of pi coding-agent src/core/compaction/utils.ts.

Shared utilities for compaction and branch summarization. One deviation:
tool-call arguments are serialized through `safe_json_stringify` (pi's
`JSON.stringify` never throws on plain JS values; Python's `json.dumps` can),
so an unserializable argument renders as `[unserializable]` instead of failing
the whole summary.
"""

import json
from dataclasses import dataclass, field
from typing import Any

from pidrei_agent.types import AgentMessage
from pidrei_ai.types import Message
from pidrei_ai.utils.text import content_text


# ---------------------------------------------------------------------------
# File Operation Tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FileOperations:
    read: set[str] = field(default_factory=set)
    written: set[str] = field(default_factory=set)
    edited: set[str] = field(default_factory=set)


def create_file_ops() -> FileOperations:
    return FileOperations()


def extract_file_ops_from_message(message: AgentMessage, file_ops: FileOperations) -> None:
    """Extract file operations from tool calls in an assistant message."""
    if getattr(message, "role", None) != "assistant":
        return
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return

    for block in content:
        if getattr(block, "type", None) != "toolCall":
            continue
        args = getattr(block, "arguments", None)
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not isinstance(path, str) or not path:
            continue
        name = getattr(block, "name", None)
        if name == "read":
            file_ops.read.add(path)
        elif name == "write":
            file_ops.written.add(path)
        elif name == "edit":
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> tuple[list[str], list[str]]:
    """Compute final file lists from file operations.
    Returns (read_files: files only read, not modified) and (modified_files)."""
    modified = file_ops.edited | file_ops.written
    read_only = sorted(path for path in file_ops.read if path not in modified)
    return read_only, sorted(modified)


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Format file operations as XML tags for summary."""
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Message Serialization
# ---------------------------------------------------------------------------

# Maximum characters for a tool result in serialized summaries.
TOOL_RESULT_MAX_CHARS = 2000


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError, ValueError:
        return "[unserializable]"


def _truncate_for_summary(text: str, max_chars: int) -> str:
    """Truncate text to a maximum character length for summarization.
    Keeps the beginning and appends a truncation marker."""
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def serialize_conversation(messages: list[Message]) -> str:
    """Serialize LLM messages to text for summarization.
    This prevents the model from treating it as a conversation to continue.
    Call convert_to_llm() first to handle custom message types.

    Tool results are truncated to keep the summarization request within
    reasonable token budgets. Full content is not needed for summarization."""
    parts: list[str] = []

    for msg in messages:
        role = getattr(msg, "role", None)
        if role == "user":
            content = content_text(msg.content, "")
            if content:
                parts.append(f"[User]: {content}")
        elif role == "assistant":
            thinking_parts: list[str] = []
            tool_calls: list[str] = []

            for block in msg.content:
                if block.type == "thinking":
                    thinking_parts.append(block.thinking)
                elif block.type == "toolCall":
                    args_str = ", ".join(f"{k}={safe_json_stringify(v)}" for k, v in block.arguments.items())
                    tool_calls.append(f"{block.name}({args_str})")

            if thinking_parts:
                parts.append("[Assistant thinking]: " + "\n".join(thinking_parts))
            if any(block.type == "text" for block in msg.content):
                parts.append(f"[Assistant]: {content_text(msg.content)}")
            if tool_calls:
                parts.append("[Assistant tool calls]: " + "; ".join(tool_calls))
        elif role == "toolResult":
            content = content_text(msg.content, "")
            if content:
                parts.append(f"[Tool result]: {_truncate_for_summary(content, TOOL_RESULT_MAX_CHARS)}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Summarization System Prompt
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between a user and an "
    "AI assistant, then produce a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. "
    "ONLY output the structured summary."
)
