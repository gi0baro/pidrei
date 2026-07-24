"""Compaction shared helpers (port of pi `harness/compaction/utils.ts`)."""

import json
from dataclasses import dataclass, field
from typing import Any

from pidrei_ai.types import Message
from pidrei_ai.utils.text import content_text

from ...types import AgentMessage


@dataclass(slots=True)
class FileOperations:
    """File paths touched by a session branch or compaction range."""

    # Files read but not necessarily modified.
    read: set[str] = field(default_factory=set)
    # Files written by full-file write operations.
    written: set[str] = field(default_factory=set)
    # Files modified by edit operations.
    edited: set[str] = field(default_factory=set)


def create_file_ops() -> FileOperations:
    """Create an empty file-operation accumulator."""
    return FileOperations()


def extract_file_ops_from_message(message: AgentMessage, file_ops: FileOperations) -> None:
    """Add file operations from assistant tool calls to an accumulator."""
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
    """Compute sorted (read-only, modified) file lists from accumulated operations."""
    modified = file_ops.edited | file_ops.written
    read_only = sorted(path for path in file_ops.read if path not in modified)
    return read_only, sorted(modified)


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Format file lists as summary metadata tags."""
    sections: list[str] = []
    if read_files:
        sections.append("<read-files>\n" + "\n".join(read_files) + "\n</read-files>")
    if modified_files:
        sections.append("<modified-files>\n" + "\n".join(modified_files) + "\n</modified-files>")
    if not sections:
        return ""
    return "\n\n" + "\n\n".join(sections)


TOOL_RESULT_MAX_CHARS = 2000


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError, ValueError:
        return "[unserializable]"


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def serialize_conversation(messages: list[Message]) -> str:
    """Serialize LLM messages to plain text for summarization prompts."""
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
