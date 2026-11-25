"""Mirror of pi coding-agent src/core/messages.ts: the coding-agent's custom
message types and their conversion to LLM messages."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pidrei_agent.types import AgentMessage
from pidrei_ai.types import ImageContent, Message, TextContent, UserMessage


COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)

COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"

BRANCH_SUMMARY_SUFFIX = "</summary>"


@dataclass(slots=True, kw_only=True)
class BashExecutionMessage:
    command: str
    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    timestamp: int
    full_output_path: str | None = None
    # Why the full output could not be saved, when it was truncated.
    full_output_error: str | None = None
    exclude_from_context: bool | None = None
    role: Literal["bashExecution"] = "bashExecution"


@dataclass(slots=True, kw_only=True)
class CustomMessage:
    custom_type: str
    content: Any  # str | list[TextContent | ImageContent]
    display: bool
    timestamp: int
    details: Any = None
    role: Literal["custom"] = "custom"


@dataclass(slots=True, kw_only=True)
class BranchSummaryMessage:
    summary: str
    # Root branch summaries have no source entry (pi: `string | null`).
    from_id: str | None
    timestamp: int
    role: Literal["branchSummary"] = "branchSummary"


@dataclass(slots=True, kw_only=True)
class CompactionSummaryMessage:
    summary: str
    tokens_before: int
    timestamp: int
    role: Literal["compactionSummary"] = "compactionSummary"


def _to_epoch_ms(timestamp: str | int) -> int:
    if isinstance(timestamp, int):
        return timestamp
    return int(datetime.fromisoformat(timestamp).timestamp() * 1000)


def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    text = f"Ran `{msg.command}`\n"
    if msg.output:
        text += f"```\n{msg.output}\n```"
    else:
        text += "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated:
        note = bash_truncation_note(full_output_path=msg.full_output_path, full_output_error=msg.full_output_error)
        text += f"\n\n[{note}]"
    return text


def bash_truncation_note(*, full_output_path: str | None, full_output_error: str | None) -> str:
    """What a truncated bash execution says about its full output: where it is, or why it could not be saved."""
    if full_output_path:
        return f"Output truncated. Full output: {full_output_path}"
    if full_output_error:
        return f"Output truncated. Could not save the full output: {full_output_error}"
    return "Output truncated"


def create_branch_summary_message(summary: str, from_id: str | None, timestamp: str | int) -> BranchSummaryMessage:
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=_to_epoch_ms(timestamp))


def create_compaction_summary_message(
    summary: str, tokens_before: int, timestamp: str | int
) -> CompactionSummaryMessage:
    return CompactionSummaryMessage(summary=summary, tokens_before=tokens_before, timestamp=_to_epoch_ms(timestamp))


def create_custom_message(
    custom_type: str,
    content: Any,
    display: bool,
    details: Any,
    timestamp: str | int,
) -> CustomMessage:
    return CustomMessage(
        custom_type=custom_type,
        content=content,
        display=display,
        details=details,
        timestamp=_to_epoch_ms(timestamp),
    )


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    converted: list[Message] = []
    for message in messages:
        role = getattr(message, "role", None)
        if role == "bashExecution":
            if message.exclude_from_context:
                continue
            converted.append(
                UserMessage(content=[TextContent(text=bash_execution_to_text(message))], timestamp=message.timestamp)
            )
        elif role == "custom":
            content: list[TextContent | ImageContent] = (
                [TextContent(text=message.content)] if isinstance(message.content, str) else message.content
            )
            converted.append(UserMessage(content=content, timestamp=message.timestamp))
        elif role == "branchSummary":
            converted.append(
                UserMessage(
                    content=[TextContent(text=BRANCH_SUMMARY_PREFIX + message.summary + BRANCH_SUMMARY_SUFFIX)],
                    timestamp=message.timestamp,
                )
            )
        elif role == "compactionSummary":
            converted.append(
                UserMessage(
                    content=[TextContent(text=COMPACTION_SUMMARY_PREFIX + message.summary + COMPACTION_SUMMARY_SUFFIX)],
                    timestamp=message.timestamp,
                )
            )
        elif role in ("system", "user", "assistant", "toolResult"):
            converted.append(message)
    return converted
