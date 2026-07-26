"""Compaction (port of pi `harness/compaction/compaction.ts`)."""

import math
import time
from dataclasses import dataclass, replace
from typing import Any

from pidrei_ai.types import Context, Model, SimpleStreamOptions, TextContent, Usage, UsageCost, UserMessage
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.retry import RetryCallbacks, RetryPolicy, retry_assistant_call
from pidrei_ai.utils.text import content_text
from pidrei_ai.utils.uuid import uuidv7

from ...types import AgentMessage
from ..messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from ..session.session import build_session_context
from ..types import CompactionEntry, CompactionError, Result, SessionTreeEntry, err, ok
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    safe_json_stringify,
    serialize_conversation,
)


@dataclass(slots=True)
class CompactionDetails:
    """File-operation details stored on generated compaction entries."""

    # Files read in the compacted history.
    read_files: list[str]
    # Files modified in the compacted history.
    modified_files: list[str]


def _extract_file_operations(
    messages: list[AgentMessage],
    entries: list[SessionTreeEntry],
    prev_compaction_index: int,
) -> FileOperations:
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev_compaction = entries[prev_compaction_index]
        if not prev_compaction.from_hook and prev_compaction.details is not None:
            details = prev_compaction.details
            if isinstance(details, dict):
                # JSONL-deserialized details keep pi's camelCase keys.
                read_files = details.get("readFiles", details.get("read_files"))
                modified_files = details.get("modifiedFiles", details.get("modified_files"))
            else:
                read_files = getattr(details, "read_files", None)
                modified_files = getattr(details, "modified_files", None)
            if isinstance(read_files, list):
                file_ops.read.update(read_files)
            if isinstance(modified_files, list):
                file_ops.edited.update(modified_files)
    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)

    return file_ops


def _get_message_from_entry(entry: SessionTreeEntry) -> AgentMessage | None:
    if entry.type == "message":
        return entry.message
    if entry.type == "custom_message":
        return create_custom_message(entry.custom_type, entry.content, entry.display, entry.details, entry.timestamp)
    if entry.type == "branch_summary":
        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
    if entry.type == "compaction":
        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)
    return None


def _get_message_from_entry_for_compaction(entry: SessionTreeEntry) -> AgentMessage | None:
    if entry.type == "compaction":
        return None
    return _get_message_from_entry(entry)


@dataclass(slots=True)
class CompactionResult:
    """Generated compaction data ready to be persisted as a compaction entry."""

    # Summary text that replaces compacted history in future context.
    summary: str
    # Estimated context tokens before compaction.
    tokens_before: int
    # Entry id where retained history starts.
    first_kept_entry_id: str | None = None
    # Usage from the LLM call(s) that generated this summary, if available.
    usage: Usage | None = None
    # Retained recent messages stored directly on the compaction entry.
    retained_tail: list[AgentMessage] | None = None
    # Optional implementation-specific details stored with the compaction entry.
    details: Any = None


async def complete_simple_with_retries(
    models,
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
):
    # Summaries are standalone requests, so isolate routing and avoid cache
    # writes that cannot be reused.
    request_options = replace(options, cache_retention="none", session_id=uuidv7())
    return await retry_assistant_call(
        lambda: models.complete_simple(model, context, request_options),
        retry,
        request_options.cancel,
        callbacks,
    )


def _combine_usage(first: Usage, second: Usage) -> Usage:
    return Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        cache_write_1h=(
            (first.cache_write_1h or 0) + (second.cache_write_1h or 0)
            if first.cache_write_1h is not None or second.cache_write_1h is not None
            else None
        ),
        reasoning=(
            (first.reasoning or 0) + (second.reasoning or 0)
            if first.reasoning is not None or second.reasoning is not None
            else None
        ),
        total_tokens=first.total_tokens + second.total_tokens,
        cost=UsageCost(
            input=first.cost.input + second.cost.input,
            output=first.cost.output + second.cost.output,
            cache_read=first.cost.cache_read + second.cost.cache_read,
            cache_write=first.cost.cache_write + second.cost.cache_write,
            total=first.cost.total + second.cost.total,
        ),
    )


@dataclass(slots=True)
class CompactionSettings:
    """Compaction thresholds and retention settings."""

    # Enable automatic compaction decisions.
    enabled: bool
    # Tokens reserved for summary prompt and output.
    reserve_tokens: int
    # Approximate recent-context tokens to keep after compaction.
    keep_recent_tokens: int


DEFAULT_COMPACTION_SETTINGS = CompactionSettings(enabled=True, reserve_tokens=16384, keep_recent_tokens=20000)


def calculate_context_tokens(usage: Usage) -> int:
    """Calculate total context tokens from provider usage."""
    return usage.total_tokens or usage.input + usage.output + usage.cache_read + usage.cache_write


def _get_assistant_usage(msg: AgentMessage) -> Usage | None:
    if (
        getattr(msg, "role", None) == "assistant"
        and hasattr(msg, "usage")
        and msg.stop_reason not in ("aborted", "error")
        and msg.usage is not None
        and calculate_context_tokens(msg.usage) > 0
    ):
        return msg.usage
    return None


def get_last_assistant_usage(entries: list[SessionTreeEntry]) -> Usage | None:
    """Return usage from the last valid assistant message in session entries."""
    for entry in reversed(entries):
        if entry.type == "message":
            usage = _get_assistant_usage(entry.message)
            if usage is not None:
                return usage
    return None


@dataclass(slots=True)
class ContextUsageEstimate:
    """Estimated context-token usage for a message list."""

    # Estimated total context tokens.
    tokens: int
    # Tokens reported by the most recent assistant usage block.
    usage_tokens: int
    # Estimated tokens after the most recent assistant usage block.
    trailing_tokens: int
    # Index of the message that provided usage, or None when none exists.
    last_usage_index: int | None


def _get_last_assistant_usage_info(messages: list[AgentMessage]) -> tuple[Usage, int] | None:
    for index in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[index])
        if usage is not None:
            return usage, index
    return None


def estimate_context_tokens(messages: list[AgentMessage]) -> ContextUsageEstimate:
    """Estimate context tokens for messages using provider usage when available."""
    usage_info = _get_last_assistant_usage_info(messages)

    if usage_info is None:
        estimated = sum(estimate_tokens(message) for message in messages)
        return ContextUsageEstimate(tokens=estimated, usage_tokens=0, trailing_tokens=estimated, last_usage_index=None)

    usage, index = usage_info
    usage_tokens = calculate_context_tokens(usage)
    trailing_tokens = sum(estimate_tokens(message) for message in messages[index + 1 :])

    return ContextUsageEstimate(
        tokens=usage_tokens + trailing_tokens,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing_tokens,
        last_usage_index=index,
    )


def should_compact(context_tokens: int, context_window: int, settings: CompactionSettings) -> bool:
    """Return whether context usage exceeds the configured compaction threshold."""
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


ESTIMATED_IMAGE_CHARS = 4800


def _estimate_text_and_image_content_chars(content: Any) -> int:
    if isinstance(content, str):
        return len(content)

    chars = 0
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type == "text" and block.text:
            chars += len(block.text)
        elif block_type == "image":
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_tokens(message: AgentMessage) -> int:
    """Estimate token count for one message using a conservative character heuristic."""
    role = getattr(message, "role", None)
    chars = 0

    if role == "user":
        chars = _estimate_text_and_image_content_chars(message.content)
        return math.ceil(chars / 4)
    if role == "assistant":
        for block in message.content:
            if block.type == "text":
                chars += len(block.text)
            elif block.type == "thinking":
                chars += len(block.thinking)
            elif block.type == "toolCall":
                chars += len(block.name) + len(safe_json_stringify(block.arguments))
        return math.ceil(chars / 4)
    if role in ("custom", "toolResult"):
        chars = _estimate_text_and_image_content_chars(message.content)
        return math.ceil(chars / 4)
    if role == "bashExecution":
        chars = len(message.command) + len(message.output)
        return math.ceil(chars / 4)
    if role in ("branchSummary", "compactionSummary"):
        chars = len(message.summary)
        return math.ceil(chars / 4)

    return 0


def _find_valid_cut_points(entries: list[SessionTreeEntry], start_index: int, end_index: int) -> list[int]:
    cut_points: list[int] = []
    for index in range(start_index, end_index):
        entry = entries[index]
        if entry.type == "message":
            role = getattr(entry.message, "role", None)
            if role in ("bashExecution", "custom", "branchSummary", "compactionSummary", "user", "assistant"):
                cut_points.append(index)
        if entry.type in ("branch_summary", "custom_message"):
            cut_points.append(index)
    return cut_points


def find_turn_start_index(entries: list[SessionTreeEntry], entry_index: int, start_index: int) -> int:
    """Find the user-visible message that starts the turn containing an entry."""
    for index in range(entry_index, start_index - 1, -1):
        entry = entries[index]
        if entry.type in ("branch_summary", "custom_message"):
            return index
        if entry.type == "message":
            role = getattr(entry.message, "role", None)
            if role in ("user", "bashExecution"):
                return index
    return -1


@dataclass(slots=True)
class CutPointResult:
    """Cut point selected for compaction."""

    # Index of the first entry retained after compaction.
    first_kept_entry_index: int
    # Index of the turn-start entry when the cut splits a turn, otherwise -1.
    turn_start_index: int
    # Whether the selected cut point splits an in-progress turn.
    is_split_turn: bool


def find_cut_point(
    entries: list[SessionTreeEntry],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """Find the compaction cut point that keeps approximately the requested recent-token budget."""
    cut_points = _find_valid_cut_points(entries, start_index, end_index)

    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)
    accumulated_tokens = 0
    cut_index = cut_points[0]

    for index in range(end_index - 1, start_index - 1, -1):
        entry = entries[index]
        if entry.type != "message":
            continue
        accumulated_tokens += estimate_tokens(entry.message)
        if accumulated_tokens >= keep_recent_tokens:
            for candidate in cut_points:
                if candidate >= index:
                    cut_index = candidate
                    break
            break
    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        if prev_entry.type == "compaction":
            break
        if prev_entry.type == "message":
            break
        cut_index -= 1
    cut_entry = entries[cut_index]
    is_user_message = cut_entry.type == "message" and getattr(cut_entry.message, "role", None) == "user"
    turn_start_index = -1 if is_user_message else find_turn_start_index(entries, cut_index, start_index)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=not is_user_message and turn_start_index != -1,
    )


SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation between a user and an "
    "AI assistant, then produce a structured summary following the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the conversation. "
    "ONLY output the structured summary."
)

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_summary(
    current_messages: list[AgentMessage],
    models,
    model: Model,
    reserve_tokens: int,
    cancel: CancelToken | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: str | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result[str, CompactionError]:
    """Generate or update a conversation summary for compaction."""
    result = await generate_summary_with_usage(
        current_messages,
        models,
        model,
        reserve_tokens,
        cancel,
        custom_instructions,
        previous_summary,
        thinking_level,
        retry,
        callbacks,
    )
    return ok(result.value[0]) if result.ok else err(result.error)


async def generate_summary_with_usage(
    current_messages: list[AgentMessage],
    models,
    model: Model,
    reserve_tokens: int,
    cancel: CancelToken | None = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: str | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result[tuple[str, Usage], CompactionError]:
    """Generate or update a conversation summary and return its provider usage."""
    max_tokens = min(math.floor(0.8 * reserve_tokens), model.max_tokens if model.max_tokens > 0 else math.inf)
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"
    llm_messages = convert_to_llm(current_messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    summarization_messages = [UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000))]

    completion_options = SimpleStreamOptions(max_tokens=int(max_tokens), cancel=cancel)
    if model.reasoning and thinking_level and thinking_level != "off":
        completion_options.reasoning = thinking_level

    response = await complete_simple_with_retries(
        models,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        retry,
        callbacks,
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed", f"Summarization failed: {response.error_message or 'Unknown error'}"
            )
        )

    return ok((content_text(response.content), response.usage))


@dataclass(slots=True)
class CompactionPreparation:
    """Prepared inputs for a compaction run."""

    # Entry id where retained history starts.
    first_kept_entry_id: str
    # Messages summarized into the history summary.
    messages_to_summarize: list[AgentMessage]
    # Prefix messages summarized separately when compaction splits a turn.
    turn_prefix_messages: list[AgentMessage]
    # Recent messages retained after compaction and stored on the compaction entry.
    retained_tail: list[AgentMessage]
    # Whether compaction splits a turn.
    is_split_turn: bool
    # Estimated context tokens before compaction.
    tokens_before: int
    # File operations extracted from summarized history.
    file_ops: FileOperations
    # Settings used to prepare compaction.
    settings: CompactionSettings
    # Previous compaction summary used for iterative updates.
    previous_summary: str | None = None


def prepare_compaction(
    path_entries: list[SessionTreeEntry],
    settings: CompactionSettings,
) -> Result[CompactionPreparation | None, CompactionError]:
    """Prepare session entries for compaction, or return Ok(None) when not applicable."""
    if not path_entries or path_entries[-1].type == "compaction":
        return ok(None)

    prev_compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if path_entries[index].type == "compaction":
            prev_compaction_index = index
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev_compaction: CompactionEntry = path_entries[prev_compaction_index]
        previous_summary = prev_compaction.summary
        first_kept_entry_index = (
            next(
                (i for i, entry in enumerate(path_entries) if entry.id == prev_compaction.first_kept_entry_id),
                -1,
            )
            if prev_compaction.first_kept_entry_id
            else -1
        )
        boundary_start = first_kept_entry_index if first_kept_entry_index >= 0 else prev_compaction_index + 1
    boundary_end = len(path_entries)

    tokens_before = estimate_context_tokens(build_session_context(path_entries).messages).tokens

    cut_point = find_cut_point(path_entries, boundary_start, boundary_end, settings.keep_recent_tokens)
    first_kept_entry = path_entries[cut_point.first_kept_entry_index]
    if not first_kept_entry.id:
        return err(CompactionError("invalid_session", "First kept entry has no UUID - session may need migration"))
    first_kept_entry_id = first_kept_entry.id

    history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index
    messages_to_summarize: list[AgentMessage] = []
    for index in range(boundary_start, history_end):
        msg = _get_message_from_entry_for_compaction(path_entries[index])
        if msg is not None:
            messages_to_summarize.append(msg)
    turn_prefix_messages: list[AgentMessage] = []
    if cut_point.is_split_turn:
        for index in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            msg = _get_message_from_entry_for_compaction(path_entries[index])
            if msg is not None:
                turn_prefix_messages.append(msg)
    retained_tail: list[AgentMessage] = []
    for index in range(cut_point.first_kept_entry_index, boundary_end):
        msg = _get_message_from_entry_for_compaction(path_entries[index])
        if msg is not None:
            retained_tail.append(msg)
    file_ops = _extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)
    if cut_point.is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    return ok(
        CompactionPreparation(
            first_kept_entry_id=first_kept_entry_id,
            messages_to_summarize=messages_to_summarize,
            turn_prefix_messages=turn_prefix_messages,
            retained_tail=retained_tail,
            is_split_turn=cut_point.is_split_turn,
            tokens_before=tokens_before,
            previous_summary=previous_summary,
            file_ops=file_ops,
            settings=settings,
        )
    )


TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


async def compact(
    preparation: CompactionPreparation,
    models,
    model: Model,
    custom_instructions: str | None = None,
    cancel: CancelToken | None = None,
    thinking_level: str | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result[CompactionResult, CompactionError]:
    """Generate compaction summary data from prepared session history."""
    if not preparation.first_kept_entry_id:
        return err(CompactionError("invalid_session", "First kept entry has no UUID - session may need migration"))

    settings = preparation.settings
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Usage | None = None
        if preparation.messages_to_summarize:
            history_result = await generate_summary_with_usage(
                preparation.messages_to_summarize,
                models,
                model,
                settings.reserve_tokens,
                cancel,
                custom_instructions,
                preparation.previous_summary,
                thinking_level,
                retry,
                callbacks,
            )
            if not history_result.ok:
                return err(history_result.error)
            history_text, history_usage = history_result.value
        turn_prefix_result = await _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            models,
            model,
            settings.reserve_tokens,
            cancel,
            thinking_level,
            retry,
            callbacks,
        )
        if not turn_prefix_result.ok:
            return err(turn_prefix_result.error)
        prefix_text, prefix_usage = turn_prefix_result.value
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{prefix_text}"
        summary_usage = _combine_usage(history_usage, prefix_usage) if history_usage is not None else prefix_usage
    else:
        summary_result = await generate_summary_with_usage(
            preparation.messages_to_summarize,
            models,
            model,
            settings.reserve_tokens,
            cancel,
            custom_instructions,
            preparation.previous_summary,
            thinking_level,
            retry,
            callbacks,
        )
        if not summary_result.ok:
            return err(summary_result.error)
        summary, summary_usage = summary_result.value

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return ok(
        CompactionResult(
            summary=summary,
            first_kept_entry_id=preparation.first_kept_entry_id,
            tokens_before=preparation.tokens_before,
            usage=summary_usage,
            retained_tail=preparation.retained_tail,
            details=CompactionDetails(read_files=read_files, modified_files=modified_files),
        )
    )


async def _generate_turn_prefix_summary(
    messages: list[AgentMessage],
    models,
    model: Model,
    reserve_tokens: int,
    cancel: CancelToken | None = None,
    thinking_level: str | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> Result[tuple[str, Usage], CompactionError]:
    max_tokens = min(math.floor(0.5 * reserve_tokens), model.max_tokens if model.max_tokens > 0 else math.inf)
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"
    summarization_messages = [UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000))]

    completion_options = SimpleStreamOptions(max_tokens=int(max_tokens), cancel=cancel)
    if model.reasoning and thinking_level and thinking_level != "off":
        completion_options.reasoning = thinking_level
    response = await complete_simple_with_retries(
        models,
        model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        completion_options,
        retry,
        callbacks,
    )
    if response.stop_reason == "aborted":
        return err(CompactionError("aborted", response.error_message or "Turn prefix summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                "summarization_failed",
                f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}",
            )
        )

    return ok((content_text(response.content), response.usage))
