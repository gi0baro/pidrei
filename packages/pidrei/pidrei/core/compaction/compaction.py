"""Mirror of pi coding-agent src/core/compaction/compaction.ts.

Context compaction for long sessions. Pure functions for compaction logic;
the session manager handles I/O, and after compaction the session is reloaded.

Session entries are the coding-agent plain camelCase dicts from
session_manager.py. Token-estimation helpers are shared with the agent
package's harness compaction port (identical logic in pi).

Deviation: pi falls back to the pi-ai compat global `completeSimple` when no
stream function is given; pidrei never ported the deprecated compat registry,
so `stream_fn` is required here (AgentSession always passes the agent's).
"""

import math
import time as time_module
from dataclasses import dataclass, replace
from typing import Any

from pidrei_agent.harness.compaction.compaction import (
    DEFAULT_COMPACTION_SETTINGS,
    CompactionSettings,
    ContextUsageEstimate,
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_tokens,
    should_compact,
)
from pidrei_ai.types import (
    AnthropicMessagesCompat,
    AnthropicRefusalFallback,
    Context,
    Model,
    SimpleStreamOptions,
    TextContent,
    ToolCall,
    Usage,
    UsageCost,
    UserMessage,
)
from pidrei_ai.utils.retry import RetryCallbacks, RetryPolicy, retry_assistant_call
from pidrei_ai.utils.tasks import gather
from pidrei_ai.utils.text import content_text
from pidrei_ai.utils.uuid import uuidv7

from ..messages import convert_to_llm
from ..session_manager import build_session_context, session_entry_to_context_messages
from .utils import (
    SUMMARIZATION_SYSTEM_PROMPT,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)


__all__ = [  # re-exported shared helpers keep pi's coding-agent module surface
    "DEFAULT_COMPACTION_SETTINGS",
    "CompactionDetails",
    "CompactionPreparation",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
    "CutPointResult",
    "calculate_context_tokens",
    "compact",
    "complete_summarization",
    "estimate_context_tokens",
    "estimate_tokens",
    "find_cut_point",
    "find_turn_start_index",
    "generate_summary",
    "generate_summary_with_usage",
    "get_last_assistant_usage",
    "prepare_compaction",
    "should_compact",
]


def _get_anthropic_summarization_fallback(model: Model) -> AnthropicRefusalFallback | None:
    if model.provider != "anthropic" or model.api != "anthropic-messages":
        return None

    compat = model.compat if isinstance(model.compat, AnthropicMessagesCompat) else None
    allowed_fallback_models = compat.allowed_fallback_models if compat else None
    # Use the primary permitted fallback for now. If future Anthropic models expose
    # broader fallback behavior, this can become a user/config pick or a full chain.
    return [allowed_fallback_models[0]] if allowed_fallback_models else None


@dataclass(slots=True)
class CompactionDetails:
    """Details stored in compaction entry details for file tracking."""

    read_files: list[str]
    modified_files: list[str]


def _extract_file_operations(
    messages: list[Any],
    entries: list[dict[str, Any]],
    prev_compaction_index: int,
) -> FileOperations:
    """Extract file operations from messages and previous compaction entries."""
    file_ops = create_file_ops()

    # Collect from previous compaction's details (if pidrei-generated)
    if prev_compaction_index >= 0:
        prev_compaction = entries[prev_compaction_index]
        if not prev_compaction.get("fromHook") and prev_compaction.get("details") is not None:
            # fromHook field kept for session file compatibility
            details = prev_compaction["details"]
            if isinstance(details, dict):
                read_files = details.get("readFiles")
                modified_files = details.get("modifiedFiles")
            else:
                read_files = getattr(details, "read_files", None)
                modified_files = getattr(details, "modified_files", None)
            if isinstance(read_files, list):
                file_ops.read.update(read_files)
            if isinstance(modified_files, list):
                file_ops.edited.update(modified_files)

    # Extract from tool calls in messages
    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)

    return file_ops


def _get_message_from_entry_for_compaction(entry: dict[str, Any]) -> Any:
    """Extract the first context message from an entry if it produces one."""
    if entry.get("type") == "compaction":
        return None
    messages = session_entry_to_context_messages(entry)
    return messages[0] if messages else None


@dataclass(slots=True)
class CompactionResult:
    """Result from compact() - SessionManager adds id/parentId when saving."""

    summary: str
    first_kept_entry_id: str
    tokens_before: int
    estimated_tokens_after: int | None = None
    # Usage from the LLM call(s) that generated this summary, if available
    usage: Usage | None = None
    # Extension-specific data (e.g. structured compaction markers)
    details: Any = None


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


def _get_assistant_usage(msg: Any) -> Usage | None:
    """Usage from an assistant message if available.
    Skips aborted, error, and all-zero usage messages."""
    if (
        getattr(msg, "role", None) == "assistant"
        and hasattr(msg, "usage")
        and msg.stop_reason not in ("aborted", "error")
        and msg.usage is not None
        and calculate_context_tokens(msg.usage) > 0
    ):
        return msg.usage
    return None


def get_last_assistant_usage(entries: list[dict[str, Any]]) -> Usage | None:
    """Find the last valid assistant message usage from session entries."""
    for entry in reversed(entries):
        if entry.get("type") == "message":
            usage = _get_assistant_usage(entry.get("message"))
            if usage is not None:
                return usage
    return None


# ---------------------------------------------------------------------------
# Cut point detection
# ---------------------------------------------------------------------------


def _is_cut_point_message(message: Any) -> bool:
    return getattr(message, "role", None) in (
        "user",
        "assistant",
        "bashExecution",
        "custom",
        "branchSummary",
        "compactionSummary",
    )


def _is_turn_start_message(message: Any) -> bool:
    return getattr(message, "role", None) in (
        "user",
        "bashExecution",
        "custom",
        "branchSummary",
        "compactionSummary",
    )


def _is_turn_start_entry(entry: dict[str, Any]) -> bool:
    if entry.get("type") == "compaction":
        return False
    return any(_is_turn_start_message(message) for message in session_entry_to_context_messages(entry))


def _find_valid_cut_points(entries: list[dict[str, Any]], start_index: int, end_index: int) -> list[int]:
    """Find valid cut points: indices of context-visible user-like or assistant
    messages. Never cut at tool results (they must follow their tool call)."""
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if entry.get("type") == "compaction":
            continue
        if any(_is_cut_point_message(message) for message in session_entry_to_context_messages(entry)):
            cut_points.append(i)
    return cut_points


def find_turn_start_index(entries: list[dict[str, Any]], entry_index: int, start_index: int) -> int:
    """Find the context-visible user-role message that starts the turn containing
    the given entry index. Returns -1 if no turn start found before the index."""
    for i in range(entry_index, start_index - 1, -1):
        if _is_turn_start_entry(entries[i]):
            return i
    return -1


@dataclass(slots=True)
class CutPointResult:
    # Index of first entry to keep
    first_kept_entry_index: int
    # Index of user message that starts the turn being split, or -1 if not splitting
    turn_start_index: int
    # Whether this cut splits a turn (cut point is not a user message)
    is_split_turn: bool


def find_cut_point(
    entries: list[dict[str, Any]],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """Find the cut point in session entries that keeps approximately
    `keep_recent_tokens`: walk backwards from newest accumulating estimated
    message sizes, stop when the budget is reached, cut at the closest valid
    cut point at or after that entry."""
    cut_points = _find_valid_cut_points(entries, start_index, end_index)

    if not cut_points:
        return CutPointResult(first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False)

    # Walk backwards from newest, accumulating estimated message sizes
    accumulated_tokens = 0
    cut_index = cut_points[0]  # Default: keep from first message (not header)

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        message_tokens = sum(estimate_tokens(message) for message in session_entry_to_context_messages(entry))
        if message_tokens == 0:
            continue
        accumulated_tokens += message_tokens

        # Check if we've exceeded the budget
        if accumulated_tokens >= keep_recent_tokens:
            # Find the closest valid cut point at or after this entry
            for candidate in cut_points:
                if candidate >= i:
                    cut_index = candidate
                    break
            break

    # Scan backwards from cut_index to include adjacent metadata entries that do
    # not affect context.
    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        # Stop at compaction boundaries or context-visible entries.
        if prev_entry.get("type") == "compaction" or session_entry_to_context_messages(prev_entry):
            break
        cut_index -= 1

    # Determine if this is a split turn
    cut_entry = entries[cut_index]
    starts_turn = _is_turn_start_entry(cut_entry)
    turn_start_index = -1 if starts_turn else find_turn_start_index(entries, cut_index, start_index)

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_index,
        is_split_turn=not starts_turn and turn_start_index != -1,
    )


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

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

UPDATE_SUMMARIZATION_INSTRUCTIONS = """Update the existing structured summary with new information. RULES:
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

UPDATE_SUMMARIZATION_PROMPT = f"""The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

{UPDATE_SUMMARIZATION_INSTRUCTIONS}"""


def _build_summarization_context(prompt_text: str) -> Context:
    """Build the provider context for a standalone summary request."""
    return Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=[UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time_module.time() * 1000))],
    )


def _create_summarization_options(
    model: Model,
    max_tokens: float,
    api_key: str | None,
    headers: dict[str, str] | None,
    env: dict[str, str] | None,
    cancel,
    thinking_level: str | None,
    session_id: str | None,
) -> SimpleStreamOptions:
    options = SimpleStreamOptions(
        max_tokens=int(max_tokens) if max_tokens != math.inf else None,
        cancel=cancel,
        api_key=api_key,
        headers=headers,
        env=env,
        session_id=session_id,
    )
    refusal_fallbacks = _get_anthropic_summarization_fallback(model)
    if refusal_fallbacks:
        options.refusal_fallbacks = refusal_fallbacks
    if model.reasoning and thinking_level and thinking_level != "off":
        options.reasoning = thinking_level
    return options


async def complete_summarization(
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
    stream_fn=None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
):
    """Shared choke point for every compaction/branch-summary summarization call.
    Wraps the single LLM call in retry_assistant_call so transient stream drops
    honor the configured retry policy instead of failing the whole compaction on
    the first attempt. Deterministic errors and aborts return immediately."""
    # Avoid cache writes for one-off summaries. Reuse caller-supplied routing when
    # available; callers without a session ID, including branch summaries, receive a
    # fresh routing ID.
    request_options = replace(
        options,
        cache_retention="none",
        session_id=options.session_id if options.session_id is not None else uuidv7(),
        tool_choice="none",
    )
    if stream_fn is None:
        raise Exception("complete_summarization requires a stream function (pi's compat registry is not ported)")

    async def produce():
        stream = await stream_fn(model, context, request_options)
        return await stream.result()

    return await retry_assistant_call(produce, retry, request_options.cancel, callbacks)


@dataclass(slots=True)
class SummaryWithUsage:
    text: str
    usage: Usage


async def generate_summary(
    current_messages: list[Any],
    model: Model,
    reserve_tokens: int,
    api_key: str | None,
    headers: dict[str, str] | None = None,
    cancel=None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: str | None = None,
    stream_fn=None,
    env: dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
    session_id: str | None = None,
) -> str:
    """Generate a summary of the conversation using the LLM.
    If previous_summary is provided, uses the update prompt to merge."""
    result = await generate_summary_with_usage(
        current_messages,
        model,
        reserve_tokens,
        api_key,
        headers,
        cancel,
        custom_instructions,
        previous_summary,
        thinking_level,
        stream_fn,
        env,
        retry,
        callbacks,
        session_id,
    )
    return result.text


async def generate_summary_with_usage(
    current_messages: list[Any],
    model: Model,
    reserve_tokens: int,
    api_key: str | None,
    headers: dict[str, str] | None = None,
    cancel=None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: str | None = None,
    stream_fn=None,
    env: dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
    session_id: str | None = None,
) -> SummaryWithUsage:
    """Generate or update a conversation summary and return its provider usage."""
    max_tokens = min(math.floor(0.8 * reserve_tokens), model.max_tokens if model.max_tokens > 0 else math.inf)

    # Use update prompt if we have a previous summary, otherwise initial prompt
    base_prompt = UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    # Serialize conversation to text so model doesn't try to continue it.
    # Convert to LLM messages first (handles custom types like bashExecution).
    llm_messages = convert_to_llm(current_messages)
    conversation_text = serialize_conversation(llm_messages)

    # Build the prompt with conversation wrapped in tags
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
    prompt_text += base_prompt

    completion_options = _create_summarization_options(
        model, max_tokens, api_key, headers, env, cancel, thinking_level, session_id
    )

    response = await complete_summarization(
        model,
        _build_summarization_context(prompt_text),
        completion_options,
        stream_fn,
        retry,
        callbacks,
    )

    if response.stop_reason == "error":
        raise Exception(f"Summarization failed: {response.error_message or 'Unknown error'}")
    if any(isinstance(block, ToolCall) for block in response.content):
        raise Exception("Summarization attempted to call a tool")

    return SummaryWithUsage(text=content_text(response.content), usage=response.usage)


# ---------------------------------------------------------------------------
# Compaction preparation (for extensions)
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CompactionPreparation:
    # Entry id of first entry to keep
    first_kept_entry_id: str
    # Messages that will be summarized and discarded
    messages_to_summarize: list[Any]
    # Messages that will be turned into turn prefix summary (if splitting)
    turn_prefix_messages: list[Any]
    # Whether this is a split turn (cut point in middle of turn)
    is_split_turn: bool
    tokens_before: int
    # File operations extracted from messages_to_summarize
    file_ops: FileOperations
    # Compaction settings from settings.json
    settings: CompactionSettings
    # Summary from previous compaction, for iterative update
    previous_summary: str | None = None


def prepare_compaction(
    path_entries: list[dict[str, Any]],
    settings: CompactionSettings,
) -> CompactionPreparation | None:
    if path_entries and path_entries[-1].get("type") == "compaction":
        return None

    prev_compaction_index = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if path_entries[i].get("type") == "compaction":
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev_compaction = path_entries[prev_compaction_index]
        previous_summary = prev_compaction.get("summary")
        first_kept_entry_index = next(
            (i for i, entry in enumerate(path_entries) if entry.get("id") == prev_compaction.get("firstKeptEntryId")),
            -1,
        )
        boundary_start = first_kept_entry_index if first_kept_entry_index >= 0 else prev_compaction_index + 1
    boundary_end = len(path_entries)

    tokens_before = estimate_context_tokens(build_session_context(path_entries).messages).tokens

    cut_point = find_cut_point(path_entries, boundary_start, boundary_end, settings.keep_recent_tokens)

    # Get id of first kept entry
    first_kept_entry = (
        path_entries[cut_point.first_kept_entry_index] if cut_point.first_kept_entry_index < len(path_entries) else None
    )
    if not first_kept_entry or not first_kept_entry.get("id"):
        return None  # Session needs migration
    first_kept_entry_id = first_kept_entry["id"]

    history_end = cut_point.turn_start_index if cut_point.is_split_turn else cut_point.first_kept_entry_index

    # Messages to summarize (will be discarded after summary)
    messages_to_summarize: list[Any] = []
    for i in range(boundary_start, history_end):
        msg = _get_message_from_entry_for_compaction(path_entries[i])
        if msg is not None:
            messages_to_summarize.append(msg)

    # Messages for turn prefix summary (if splitting a turn)
    turn_prefix_messages: list[Any] = []
    if cut_point.is_split_turn:
        for i in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            msg = _get_message_from_entry_for_compaction(path_entries[i])
            if msg is not None:
                turn_prefix_messages.append(msg)

    if not messages_to_summarize and not turn_prefix_messages:
        return None

    # Extract file operations from messages and previous compaction
    file_ops = _extract_file_operations(messages_to_summarize, path_entries, prev_compaction_index)

    # Also extract file ops from turn prefix if splitting
    if cut_point.is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut_point.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Main compaction function
# ---------------------------------------------------------------------------

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
    model: Model,
    api_key: str | None,
    headers: dict[str, str] | None = None,
    custom_instructions: str | None = None,
    cancel=None,
    thinking_level: str | None = None,
    stream_fn=None,
    env: dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
    session_id: str | None = None,
) -> CompactionResult:
    """Generate summaries for compaction using prepared data.

    `session_id` is an optional routing session ID forwarded without enabling
    prompt caching.
    Returns CompactionResult - SessionManager adds id/parentId when saving."""
    settings = preparation.settings

    # Generate summaries and merge into one
    if preparation.is_split_turn and preparation.turn_prefix_messages:
        # The history and turn-prefix summaries are independent LLM calls;
        # run them concurrently so compaction costs max, not sum.
        history_text = "No prior history."
        history_usage: Usage | None = None
        turn_prefix_call = _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            model,
            settings.reserve_tokens,
            api_key,
            headers,
            env,
            cancel,
            thinking_level,
            stream_fn,
            retry,
            callbacks,
            session_id,
        )
        if preparation.messages_to_summarize:
            history_result, turn_prefix_result = await gather(
                generate_summary_with_usage(
                    preparation.messages_to_summarize,
                    model,
                    settings.reserve_tokens,
                    api_key,
                    headers,
                    cancel,
                    custom_instructions,
                    preparation.previous_summary,
                    thinking_level,
                    stream_fn,
                    env,
                    retry,
                    callbacks,
                    session_id,
                ),
                turn_prefix_call,
            )
            history_text = history_result.text
            history_usage = history_result.usage
        else:
            turn_prefix_result = await turn_prefix_call
        # Merge into single summary
        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_result.text}"
        summary_usage = (
            _combine_usage(history_usage, turn_prefix_result.usage)
            if history_usage is not None
            else turn_prefix_result.usage
        )
    else:
        # Just generate history summary
        result = await generate_summary_with_usage(
            preparation.messages_to_summarize,
            model,
            settings.reserve_tokens,
            api_key,
            headers,
            cancel,
            custom_instructions,
            preparation.previous_summary,
            thinking_level,
            stream_fn,
            env,
            retry,
            callbacks,
            session_id,
        )
        summary = result.text
        summary_usage = result.usage

    # Compute file lists and append to summary
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    if not preparation.first_kept_entry_id:
        raise Exception("First kept entry has no id - session may need migration")

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        usage=summary_usage,
        details={"readFiles": read_files, "modifiedFiles": modified_files},
    )


async def _generate_turn_prefix_summary(
    messages: list[Any],
    model: Model,
    reserve_tokens: int,
    api_key: str | None,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    cancel=None,
    thinking_level: str | None = None,
    stream_fn=None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
    session_id: str | None = None,
) -> SummaryWithUsage:
    """Generate a summary for a turn prefix (when splitting a turn)."""
    # Smaller budget for turn prefix
    max_tokens = min(math.floor(0.5 * reserve_tokens), model.max_tokens if model.max_tokens > 0 else math.inf)
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"

    response = await complete_summarization(
        model,
        _build_summarization_context(prompt_text),
        _create_summarization_options(model, max_tokens, api_key, headers, env, cancel, thinking_level, session_id),
        stream_fn,
        retry,
        callbacks,
    )

    if response.stop_reason == "error":
        raise Exception(f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}")
    if any(isinstance(block, ToolCall) for block in response.content):
        raise Exception("Turn prefix summarization attempted to call a tool")

    return SummaryWithUsage(text=content_text(response.content), usage=response.usage)
