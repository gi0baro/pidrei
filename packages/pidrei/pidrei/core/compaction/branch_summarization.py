"""Mirror of pi coding-agent src/core/compaction/branch-summarization.ts.

Branch summarization for tree navigation: when navigating to a different
point in the session tree, this generates a summary of the branch being left
so context isn't lost.
"""

import time as time_module
from dataclasses import dataclass
from typing import Any

from pidrei_ai.types import Context, Model, SimpleStreamOptions, TextContent, ToolCall, Usage, UserMessage
from pidrei_ai.utils.retry import RetryCallbacks, RetryPolicy
from pidrei_ai.utils.text import content_text

from ..messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from .compaction import complete_summarization, estimate_tokens, get_summarization_failure
from .utils import (
    SUMMARIZATION_SYSTEM_PROMPT,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)


@dataclass(slots=True)
class BranchSummaryResult:
    summary: str | None = None
    usage: Usage | None = None
    read_files: list[str] | None = None
    modified_files: list[str] | None = None
    aborted: bool | None = None
    error: str | None = None


@dataclass(slots=True)
class BranchSummaryDetails:
    """Details stored in branch_summary entry details for file tracking."""

    read_files: list[str]
    modified_files: list[str]


@dataclass(slots=True)
class BranchPreparation:
    # Messages extracted for summarization, in chronological order
    messages: list[Any]
    # File operations extracted from tool calls
    file_ops: FileOperations
    # Total estimated tokens in messages
    total_tokens: int


@dataclass(slots=True)
class CollectEntriesResult:
    # Entries to summarize, in chronological order
    entries: list[dict[str, Any]]
    # Common ancestor between old and new position, if any
    common_ancestor_id: str | None


def collect_entries_for_branch_summary(
    session: Any,
    old_leaf_id: str | None,
    target_id: str,
) -> CollectEntriesResult:
    """Collect entries that should be summarized when navigating from one
    position to another. Walks from old_leaf_id back to the common ancestor
    with target_id. Does NOT stop at compaction boundaries - those are included
    and their summaries become context."""
    # If no old position, nothing to summarize
    if not old_leaf_id:
        return CollectEntriesResult(entries=[], common_ancestor_id=None)

    # Find common ancestor (deepest node that's on both paths)
    old_path = {entry["id"] for entry in session.get_branch(old_leaf_id)}
    target_path = session.get_branch(target_id)

    # target_path is root-first, so iterate backwards to find deepest common ancestor
    common_ancestor_id: str | None = None
    for entry in reversed(target_path):
        if entry["id"] in old_path:
            common_ancestor_id = entry["id"]
            break

    # Collect entries from old leaf back to common ancestor
    entries: list[dict[str, Any]] = []
    current: str | None = old_leaf_id

    while current and current != common_ancestor_id:
        entry = session.get_entry(current)
        if entry is None:
            break
        entries.append(entry)
        current = entry.get("parentId")

    # Reverse to get chronological order
    entries.reverse()

    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


def _get_message_from_entry(entry: dict[str, Any]) -> Any:
    """Extract an agent message from a session entry. Similar to the compaction
    variant but also handles compaction entries."""
    entry_type = entry.get("type")
    if entry_type == "message":
        # Skip tool results - context is in assistant's tool call
        if getattr(entry.get("message"), "role", None) == "toolResult":
            return None
        return entry.get("message")
    if entry_type == "custom_message":
        return create_custom_message(
            entry.get("customType"),
            entry.get("content"),
            entry.get("display"),
            entry.get("details"),
            entry.get("timestamp"),
        )
    if entry_type == "branch_summary":
        return create_branch_summary_message(entry.get("summary"), entry.get("fromId"), entry.get("timestamp"))
    if entry_type == "compaction":
        return create_compaction_summary_message(
            entry.get("summary"), entry.get("tokensBefore"), entry.get("timestamp")
        )
    # thinking_level_change / model_change / custom / label / session_info
    # don't contribute to conversation content.
    return None


def prepare_branch_entries(entries: list[dict[str, Any]], token_budget: int = 0) -> BranchPreparation:
    """Prepare entries for summarization with token budget.

    Walks entries from NEWEST to OLDEST, adding messages until we hit the token
    budget, so the most recent context is kept when the branch is too long.
    Also collects file operations from tool calls and existing branch_summary
    entries' details (for cumulative tracking)."""
    messages: list[Any] = []
    file_ops = create_file_ops()
    total_tokens = 0

    # First pass: collect file ops from ALL entries (even if they don't fit in
    # the token budget), only from pidrei-generated summaries (fromHook != true).
    for entry in entries:
        if entry.get("type") == "branch_summary" and not entry.get("fromHook") and entry.get("details") is not None:
            details = entry["details"]
            read_files = details.get("readFiles") if isinstance(details, dict) else None
            modified_files = details.get("modifiedFiles") if isinstance(details, dict) else None
            if isinstance(read_files, list):
                file_ops.read.update(read_files)
            if isinstance(modified_files, list):
                # Modified files go into edited for proper deduplication
                file_ops.edited.update(modified_files)

    # Second pass: walk from newest to oldest, adding messages until token budget
    for entry in reversed(entries):
        message = _get_message_from_entry(entry)
        if message is None:
            continue

        # Extract file ops from assistant messages (tool calls)
        extract_file_ops_from_message(message, file_ops)

        tokens = estimate_tokens(message)

        # Check budget before adding
        if token_budget > 0 and total_tokens + tokens > token_budget:
            # If this is a summary entry, try to fit it anyway as it's important context
            if entry.get("type") in ("compaction", "branch_summary") and total_tokens < token_budget * 0.9:
                messages.insert(0, message)
                total_tokens += tokens
            # Stop - we've hit the budget
            break

        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(messages=messages, file_ops=file_ops, total_tokens=total_tokens)


BRANCH_SUMMARY_PREAMBLE = """The user explored a different conversation branch before returning here.
Summary of that exploration:

"""

BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


async def generate_branch_summary(
    entries: list[dict[str, Any]],
    *,
    model: Model,
    cancel,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    env: dict[str, str] | None = None,
    custom_instructions: str | None = None,
    replace_instructions: bool | None = None,
    reserve_tokens: int = 16384,
    stream_fn=None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> BranchSummaryResult:
    """Generate a summary of abandoned branch entries."""
    # Token budget = context window minus reserved space for prompt + response
    context_window = model.context_window or 128000
    token_budget = context_window - reserve_tokens

    preparation = prepare_branch_entries(entries, token_budget)
    messages = preparation.messages

    if not messages:
        return BranchSummaryResult(summary="No content to summarize")

    # Transform to LLM-compatible messages, then serialize to text.
    # Serialization prevents the model from treating it as a conversation to continue.
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)

    # Build prompt
    if replace_instructions and custom_instructions:
        instructions = custom_instructions
    elif custom_instructions:
        instructions = f"{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {custom_instructions}"
    else:
        instructions = BRANCH_SUMMARY_PROMPT
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"

    summarization_messages = [
        UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time_module.time() * 1000))
    ]

    # Call LLM for summarization. Prefer the session stream function so request
    # behavior (timeouts, retries, attribution headers) stays consistent without
    # running through agent state/events. Retried via complete_summarization so
    # transient stream drops reuse the configured retry policy.
    context = Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages)
    max_tokens = min(4096, model.max_tokens if model.max_tokens > 0 else float("inf"))
    request_options = SimpleStreamOptions(
        api_key=api_key, headers=headers, env=env, cancel=cancel, max_tokens=max_tokens
    )
    response = await complete_summarization(model, context, request_options, stream_fn, retry, callbacks)

    # Check if aborted or errored
    if response.stop_reason == "aborted":
        return BranchSummaryResult(aborted=True)
    failure = get_summarization_failure(response, "Branch summarization")
    if failure:
        return BranchSummaryResult(error=failure)
    if any(isinstance(block, ToolCall) for block in response.content):
        return BranchSummaryResult(error="Branch summarization attempted to call a tool")

    summary = content_text(response.content)

    # Prepend preamble to provide context about the branch summary
    summary = BRANCH_SUMMARY_PREAMBLE + summary

    # Compute file lists and append to summary
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return BranchSummaryResult(
        summary=summary or "No summary generated",
        usage=response.usage,
        read_files=read_files,
        modified_files=modified_files,
    )
