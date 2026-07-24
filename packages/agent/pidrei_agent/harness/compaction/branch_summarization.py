"""Branch summarization (port of pi `harness/compaction/branch-summarization.ts`)."""

import time
from dataclasses import dataclass

from pidrei_ai.types import Context, Model, SimpleStreamOptions, TextContent, Usage, UserMessage
from pidrei_ai.utils.cancel import CancelToken
from pidrei_ai.utils.retry import RetryCallbacks, RetryPolicy
from pidrei_ai.utils.text import content_text

from ...types import AgentMessage
from ..messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from ..session.session import Session
from ..types import BranchSummaryError, Result, SessionError, SessionTreeEntry, err, ok
from .compaction import SUMMARIZATION_SYSTEM_PROMPT, complete_simple_with_retries, estimate_tokens
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)


@dataclass(slots=True)
class BranchSummaryDetails:
    """File-operation details stored on generated branch summary entries."""

    # Files read while exploring the summarized branch.
    read_files: list[str]
    # Files modified while exploring the summarized branch.
    modified_files: list[str]


@dataclass(slots=True)
class BranchSummaryResult:
    summary: str
    read_files: list[str]
    modified_files: list[str]
    usage: Usage | None = None


@dataclass(slots=True)
class BranchPreparation:
    """Prepared branch content for summarization."""

    # Messages selected for the branch summary.
    messages: list[AgentMessage]
    # File operations extracted from the branch.
    file_ops: FileOperations
    # Estimated token count for selected messages.
    total_tokens: int


@dataclass(slots=True)
class CollectEntriesResult:
    """Entries selected for branch summarization."""

    # Entries to summarize in chronological order.
    entries: list[SessionTreeEntry]
    # Deepest common ancestor between the previous leaf and target entry.
    common_ancestor_id: str | None


@dataclass(slots=True, kw_only=True)
class GenerateBranchSummaryOptions:
    """Options for generating a branch summary."""

    # Provider collection the summarization request goes through; owns auth resolution.
    models: object
    # Model used for summarization.
    model: Model
    # Cancel token for the summarization request (pi: `signal`).
    cancel: CancelToken | None = None
    # Optional instructions appended to or replacing the default prompt.
    custom_instructions: str | None = None
    # Replace the default prompt with custom instructions instead of appending them.
    replace_instructions: bool | None = None
    # Tokens reserved for prompt and model output. Defaults to 16384.
    reserve_tokens: int = 16384
    # Optional retry policy for transient summarization errors.
    retry: RetryPolicy | None = None
    # Optional callbacks for retry reporting.
    callbacks: RetryCallbacks | None = None


async def collect_entries_for_branch_summary(
    session: Session,
    old_leaf_id: str | None,
    target_id: str,
) -> CollectEntriesResult:
    """Collect entries to summarize before navigating to a different session tree entry."""
    if not old_leaf_id:
        return CollectEntriesResult(entries=[], common_ancestor_id=None)
    old_path = {entry.id for entry in await session.get_branch(old_leaf_id)}
    target_path = await session.get_branch(target_id)
    common_ancestor_id: str | None = None
    for entry in reversed(target_path):
        if entry.id in old_path:
            common_ancestor_id = entry.id
            break
    entries: list[SessionTreeEntry] = []
    current: str | None = old_leaf_id

    while current and current != common_ancestor_id:
        entry = await session.get_entry(current)
        if entry is None:
            raise SessionError("invalid_session", f"Entry {current} not found")
        entries.append(entry)
        current = entry.parent_id
    entries.reverse()

    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


def _get_message_from_entry(entry: SessionTreeEntry) -> AgentMessage | None:
    if entry.type == "message":
        if getattr(entry.message, "role", None) == "toolResult":
            return None
        return entry.message
    if entry.type == "custom_message":
        return create_custom_message(entry.custom_type, entry.content, entry.display, entry.details, entry.timestamp)
    if entry.type == "branch_summary":
        return create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
    if entry.type == "compaction":
        return create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp)
    return None


def prepare_branch_entries(entries: list[SessionTreeEntry], token_budget: int = 0) -> BranchPreparation:
    """Prepare branch entries for summarization within an optional token budget."""
    messages: list[AgentMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0
    for entry in entries:
        if entry.type == "branch_summary" and not entry.from_hook and entry.details is not None:
            details = entry.details
            if isinstance(details, dict):
                read_files = details.get("readFiles", details.get("read_files"))
                modified_files = details.get("modifiedFiles", details.get("modified_files"))
            else:
                read_files = getattr(details, "read_files", None)
                modified_files = getattr(details, "modified_files", None)
            if isinstance(read_files, list):
                file_ops.read.update(read_files)
            if isinstance(modified_files, list):
                file_ops.edited.update(modified_files)
    for entry in reversed(entries):
        message = _get_message_from_entry(entry)
        if message is None:
            continue
        extract_file_ops_from_message(message, file_ops)

        tokens = estimate_tokens(message)
        if token_budget > 0 and total_tokens + tokens > token_budget:
            if entry.type in ("compaction", "branch_summary") and total_tokens < token_budget * 0.9:
                messages.insert(0, message)
                total_tokens += tokens
            break

        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(messages=messages, file_ops=file_ops, total_tokens=total_tokens)


BRANCH_SUMMARY_PREAMBLE = (
    "The user explored a different conversation branch before returning here.\nSummary of that exploration:\n\n"
)

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
    entries: list[SessionTreeEntry],
    options: GenerateBranchSummaryOptions,
) -> Result[BranchSummaryResult, BranchSummaryError]:
    """Generate a summary for abandoned branch entries."""
    context_window = options.model.context_window or 128000
    token_budget = context_window - options.reserve_tokens

    preparation = prepare_branch_entries(entries, token_budget)

    if not preparation.messages:
        return ok(BranchSummaryResult(summary="No content to summarize", read_files=[], modified_files=[]))
    llm_messages = convert_to_llm(preparation.messages)
    conversation_text = serialize_conversation(llm_messages)
    if options.replace_instructions and options.custom_instructions:
        instructions = options.custom_instructions
    elif options.custom_instructions:
        instructions = f"{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {options.custom_instructions}"
    else:
        instructions = BRANCH_SUMMARY_PROMPT
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"

    summarization_messages = [UserMessage(content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000))]
    response = await complete_simple_with_retries(
        options.models,
        options.model,
        Context(system_prompt=SUMMARIZATION_SYSTEM_PROMPT, messages=summarization_messages),
        SimpleStreamOptions(cancel=options.cancel, max_tokens=2048),
        options.retry,
        options.callbacks,
    )
    if response.stop_reason == "aborted":
        return err(BranchSummaryError("aborted", response.error_message or "Branch summary aborted"))
    if response.stop_reason == "error":
        return err(
            BranchSummaryError(
                "summarization_failed", f"Branch summary failed: {response.error_message or 'Unknown error'}"
            )
        )

    summary = content_text(response.content)
    summary = BRANCH_SUMMARY_PREAMBLE + summary
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return ok(
        BranchSummaryResult(
            summary=summary or "No summary generated",
            usage=response.usage,
            read_files=read_files,
            modified_files=modified_files,
        )
    )
