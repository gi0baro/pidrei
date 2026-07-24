"""Session tree wrapper and context building (port of pi `harness/session/session.ts`)."""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..messages import create_branch_summary_message, create_compaction_summary_message, create_custom_message
from ..types import (
    ActiveToolsChangeEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    CustomMessageEntry,
    LabelEntry,
    MessageEntry,
    ModelChangeEntry,
    SessionContext,
    SessionEntryCursorOptions,
    SessionError,
    SessionInfoEntry,
    SessionMetadata,
    SessionModelRef,
    SessionStats,
    SessionStorage,
    SessionTreeEntry,
    ThinkingLevelChangeEntry,
)


# Additional entry transform applied after the default compaction transform.
type ContextEntryTransform = Callable[[Sequence[SessionTreeEntry]], Sequence[SessionTreeEntry]]

# Optional custom-entry projector. Custom entries are omitted from model context by default.
type CustomEntryContextMessageProjector = Callable[[CustomEntry, int, Sequence[SessionTreeEntry]], Sequence[Any] | None]


@dataclass(slots=True)
class SessionContextBuildOptions:
    # Additional entry transforms applied after the default compaction transform.
    entry_transforms: list[ContextEntryTransform] = field(default_factory=list)
    # Optional custom-entry projectors keyed by custom type.
    entry_projectors: Mapping[str, CustomEntryContextMessageProjector] = field(default_factory=dict)


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _derive_session_context_state(
    path_entries: Sequence[SessionTreeEntry],
) -> tuple[str, SessionModelRef | None, list[str] | None]:
    thinking_level = "off"
    model: SessionModelRef | None = None
    active_tool_names: list[str] | None = None

    for entry in path_entries:
        if entry.type == "thinking_level_change":
            thinking_level = entry.thinking_level
        elif entry.type == "model_change":
            model = SessionModelRef(provider=entry.provider, model_id=entry.model_id)
        elif entry.type == "message" and getattr(entry.message, "role", None) == "assistant":
            model = SessionModelRef(provider=entry.message.provider, model_id=entry.message.model)
        elif entry.type == "active_tools_change":
            active_tool_names = list(entry.active_tool_names)

    return thinking_level, model, active_tool_names


def default_context_entry_transform(path_entries: Sequence[SessionTreeEntry]) -> list[SessionTreeEntry]:
    compaction: CompactionEntry | None = None
    for entry in path_entries:
        if entry.type == "compaction":
            compaction = entry
    if compaction is None:
        return list(path_entries)

    entries: list[SessionTreeEntry] = [compaction]
    compaction_idx = next(
        index for index, entry in enumerate(path_entries) if entry.type == "compaction" and entry.id == compaction.id
    )
    if compaction.retained_tail is not None:
        entries.extend(path_entries[compaction_idx + 1 :])
        return entries
    if compaction.first_kept_entry_id:
        found_first_kept = False
        for entry in path_entries[:compaction_idx]:
            if entry.id == compaction.first_kept_entry_id:
                found_first_kept = True
            if found_first_kept:
                entries.append(entry)
    entries.extend(path_entries[compaction_idx + 1 :])
    return entries


def build_context_entries(
    path_entries: Sequence[SessionTreeEntry],
    options: SessionContextBuildOptions | None = None,
) -> list[SessionTreeEntry]:
    options = options if options is not None else SessionContextBuildOptions()
    entries = default_context_entry_transform(path_entries)
    for transform in options.entry_transforms:
        entries = list(transform(entries))
    return entries


def session_entry_to_context_messages(
    entry: SessionTreeEntry,
    index: int,
    entries: Sequence[SessionTreeEntry],
    options: SessionContextBuildOptions | None = None,
) -> list[Any]:
    options = options if options is not None else SessionContextBuildOptions()
    if entry.type == "message":
        return [entry.message]
    if entry.type == "custom_message":
        return [create_custom_message(entry.custom_type, entry.content, entry.display, entry.details, entry.timestamp)]
    if entry.type == "compaction":
        return [
            create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp),
            *(entry.retained_tail or []),
        ]
    if entry.type == "branch_summary" and entry.summary:
        return [create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)]
    if entry.type == "custom":
        projector = options.entry_projectors.get(entry.custom_type)
        if projector is not None:
            return list(projector(entry, index, entries) or [])
        return []
    return []


def build_session_context(
    path_entries: Sequence[SessionTreeEntry],
    options: SessionContextBuildOptions | None = None,
) -> SessionContext:
    thinking_level, model, active_tool_names = _derive_session_context_state(path_entries)
    context_entries = build_context_entries(path_entries, options)
    messages = [
        message
        for index, entry in enumerate(context_entries)
        for message in session_entry_to_context_messages(entry, index, context_entries, options)
    ]
    return SessionContext(
        messages=messages, thinking_level=thinking_level, model=model, active_tool_names=active_tool_names
    )


class Session:
    """Session tree API over a `SessionStorage` backend."""

    def __init__(self, storage: SessionStorage, context_build_options: SessionContextBuildOptions | None = None):
        self._storage = storage
        self._context_build_options = (
            context_build_options if context_build_options is not None else SessionContextBuildOptions()
        )

    async def get_metadata(self) -> SessionMetadata:
        return await self._storage.get_metadata()

    def get_storage(self) -> SessionStorage:
        return self._storage

    async def get_leaf_id(self) -> str | None:
        return await self._storage.get_leaf_id()

    async def get_entry(self, id: str) -> SessionTreeEntry | None:
        return await self._storage.get_entry(id)

    async def get_entries(self, options: SessionEntryCursorOptions | None = None) -> list[SessionTreeEntry]:
        return await self._storage.get_entries(options)

    async def get_branch(self, from_id: str | None = None) -> list[SessionTreeEntry]:
        leaf_id = from_id if from_id is not None else await self._storage.get_leaf_id()
        return await self._storage.get_path_to_root_or_compaction(leaf_id)

    async def build_context_entries(self, options: SessionContextBuildOptions | None = None) -> list[SessionTreeEntry]:
        return build_context_entries(await self.get_branch(), self._merge_context_build_options(options))

    async def build_context(self, options: SessionContextBuildOptions | None = None) -> SessionContext:
        return build_session_context(await self.get_branch(), self._merge_context_build_options(options))

    def _merge_context_build_options(self, options: SessionContextBuildOptions | None) -> SessionContextBuildOptions:
        options = options if options is not None else SessionContextBuildOptions()
        return SessionContextBuildOptions(
            entry_transforms=[*self._context_build_options.entry_transforms, *options.entry_transforms],
            entry_projectors={**self._context_build_options.entry_projectors, **options.entry_projectors},
        )

    async def get_label(self, id: str) -> str | None:
        return await self._storage.get_label(id)

    async def get_session_stats(self) -> SessionStats:
        return await self._storage.get_session_stats()

    async def get_session_name(self) -> str | None:
        return await self._storage.get_session_name()

    async def _append_typed_entry(self, entry: SessionTreeEntry) -> str:
        await self._storage.append_entry(entry)
        return entry.id

    async def append_message(self, message: Any) -> str:
        return await self._append_typed_entry(
            MessageEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                message=message,
            )
        )

    async def append_thinking_level_change(self, thinking_level: str) -> str:
        return await self._append_typed_entry(
            ThinkingLevelChangeEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                thinking_level=thinking_level,
            )
        )

    async def append_model_change(self, provider: str, model_id: str) -> str:
        return await self._append_typed_entry(
            ModelChangeEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                provider=provider,
                model_id=model_id,
            )
        )

    async def append_active_tools_change(self, active_tool_names: list[str]) -> str:
        return await self._append_typed_entry(
            ActiveToolsChangeEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                active_tool_names=list(active_tool_names),
            )
        )

    async def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str | None,
        tokens_before: int,
        details: Any = None,
        from_hook: bool | None = None,
        usage: Any = None,
        retained_tail: list[Any] | None = None,
    ) -> str:
        return await self._append_typed_entry(
            CompactionEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
                retained_tail=retained_tail,
                details=details,
                usage=usage,
                from_hook=from_hook,
            )
        )

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._append_typed_entry(
            CustomEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                custom_type=custom_type,
                data=data,
            )
        )

    async def append_custom_message_entry(
        self, custom_type: str, content: Any, display: bool, details: Any = None
    ) -> str:
        return await self._append_typed_entry(
            CustomMessageEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                custom_type=custom_type,
                content=content,
                display=display,
                details=details,
            )
        )

    async def append_label(self, target_id: str, label: str | None) -> str:
        if await self._storage.get_entry(target_id) is None:
            raise SessionError("not_found", f"Entry {target_id} not found")
        return await self._append_typed_entry(
            LabelEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                target_id=target_id,
                label=label,
            )
        )

    async def append_session_name(self, name: str) -> str:
        sanitized_name = re.sub(r"[\r\n]+", " ", name).strip()
        return await self._append_typed_entry(
            SessionInfoEntry(
                id=await self._storage.create_entry_id(),
                parent_id=await self._storage.get_leaf_id(),
                timestamp=_iso_now(),
                name=sanitized_name,
            )
        )

    async def move_to(
        self,
        entry_id: str | None,
        summary: dict[str, Any] | None = None,
    ) -> str | None:
        """Move the leaf; optionally append a branch summary.

        `summary` mirrors pi's shape: {"summary": str, "details"?, "usage"?, "from_hook"?}.
        """
        if entry_id is not None and await self._storage.get_entry(entry_id) is None:
            raise SessionError("not_found", f"Entry {entry_id} not found")
        await self._storage.set_leaf_id(entry_id)
        if summary is None:
            return None
        return await self._append_typed_entry(
            BranchSummaryEntry(
                id=await self._storage.create_entry_id(),
                parent_id=entry_id,
                timestamp=_iso_now(),
                from_id=entry_id if entry_id is not None else "root",
                summary=summary["summary"],
                details=summary.get("details"),
                usage=summary.get("usage"),
                from_hook=summary.get("from_hook"),
            )
        )
