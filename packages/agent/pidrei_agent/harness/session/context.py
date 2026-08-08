"""Model-context projection over v4 entries (port of pi `session/context.ts`)."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..messages import create_branch_summary_message, create_compaction_summary_message
from .types import CustomEntry, Entry


@dataclass(slots=True)
class SessionModelRef:
    provider: str
    model_id: str


@dataclass(slots=True)
class SessionContext:
    messages: list[Any]  # AgentMessage[]
    thinking_level: str
    model: SessionModelRef | None
    active_tool_names: list[str] | None


type ContextEntryTransform = Callable[[Sequence[Entry]], Sequence[Entry]]

type CustomEntryContextMessageProjector = Callable[[CustomEntry, int, Sequence[Entry]], Sequence[Any] | None]


@dataclass(slots=True)
class SessionContextBuildOptions:
    entry_transforms: list[ContextEntryTransform] = field(default_factory=list)
    entry_projectors: Mapping[str, CustomEntryContextMessageProjector] = field(default_factory=dict)


def _derive_session_context_state(
    path_entries: Sequence[Entry],
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


def default_context_entry_transform(path_entries: Sequence[Entry]) -> list[Entry]:
    for index in range(len(path_entries) - 1, -1, -1):
        entry = path_entries[index]
        if entry.type == "compaction":
            return [entry, *path_entries[index + 1 :]]
    return list(path_entries)


def build_context_entries(
    path_entries: Sequence[Entry], options: SessionContextBuildOptions | None = None
) -> list[Entry]:
    options = options if options is not None else SessionContextBuildOptions()
    entries = default_context_entry_transform(path_entries)
    for transform in options.entry_transforms:
        entries = list(transform(entries))
    return entries


def session_entry_to_context_messages(
    entry: Entry,
    index: int,
    entries: Sequence[Entry],
    options: SessionContextBuildOptions | None = None,
) -> list[Any]:
    options = options if options is not None else SessionContextBuildOptions()
    if entry.type == "message":
        message = entry.message
        if getattr(message, "role", None) == "assistant" and message.stop_reason == "deferred":
            return []
        return [message]
    if entry.type == "compaction":
        return [
            create_compaction_summary_message(entry.summary, entry.tokens_before, entry.timestamp),
            *entry.retained_tail,
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
    path_entries: Sequence[Entry], options: SessionContextBuildOptions | None = None
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
