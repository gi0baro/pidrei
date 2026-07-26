"""In-memory session storage (port of pi `harness/session/memory-storage.ts`).

State mutations are guarded by a thread lock: storage methods are called from
multiple tonio tasks (harness turn vs background compaction later) and the
critical sections are sync-only.
"""

import threading
from datetime import UTC, datetime

from pidrei_ai.utils.uuid import uuidv7

from ..types import (
    LeafEntry,
    SessionEntryCursorOptions,
    SessionError,
    SessionMetadata,
    SessionStats,
    SessionTreeEntry,
)


def _update_label_cache(labels_by_id: dict[str, str], entry: SessionTreeEntry) -> None:
    if entry.type != "label":
        return
    label = entry.label.strip() if entry.label else ""
    if label:
        labels_by_id[entry.target_id] = label
    else:
        labels_by_id.pop(entry.target_id, None)


def _build_labels_by_id(entries: list[SessionTreeEntry]) -> dict[str, str]:
    labels_by_id: dict[str, str] = {}
    for entry in entries:
        _update_label_cache(labels_by_id, entry)
    return labels_by_id


def generate_entry_id(existing_ids) -> str:
    for _ in range(100):
        # The uuidv7 prefix is timestamp-derived and nearly constant between
        # calls, so short ids must come from the random tail.
        entry_id = uuidv7()[-8:]
        if entry_id not in existing_ids:
            return entry_id
    return uuidv7()


def _leaf_id_after_entry(entry: SessionTreeEntry) -> str | None:
    return entry.target_id if entry.type == "leaf" else entry.id


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class InMemorySessionStorage:
    def __init__(
        self,
        entries: list[SessionTreeEntry] | None = None,
        metadata: SessionMetadata | None = None,
    ):
        self._lock = threading.Lock()
        self._entries: list[SessionTreeEntry] = list(entries) if entries is not None else []
        self._by_id: dict[str, SessionTreeEntry] = {entry.id: entry for entry in self._entries}
        self._labels_by_id = _build_labels_by_id(self._entries)
        self._leaf_id: str | None = None
        for entry in self._entries:
            self._leaf_id = _leaf_id_after_entry(entry)
        if self._leaf_id is not None and self._leaf_id not in self._by_id:
            raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
        self._metadata = metadata if metadata is not None else SessionMetadata(id=uuidv7(), created_at=_iso_now())

    async def get_metadata(self) -> SessionMetadata:
        return self._metadata

    async def get_leaf_id(self) -> str | None:
        with self._lock:
            if self._leaf_id is not None and self._leaf_id not in self._by_id:
                raise SessionError("invalid_session", f"Entry {self._leaf_id} not found")
            return self._leaf_id

    async def set_leaf_id(self, leaf_id: str | None) -> None:
        with self._lock:
            if leaf_id is not None and leaf_id not in self._by_id:
                raise SessionError("not_found", f"Entry {leaf_id} not found")
            entry = LeafEntry(
                id=generate_entry_id(self._by_id),
                parent_id=self._leaf_id,
                timestamp=_iso_now(),
                target_id=leaf_id,
            )
            self._entries.append(entry)
            self._by_id[entry.id] = entry
            self._leaf_id = leaf_id

    async def create_entry_id(self) -> str:
        with self._lock:
            return generate_entry_id(self._by_id)

    async def append_entry(self, entry: SessionTreeEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            self._by_id[entry.id] = entry
            _update_label_cache(self._labels_by_id, entry)
            self._leaf_id = _leaf_id_after_entry(entry)

    async def get_entry(self, id: str) -> SessionTreeEntry | None:
        return self._by_id.get(id)

    async def find_entries(self, type: str) -> list[SessionTreeEntry]:
        with self._lock:
            return [entry for entry in self._entries if entry.type == type]

    async def get_label(self, id: str) -> str | None:
        return self._labels_by_id.get(id)

    async def get_session_name(self) -> str | None:
        entries = await self.find_entries("session_info")
        if not entries:
            return None
        name = entries[-1].name
        return (name.strip() or None) if name else None

    async def get_session_stats(self) -> SessionStats:
        message_count = 0
        cached_tokens = 0
        uncached_tokens = 0
        total_tokens = 0
        cost_total = 0.0
        with self._lock:
            entries = list(self._entries)
        for entry in entries:
            if entry.type == "message":
                message_count += 1
            if entry.type == "message":
                usage = entry.message.usage if getattr(entry.message, "role", None) == "assistant" else None
            elif entry.type in ("compaction", "branch_summary"):
                usage = entry.usage
            else:
                usage = None
            if usage is None or usage.cost is None:
                continue
            cached_tokens += usage.cache_read
            uncached_tokens += usage.input + usage.cache_write
            total_tokens += usage.input + usage.output + usage.cache_read + usage.cache_write
            cost_total += usage.cost.total
        return SessionStats(
            message_count=message_count,
            cached_tokens=cached_tokens,
            uncached_tokens=uncached_tokens,
            total_tokens=total_tokens,
            cost_total=cost_total,
        )

    async def get_path_to_root_or_compaction(self, leaf_id: str | None) -> list[SessionTreeEntry]:
        if leaf_id is None:
            return []
        with self._lock:
            path: list[SessionTreeEntry] = []
            stop_at_entry_id: str | None = None
            current = self._by_id.get(leaf_id)
            if current is None:
                raise SessionError("not_found", f"Entry {leaf_id} not found")
            while current is not None:
                path.insert(0, current)
                if stop_at_entry_id is not None and current.id == stop_at_entry_id:
                    break
                if current.type == "compaction":
                    if current.retained_tail is not None:
                        break
                    stop_at_entry_id = current.first_kept_entry_id
                if not current.parent_id:
                    break
                parent = self._by_id.get(current.parent_id)
                if parent is None:
                    raise SessionError("invalid_session", f"Entry {current.parent_id} not found")
                current = parent
            return path

    async def get_entries(self, options: SessionEntryCursorOptions | None = None) -> list[SessionTreeEntry]:
        with self._lock:
            start = options.after_entry_seq if options is not None and options.after_entry_seq is not None else 0
            if options is None or options.limit is None:
                return self._entries[start:]
            return self._entries[start : start + options.limit]
