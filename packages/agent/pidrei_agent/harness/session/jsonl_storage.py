"""JSONL session storage (port of pi `harness/session/jsonl-storage.ts`).

File format is pi's: a v3 `{"type":"session",...}` header line followed by one
camelCase JSON entry per line (see `serde.py`).

Concurrency (vs pi's single JS thread): appends — the file write plus the
in-memory mutation — are serialized by a tonio `Lock` (the critical section
awaits the filesystem), giving the single-writer append ordering contract;
the in-memory maps are additionally guarded by a thread lock for readers.
"""

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from tonio.colored import sync

from ..types import JsonlSessionMetadata, LeafEntry, SessionEntryCursorOptions, SessionError, SessionStats, to_error
from .memory_storage import _build_labels_by_id, _update_label_cache, generate_entry_id
from .repo_utils import get_file_system_result_or_throw
from .serde import UnknownEntry, parse_entry, serialize_entry


type JsonlEntry = Any  # SessionTreeEntry | UnknownEntry


@dataclass(slots=True, kw_only=True)
class _SessionHeader:
    id: str
    timestamp: str
    cwd: str
    parent_session: str | None = None
    metadata: dict[str, Any] | None = None


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _invalid_session(file_path: str, message: str, cause: Exception | None = None) -> SessionError:
    return SessionError("invalid_session", f"Invalid JSONL session file {file_path}: {message}", cause)


def _invalid_entry(file_path: str, line_number: int, message: str, cause: Exception | None = None) -> SessionError:
    return SessionError("invalid_entry", f"Invalid JSONL session file {file_path}: line {line_number} {message}", cause)


def _parse_header_line(line: str, file_path: str) -> _SessionHeader:
    try:
        parsed = json.loads(line)
    except ValueError as error:
        raise _invalid_session(file_path, "first line is not a valid session header", to_error(error)) from error
    if not isinstance(parsed, dict):
        raise _invalid_session(file_path, "first line is not a valid session header")
    if parsed.get("type") != "session":
        raise _invalid_session(file_path, "first line is not a valid session header")
    if parsed.get("version") != 3:
        raise _invalid_session(file_path, "unsupported session version")
    header_id = parsed.get("id")
    if not isinstance(header_id, str) or not header_id:
        raise _invalid_session(file_path, "session header is missing id")
    timestamp = parsed.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise _invalid_session(file_path, "session header is missing timestamp")
    cwd = parsed.get("cwd")
    if not isinstance(cwd, str) or not cwd:
        raise _invalid_session(file_path, "session header is missing cwd")
    parent_session = parsed.get("parentSession")
    if parent_session is not None and not isinstance(parent_session, str):
        raise _invalid_session(file_path, "session header parentSession must be a string")
    metadata = parsed.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise _invalid_session(file_path, "session header metadata must be an object")
    return _SessionHeader(id=header_id, timestamp=timestamp, cwd=cwd, parent_session=parent_session, metadata=metadata)


def _parse_entry_line(line: str, file_path: str, line_number: int) -> JsonlEntry:
    try:
        parsed = json.loads(line)
    except ValueError as error:
        raise _invalid_entry(file_path, line_number, "is not valid JSON", to_error(error)) from error
    if not isinstance(parsed, dict):
        raise _invalid_entry(file_path, line_number, "is not a valid session entry")
    if not isinstance(parsed.get("type"), str):
        raise _invalid_entry(file_path, line_number, "is missing entry type")
    entry_id = parsed.get("id")
    if not isinstance(entry_id, str) or not entry_id:
        raise _invalid_entry(file_path, line_number, "is missing entry id")
    parent_id = parsed.get("parentId")
    if parent_id is not None and not isinstance(parent_id, str):
        raise _invalid_entry(file_path, line_number, "has invalid parentId")
    timestamp = parsed.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        raise _invalid_entry(file_path, line_number, "is missing timestamp")
    if parsed.get("type") == "leaf":
        target_id = parsed.get("targetId")
        if target_id is not None and not isinstance(target_id, str):
            raise _invalid_entry(file_path, line_number, "has invalid targetId")
    return parse_entry(parsed)


def _leaf_id_after_entry(entry: JsonlEntry) -> str | None:
    return entry.target_id if entry.type == "leaf" else entry.id


def _header_to_session_metadata(header: _SessionHeader, path: str) -> JsonlSessionMetadata:
    return JsonlSessionMetadata(
        id=header.id,
        created_at=header.timestamp,
        cwd=header.cwd,
        path=path,
        parent_session_path=header.parent_session,
        metadata=header.metadata,
    )


def _serialize_header(header: _SessionHeader) -> str:
    data: dict[str, Any] = {
        "type": "session",
        "version": 3,
        "id": header.id,
        "timestamp": header.timestamp,
        "cwd": header.cwd,
    }
    if header.parent_session is not None:
        data["parentSession"] = header.parent_session
    if header.metadata is not None:
        data["metadata"] = header.metadata
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _entry_line(entry: JsonlEntry) -> str:
    return json.dumps(serialize_entry(entry), ensure_ascii=False, separators=(",", ":"))


async def load_jsonl_session_metadata(fs, file_path: str) -> JsonlSessionMetadata:
    lines = get_file_system_result_or_throw(
        await fs.read_text_lines(file_path, max_lines=1), f"Failed to read session header {file_path}"
    )
    line = lines[0] if lines else None
    if line and line.strip():
        return _header_to_session_metadata(_parse_header_line(line, file_path), file_path)
    raise _invalid_session(file_path, "missing session header")


async def _load_jsonl_storage(fs, file_path: str) -> tuple[_SessionHeader, list[JsonlEntry], str | None]:
    content = get_file_system_result_or_throw(await fs.read_text_file(file_path), f"Failed to read session {file_path}")
    lines = [(index, line) for index, line in enumerate(content.split("\n")) if line.strip()]
    if not lines:
        raise _invalid_session(file_path, "missing session header")

    header = _parse_header_line(lines[0][1], file_path)
    entries: list[JsonlEntry] = []
    leaf_id: str | None = None
    for line_index, (_, line) in enumerate(lines[1:], start=1):
        entry = _parse_entry_line(line, file_path, line_index + 1)
        entries.append(entry)
        leaf_id = _leaf_id_after_entry(entry)
    return header, entries, leaf_id


class JsonlSessionStorage:
    def __init__(self, fs, file_path: str, header: _SessionHeader, entries: list[JsonlEntry], leaf_id: str | None):
        """Internal; use `JsonlSessionStorage.open` / `JsonlSessionStorage.create`."""
        self._fs = fs
        self._file_path = file_path
        self._metadata = _header_to_session_metadata(header, file_path)
        self._entries = entries
        self._by_id: dict[str, JsonlEntry] = {entry.id: entry for entry in entries}
        self._labels_by_id = _build_labels_by_id(entries)
        self._current_leaf_id = leaf_id
        self._state_lock = threading.Lock()
        # Serializes append operations (file write + state mutation): the
        # single-writer ordering contract.
        self._append_lock = sync.Lock()

    @staticmethod
    async def open(fs, file_path: str) -> JsonlSessionStorage:
        header, entries, leaf_id = await _load_jsonl_storage(fs, file_path)
        return JsonlSessionStorage(fs, file_path, header, entries, leaf_id)

    @staticmethod
    async def create(
        fs,
        file_path: str,
        cwd: str,
        session_id: str,
        parent_session_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JsonlSessionStorage:
        header = _SessionHeader(
            id=session_id, timestamp=_iso_now(), cwd=cwd, parent_session=parent_session_path, metadata=metadata
        )
        get_file_system_result_or_throw(
            await fs.write_file(file_path, _serialize_header(header) + "\n"),
            f"Failed to create session {file_path}",
        )
        return JsonlSessionStorage(fs, file_path, header, [], None)

    async def get_metadata(self) -> JsonlSessionMetadata:
        return self._metadata

    async def get_leaf_id(self) -> str | None:
        with self._state_lock:
            if self._current_leaf_id is not None and self._current_leaf_id not in self._by_id:
                raise SessionError("invalid_session", f"Entry {self._current_leaf_id} not found")
            return self._current_leaf_id

    async def set_leaf_id(self, leaf_id: str | None) -> None:
        async with self._append_lock:
            with self._state_lock:
                if leaf_id is not None and leaf_id not in self._by_id:
                    raise SessionError("not_found", f"Entry {leaf_id} not found")
                entry = LeafEntry(
                    id=generate_entry_id(self._by_id),
                    parent_id=self._current_leaf_id,
                    timestamp=_iso_now(),
                    target_id=leaf_id,
                )
            get_file_system_result_or_throw(
                await self._fs.append_file(self._file_path, _entry_line(entry) + "\n"),
                f"Failed to append session leaf {entry.id}",
            )
            with self._state_lock:
                self._entries.append(entry)
                self._by_id[entry.id] = entry
                self._current_leaf_id = leaf_id

    async def create_entry_id(self) -> str:
        with self._state_lock:
            return generate_entry_id(self._by_id)

    async def append_entry(self, entry: JsonlEntry) -> None:
        async with self._append_lock:
            get_file_system_result_or_throw(
                await self._fs.append_file(self._file_path, _entry_line(entry) + "\n"),
                f"Failed to append session entry {entry.id}",
            )
            with self._state_lock:
                self._entries.append(entry)
                self._by_id[entry.id] = entry
                _update_label_cache(self._labels_by_id, entry)
                self._current_leaf_id = _leaf_id_after_entry(entry)

    async def get_entry(self, id: str) -> JsonlEntry | None:
        return self._by_id.get(id)

    async def find_entries(self, type: str) -> list[JsonlEntry]:
        with self._state_lock:
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
        with self._state_lock:
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

    async def get_path_to_root_or_compaction(self, leaf_id: str | None) -> list[JsonlEntry]:
        if leaf_id is None:
            return []
        with self._state_lock:
            path: list[JsonlEntry] = []
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

    async def get_entries(self, options: SessionEntryCursorOptions | None = None) -> list[JsonlEntry]:
        with self._state_lock:
            start = options.after_entry_seq if options is not None and options.after_entry_seq is not None else 0
            if options is None or options.limit is None:
                return self._entries[start:]
            return self._entries[start : start + options.limit]


__all__ = ["JsonlSessionStorage", "UnknownEntry", "load_jsonl_session_metadata"]
