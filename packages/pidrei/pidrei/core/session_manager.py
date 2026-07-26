"""Mirror of pi coding-agent src/core/session-manager.ts.

Session entries are plain camelCase dicts (pi's open JSON objects): unknown
entry types and extension fields survive load/append/save untouched. The
`message` value inside message entries (and the `usage` value on compaction/
branch_summary entries) is decoded into pidrei dataclasses in memory and
serialized back through the pidrei_agent wire codec on write, so files stay
pi-shaped byte for byte.

The manager itself uses synchronous file I/O like pi (Node *Sync calls);
mutating operations are guarded by an RLock because tonio listeners run on
real threads. The async discovery helpers (`list`, `list_all`) run their
blocking scans via `tonio.spawn_blocking` with pi's 10-way concurrency bound.
"""

import json
import math
import os
import threading
import uuid as uuid_module
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import tonio.colored as tonio

from pidrei_agent.harness.session.serde import parse_message, parse_usage, serialize_message, serialize_usage
from pidrei_ai.utils.uuid import uuidv7

from ..config import get_agent_dir as get_default_agent_dir, get_sessions_dir
from ..utils.paths import normalize_path, resolve_path
from .messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)


CURRENT_SESSION_VERSION = 3

_SESSION_READ_BUFFER_SIZE = 1024 * 1024
# Bound synchronous header discovery while allowing large cwd and custom metadata fields.
_MAX_SESSION_HEADER_SCAN_BYTES = 1024 * 1024
_SESSION_HEADER_READ_BUFFER_SIZE = 4096

_MAX_CONCURRENT_SESSION_INFO_LOADS = 10

# leaf_id sentinel: `...` = "use current leaf / last entry", None = pi's explicit null.
_UNSET = ...


@dataclass(slots=True)
class SessionContextModel:
    provider: str
    model_id: str


@dataclass(slots=True)
class SessionContext:
    messages: list[Any]
    thinking_level: str
    model: SessionContextModel | None


@dataclass(slots=True)
class SessionTreeNode:
    """Tree node for get_tree() - defensive copy of session structure."""

    entry: dict[str, Any]
    children: list[SessionTreeNode]
    # Resolved label for this entry, if any.
    label: str | None = None
    # Timestamp of the latest label change for this entry, if any.
    label_timestamp: str | None = None


@dataclass(slots=True)
class SessionInfo:
    path: str
    id: str
    # Working directory where the session was started. Empty string for old sessions.
    cwd: str
    created: datetime
    modified: datetime
    message_count: int
    first_message: str
    all_messages_text: str
    # User-defined display name from session_info entries.
    name: str | None = None
    # Path to the parent session (if this session was forked).
    parent_session_path: str | None = None


class SessionHeaderScanLimitError(Exception):
    def __init__(self, file_path: str):
        super().__init__(f"Session header exceeds {_MAX_SESSION_HEADER_SCAN_BYTES}-byte scan limit: {file_path}")
        self.name = "SessionHeaderScanLimitError"


def _iso_now() -> str:
    # Millisecond precision like JS Date.toISOString(): session file names derive
    # from this ("<timestamp with [:.] -> ->_<id>.jsonl").
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _iso_to_epoch_ms(timestamp: Any) -> float:
    if not isinstance(timestamp, str):
        return float("nan")
    try:
        return datetime.fromisoformat(timestamp).timestamp() * 1000
    except ValueError:
        return float("nan")


def _dump_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _create_session_id() -> str:
    return uuidv7()


def assert_valid_session_id(session_id: str) -> None:
    import re

    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", session_id or ""):
        raise Exception(
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', "
            "and start and end with an alphanumeric character"
        )


def _generate_id(existing) -> str:
    """Generate a unique short ID (8 hex chars, collision-checked)."""
    for _ in range(100):
        new_id = str(uuid_module.uuid4())[:8]
        if new_id not in existing:
            return new_id
    # Fallback to full UUID if somehow we have collisions
    return str(uuid_module.uuid4())


# ---------------------------------------------------------------------------
# Wire <-> in-memory entry conversion
# ---------------------------------------------------------------------------


def _decode_entry(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("type") == "message" and isinstance(entry.get("message"), dict):
        entry["message"] = parse_message(entry["message"])
    elif entry.get("type") in ("compaction", "branch_summary") and isinstance(entry.get("usage"), dict):
        entry["usage"] = parse_usage(entry["usage"])
    return entry


def _entry_to_wire(entry: dict[str, Any]) -> dict[str, Any]:
    if entry.get("type") == "message" and entry.get("message") is not None and not isinstance(entry["message"], dict):
        wire = dict(entry)
        wire["message"] = serialize_message(entry["message"])
        return wire
    if entry.get("type") in ("compaction", "branch_summary") and entry.get("usage") is not None:
        wire = dict(entry)
        wire["usage"] = serialize_usage(entry["usage"])
        return wire
    return entry


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


def _migrate_v1_to_v2(entries: list[dict[str, Any]]) -> None:
    """Migrate v1 -> v2: add id/parentId tree structure. Mutates in place."""
    ids: set[str] = set()
    prev_id: str | None = None

    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 2
            continue

        entry["id"] = _generate_id(ids)
        ids.add(entry["id"])
        entry["parentId"] = prev_id
        prev_id = entry["id"]

        # Convert firstKeptEntryIndex to firstKeptEntryId for compaction
        if entry.get("type") == "compaction" and isinstance(entry.get("firstKeptEntryIndex"), int):
            index = entry["firstKeptEntryIndex"]
            target_entry = entries[index] if 0 <= index < len(entries) else None
            if target_entry and target_entry.get("type") != "session":
                entry["firstKeptEntryId"] = target_entry["id"]
            del entry["firstKeptEntryIndex"]


def _migrate_v2_to_v3(entries: list[dict[str, Any]]) -> None:
    """Migrate v2 -> v3: rename hookMessage role to custom. Mutates in place."""
    for entry in entries:
        if entry.get("type") == "session":
            entry["version"] = 3
            continue

        if entry.get("type") == "message":
            message = entry.get("message")
            if isinstance(message, dict) and message.get("role") == "hookMessage":
                message["role"] = "custom"


def _migrate_to_current_version(entries: list[dict[str, Any]]) -> bool:
    """Run all necessary migrations to bring entries to current version.
    Mutates entries in place. Returns True if any migration was applied."""
    header = next((entry for entry in entries if entry.get("type") == "session"), None)
    version = header.get("version") if header and header.get("version") is not None else 1

    if version >= CURRENT_SESSION_VERSION:
        return False

    if version < 2:
        _migrate_v1_to_v2(entries)
    if version < 3:
        _migrate_v2_to_v3(entries)

    return True


def migrate_session_entries(entries: list[dict[str, Any]]) -> None:
    """Exported for testing. Operates on raw wire dicts."""
    _migrate_to_current_version(entries)


def parse_session_entries(content: str) -> list[dict[str, Any]]:
    """Exported for compaction tests. Returns decoded entries."""
    entries: list[dict[str, Any]] = []
    for line in content.strip().split("\n"):
        entry = _parse_session_entry_line(line)
        if entry is not None:
            entries.append(_decode_entry(entry))
    return entries


def _parse_session_entry_line(line: str) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        parsed = json.loads(line)
    except Exception:
        # Skip malformed lines
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------


def get_latest_compaction_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in reversed(entries):
        if entry.get("type") == "compaction":
            return entry
    return None


def _build_entry_index(
    entries: list[dict[str, Any]], by_id: dict[str, dict[str, Any]] | None = None
) -> dict[str, dict[str, Any]]:
    if by_id is not None:
        return by_id
    return {entry["id"]: entry for entry in entries}


def _build_session_path(
    entries: list[dict[str, Any]],
    leaf_id: Any = _UNSET,
    by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    index = _build_entry_index(entries, by_id)
    leaf: dict[str, Any] | None = None
    if leaf_id is None:
        return []
    if leaf_id is not _UNSET:
        leaf = index.get(leaf_id)
    if leaf is None:
        leaf = entries[-1] if entries else None
    if leaf is None:
        return []

    path: list[dict[str, Any]] = []
    current: dict[str, Any] | None = leaf
    while current is not None:
        path.append(current)
        parent_id = current.get("parentId")
        current = index.get(parent_id) if parent_id else None
    path.reverse()
    return path


def _get_session_context_settings(path: list[dict[str, Any]]) -> tuple[str, SessionContextModel | None]:
    thinking_level = "off"
    model: SessionContextModel | None = None

    for entry in path:
        if entry.get("type") == "thinking_level_change":
            thinking_level = entry.get("thinkingLevel")
        elif entry.get("type") == "model_change":
            model = SessionContextModel(provider=entry.get("provider"), model_id=entry.get("modelId"))
        elif entry.get("type") == "message" and getattr(entry.get("message"), "role", None) == "assistant":
            message = entry["message"]
            model = SessionContextModel(provider=message.provider, model_id=message.model)

    return thinking_level, model


def session_entry_to_context_messages(entry: dict[str, Any]) -> list[Any]:
    """Project one selected session entry into LLM/runtime messages.
    Plain custom entries are display/state entries and do not participate in context."""
    entry_type = entry.get("type")
    if entry_type == "message":
        message = entry.get("message")
        # Session files are parsed without validation; old versions, forks, or
        # hand-edited files can contain messages with null/missing content.
        if getattr(message, "role", None) in ("user", "assistant", "toolResult") and message.content is None:
            import dataclasses

            return [dataclasses.replace(message, content=[])]
        return [message]
    if entry_type == "custom_message":
        content = entry.get("content")
        return [
            create_custom_message(
                entry.get("customType"),
                content if content is not None else [],
                entry.get("display"),
                entry.get("details"),
                entry.get("timestamp"),
            )
        ]
    if entry_type == "branch_summary" and entry.get("summary"):
        return [create_branch_summary_message(entry["summary"], entry.get("fromId"), entry.get("timestamp"))]
    if entry_type == "compaction":
        return [
            create_compaction_summary_message(entry.get("summary"), entry.get("tokensBefore"), entry.get("timestamp"))
        ]
    return []


def build_context_entries(
    entries: list[dict[str, Any]],
    leaf_id: Any = _UNSET,
    by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the active, compaction-aware session entry list.

    This follows the current leaf path. If the path contains compaction entries,
    the latest compaction is represented by the compaction entry itself, followed
    by the kept entries starting at firstKeptEntryId and all entries after the
    compaction entry. Older summarized entries are omitted."""
    path = _build_session_path(entries, leaf_id, by_id)
    compaction: dict[str, Any] | None = None

    for entry in path:
        if entry.get("type") == "compaction":
            compaction = entry

    if compaction is None:
        return path

    compaction_idx = next((i for i, entry in enumerate(path) if entry.get("id") == compaction["id"]), -1)
    if compaction_idx < 0:
        return path

    context_entries: list[dict[str, Any]] = [compaction]
    found_first_kept = False
    for i in range(compaction_idx):
        entry = path[i]
        if entry.get("id") == compaction.get("firstKeptEntryId"):
            found_first_kept = True
        if found_first_kept:
            context_entries.append(entry)
    context_entries.extend(path[compaction_idx + 1 :])
    return context_entries


def build_session_context(
    entries: list[dict[str, Any]],
    leaf_id: Any = _UNSET,
    by_id: dict[str, dict[str, Any]] | None = None,
) -> SessionContext:
    """Build the session context from entries using tree traversal.
    If leaf_id is provided, walks from that entry to root.
    Handles compaction and branch summaries along the path."""
    path = _build_session_path(entries, leaf_id, by_id)
    thinking_level, model = _get_session_context_settings(path)
    messages = [
        message
        for entry in build_context_entries(entries, leaf_id, by_id)
        for message in session_entry_to_context_messages(entry)
    ]
    return SessionContext(messages=messages, thinking_level=thinking_level, model=model)


# ---------------------------------------------------------------------------
# Session directory / discovery
# ---------------------------------------------------------------------------


def _get_default_session_dir_path(cwd: str, agent_dir: str | None = None) -> str:
    """Compute the default session directory for a cwd.
    Encodes cwd into a safe directory name under ~/.pidrei/agent/sessions/."""
    import re

    resolved_cwd = resolve_path(cwd)
    resolved_agent_dir = resolve_path(agent_dir if agent_dir is not None else get_default_agent_dir())
    safe_path = "--" + re.sub(r"[/\\:]", "-", re.sub(r"^[/\\]", "", resolved_cwd)) + "--"
    return os.path.join(resolved_agent_dir, "sessions", safe_path)


def get_default_session_dir(cwd: str, agent_dir: str | None = None) -> str:
    session_dir = _get_default_session_dir_path(cwd, agent_dir)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir, exist_ok=True)
    return session_dir


def load_entries_from_file(file_path: str) -> list[dict[str, Any]]:
    """Exported for testing. Returns decoded entries (header included)."""
    entries = _load_wire_entries_from_file(file_path)
    return [_decode_entry(entry) for entry in entries]


def _load_wire_entries_from_file(file_path: str) -> list[dict[str, Any]]:
    resolved_file_path = normalize_path(file_path)
    if not os.path.exists(resolved_file_path):
        return []

    import codecs

    # Chunked read with an incremental decoder: session files can be far larger
    # than what should ever be materialized as a single string (pi streams with
    # a 1 MiB buffer for the same reason).
    entries: list[dict[str, Any]] = []
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    with open(resolved_file_path, "rb") as handle:
        while True:
            data = handle.read(_SESSION_READ_BUFFER_SIZE)
            if not data:
                break
            pending += decoder.decode(data)
            line_start = 0
            newline_index = pending.find("\n", line_start)
            while newline_index != -1:
                entry = _parse_session_entry_line(pending[line_start:newline_index])
                if entry is not None:
                    entries.append(entry)
                line_start = newline_index + 1
                newline_index = pending.find("\n", line_start)
            pending = pending[line_start:]

    pending += decoder.decode(b"", True)
    final_entry = _parse_session_entry_line(pending)
    if final_entry is not None:
        entries.append(final_entry)

    # Validate session header
    if not entries:
        return entries
    header = entries[0]
    if header.get("type") != "session" or not isinstance(header.get("id"), str):
        return []

    return entries


def _parse_session_header_candidate(line: str) -> Any:
    """Inspect a physical line while searching for the first parsed session entry.
    Blank and malformed lines are skipped to match load_entries_from_file().
    Returns _UNSET to keep scanning, None for a parsed non-header entry, or the header."""
    if not line.strip():
        return _UNSET
    entry = _parse_session_entry_line(line)
    if entry is None:
        return _UNSET
    if entry.get("type") != "session" or not isinstance(entry.get("id"), str):
        return None
    return entry


def _read_session_header(file_path: str) -> dict[str, Any] | None:
    import codecs

    with open(file_path, "rb") as handle:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        line_chunks: list[str] = []
        scanned_bytes = 0

        while scanned_bytes < _MAX_SESSION_HEADER_SCAN_BYTES:
            read_length = min(_SESSION_HEADER_READ_BUFFER_SIZE, _MAX_SESSION_HEADER_SCAN_BYTES - scanned_bytes)
            data = handle.read(read_length)
            if not data:
                line_chunks.append(decoder.decode(b"", True))
                candidate = _parse_session_header_candidate("".join(line_chunks))
                return candidate if candidate is not _UNSET else None
            scanned_bytes += len(data)

            chunk = decoder.decode(data)
            line_start = 0
            newline_index = chunk.find("\n", line_start)
            while newline_index != -1:
                line_chunks.append(chunk[line_start:newline_index])
                candidate = _parse_session_header_candidate("".join(line_chunks))
                if candidate is not _UNSET:
                    return candidate
                line_chunks.clear()
                line_start = newline_index + 1
                newline_index = chunk.find("\n", line_start)
            line_chunks.append(chunk[line_start:])

        # Probe for EOF so a final header without a newline is allowed when it ends
        # exactly at the scan limit. Any additional byte exceeds the bounded scan.
        if not handle.read(1):
            line_chunks.append(decoder.decode(b"", True))
            candidate = _parse_session_header_candidate("".join(line_chunks))
            return candidate if candidate is not _UNSET else None
        raise SessionHeaderScanLimitError(file_path)


def _read_session_header_for_discovery(file_path: str) -> dict[str, Any] | None:
    try:
        return _read_session_header(file_path)
    except Exception:
        # Discovery is best-effort: unreadable or oversized files are not sessions,
        # and one corrupt file must not prevent other sessions from being found.
        return None


def _get_session_header_cwd(header: dict[str, Any]) -> str | None:
    cwd = header.get("cwd")
    return cwd if isinstance(cwd, str) else None


def _session_cwd_matches(cwd: str | None, resolved_cwd: str) -> bool:
    return cwd is not None and cwd != "" and resolve_path(cwd) == resolved_cwd


def find_most_recent_session(session_dir: str, cwd: str | None = None) -> str | None:
    """Exported for testing."""
    resolved_session_dir = normalize_path(session_dir)
    resolved_cwd = resolve_path(cwd) if cwd else None
    try:
        candidates = []
        for name in os.listdir(resolved_session_dir):
            if not name.endswith(".jsonl"):
                continue
            path = os.path.join(resolved_session_dir, name)
            header = _read_session_header_for_discovery(path)
            if header is None:
                continue
            if resolved_cwd and not _session_cwd_matches(_get_session_header_cwd(header), resolved_cwd):
                continue
            candidates.append((path, os.stat(path).st_mtime))
        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates[0][0] if candidates else None
    except Exception:
        # Directory access and stat races make recent-session discovery unavailable.
        return None


# ---------------------------------------------------------------------------
# Session info scanning (raw wire dicts; used by list()/list_all())
# ---------------------------------------------------------------------------


def _extract_text_content_from_wire(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _get_message_activity_time(entry: dict[str, Any]) -> float | None:
    message = entry.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("role"), str) or "content" not in message:
        return None
    if message.get("role") not in ("user", "assistant"):
        return None

    msg_timestamp = message.get("timestamp")
    if isinstance(msg_timestamp, (int, float)) and not isinstance(msg_timestamp, bool):
        return float(msg_timestamp)

    parsed = _iso_to_epoch_ms(entry.get("timestamp"))
    return None if math.isnan(parsed) else parsed


def _build_session_info(file_path: str) -> SessionInfo | None:
    try:
        stat_result = os.stat(file_path)
        header: dict[str, Any] | None = None
        message_count = 0
        first_message = ""
        all_messages: list[str] = []
        name: str | None = None
        last_activity_time: float | None = None

        with open(file_path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                entry = _parse_session_entry_line(line.rstrip("\n").rstrip("\r"))
                if entry is None:
                    continue

                if header is None:
                    if entry.get("type") != "session":
                        return None
                    header = entry
                    continue

                # Extract session name (use latest, including explicit clears)
                if entry.get("type") == "session_info":
                    raw_name = entry.get("name")
                    name = (raw_name.strip() if isinstance(raw_name, str) else "") or None

                if entry.get("type") != "message":
                    continue
                message_count += 1

                activity_time = _get_message_activity_time(entry)
                if activity_time is not None:
                    last_activity_time = max(last_activity_time if last_activity_time is not None else 0, activity_time)

                message = entry.get("message")
                if not isinstance(message, dict) or "content" not in message:
                    continue
                if message.get("role") not in ("user", "assistant"):
                    continue

                text_content = _extract_text_content_from_wire(message.get("content"))
                if not text_content:
                    continue

                all_messages.append(text_content)
                if not first_message and message.get("role") == "user":
                    first_message = text_content

        if header is None:
            return None

        cwd = header.get("cwd") if isinstance(header.get("cwd"), str) else ""
        parent_session_path = header.get("parentSession")
        header_time = _iso_to_epoch_ms(header.get("timestamp"))
        if last_activity_time is not None and last_activity_time > 0:
            modified = datetime.fromtimestamp(last_activity_time / 1000, tz=UTC)
        elif not math.isnan(header_time):
            modified = datetime.fromtimestamp(header_time / 1000, tz=UTC)
        else:
            modified = datetime.fromtimestamp(stat_result.st_mtime, tz=UTC)

        created_time = _iso_to_epoch_ms(header.get("timestamp"))
        created = (
            datetime.fromtimestamp(created_time / 1000, tz=UTC)
            if not math.isnan(created_time)
            else datetime.fromtimestamp(0, tz=UTC)
        )

        return SessionInfo(
            path=file_path,
            id=header.get("id"),
            cwd=cwd,
            name=name,
            parent_session_path=parent_session_path,
            created=created,
            modified=modified,
            message_count=message_count,
            first_message=first_message or "(no messages)",
            all_messages_text=" ".join(all_messages),
        )
    except Exception:
        return None


async def _build_session_infos_with_concurrency(
    files: list[str], on_loaded: Callable[[], None]
) -> list[SessionInfo | None]:
    results: list[SessionInfo | None] = [None] * len(files)
    next_index = 0
    guard = threading.Lock()

    async def worker() -> None:
        nonlocal next_index
        while True:
            with guard:
                index = next_index
                next_index += 1
            if index >= len(files):
                return
            try:
                results[index] = await tonio.spawn_blocking(_build_session_info, files[index])
            except Exception:
                results[index] = None
            finally:
                on_loaded()

    worker_count = min(_MAX_CONCURRENT_SESSION_INFO_LOADS, len(files))
    if worker_count > 0:
        await tonio.spawn(*(worker() for _ in range(worker_count)))
    return results


async def _list_sessions_from_dir(
    directory: str,
    on_progress: Callable[[int, int], None] | None = None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> list[SessionInfo]:
    sessions: list[SessionInfo] = []
    if not os.path.exists(directory):
        return sessions

    try:
        files = [
            os.path.join(directory, name)
            for name in await tonio.spawn_blocking(os.listdir, directory)
            if name.endswith(".jsonl")
        ]
        total = progress_total if progress_total is not None else len(files)

        loaded = 0
        loaded_guard = threading.Lock()

        def on_loaded() -> None:
            nonlocal loaded
            with loaded_guard:
                loaded += 1
                current = loaded
            if on_progress is not None:
                on_progress(progress_offset + current, total)

        results = await _build_session_infos_with_concurrency(files, on_loaded)
        sessions.extend(info for info in results if info is not None)
    except Exception:
        # Return empty list on error
        pass

    return sessions


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


class SessionManager:
    """Manages conversation sessions as append-only trees stored in JSONL files.

    Each session entry has an id and parentId forming a tree structure. The
    "leaf" pointer tracks the current position. Appending creates a child of the
    current leaf. Branching moves the leaf to an earlier entry, allowing new
    branches without modifying history.

    Use build_session_context() to get the resolved message list for the LLM,
    which handles compaction summaries and follows the path from root to leaf.
    """

    def __init__(
        self,
        cwd: str,
        session_dir: str,
        session_file: str | None,
        persist: bool,
        new_session_options: dict[str, Any] | None = None,
        preloaded_file_entries: list[dict[str, Any]] | None = None,
        *,
        _internal: bool = False,
    ):
        if not _internal:
            raise Exception("Use SessionManager.create/open/continue_recent/in_memory/fork_from")
        self._lock = threading.RLock()
        self._session_id: str = ""
        self._session_file: str | None = None
        self._cwd = resolve_path(cwd)
        self._session_dir = normalize_path(session_dir) if session_dir else ""
        self._persist = persist
        self._flushed = False
        self._file_entries: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._labels_by_id: dict[str, str] = {}
        self._label_timestamps_by_id: dict[str, str] = {}
        self._leaf_id: str | None = None

        if persist and self._session_dir and not os.path.exists(self._session_dir):
            os.makedirs(self._session_dir, exist_ok=True)

        if session_file:
            self._set_session_file(session_file, preloaded_file_entries)
        else:
            self.new_session(new_session_options)

    # -- session file handling -------------------------------------------------

    def set_session_file(self, session_file: str) -> None:
        """Switch to a different session file (used for resume and branching)."""
        with self._lock:
            self._set_session_file(session_file)

    def _set_session_file(self, session_file: str, preloaded_file_entries: list[dict[str, Any]] | None = None) -> None:
        self._session_file = resolve_path(session_file)
        if os.path.exists(self._session_file):
            wire_entries = (
                preloaded_file_entries
                if preloaded_file_entries is not None
                else _load_wire_entries_from_file(self._session_file)
            )

            # If file was empty, initialize it with a valid session header. If it was
            # non-empty but did not parse as a pi session, fail without modifying it.
            if not wire_entries:
                explicit_path = self._session_file
                if os.stat(explicit_path).st_size > 0:
                    raise Exception(f"Session file is not a valid pidrei session: {explicit_path}")
                self.new_session()
                self._session_file = explicit_path
                self._rewrite_file()
                self._flushed = True
                return

            header = next((entry for entry in wire_entries if entry.get("type") == "session"), None)
            self._session_id = header.get("id") if header and header.get("id") else _create_session_id()

            migrated = _migrate_to_current_version(wire_entries)
            self._file_entries = [
                entry if entry.get("type") == "session" else _decode_entry(entry) for entry in wire_entries
            ]
            if migrated:
                self._rewrite_file()

            self._build_index()
            self._flushed = True
        else:
            explicit_path = self._session_file
            self.new_session()
            self._session_file = explicit_path  # preserve explicit path from --session flag

    def new_session(self, options: dict[str, Any] | None = None) -> str | None:
        with self._lock:
            session_id = (options or {}).get("id")
            if session_id is not None:
                assert_valid_session_id(session_id)
            self._session_id = session_id if session_id is not None else _create_session_id()
            timestamp = _iso_now()
            header: dict[str, Any] = {
                "type": "session",
                "version": CURRENT_SESSION_VERSION,
                "id": self._session_id,
                "timestamp": timestamp,
                "cwd": self._cwd,
            }
            parent_session = (options or {}).get("parentSession")
            if parent_session is not None:
                header["parentSession"] = parent_session
            self._file_entries = [header]
            self._by_id.clear()
            self._labels_by_id.clear()
            self._label_timestamps_by_id.clear()
            self._leaf_id = None
            self._flushed = False

            if self._persist:
                file_timestamp = timestamp.replace(":", "-").replace(".", "-")
                self._session_file = os.path.join(self.get_session_dir(), f"{file_timestamp}_{self._session_id}.jsonl")
            return self._session_file

    def _build_index(self) -> None:
        self._by_id.clear()
        self._labels_by_id.clear()
        self._label_timestamps_by_id.clear()
        self._leaf_id = None
        for entry in self._file_entries:
            if entry.get("type") == "session":
                continue
            self._by_id[entry["id"]] = entry
            self._leaf_id = entry["id"]
            if entry.get("type") == "label":
                if entry.get("label"):
                    self._labels_by_id[entry["targetId"]] = entry["label"]
                    self._label_timestamps_by_id[entry["targetId"]] = entry["timestamp"]
                else:
                    self._labels_by_id.pop(entry["targetId"], None)
                    self._label_timestamps_by_id.pop(entry["targetId"], None)

    def _rewrite_file(self) -> None:
        if not self._persist or not self._session_file:
            return
        with open(self._session_file, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(_dump_json(_entry_to_wire(entry)) + "\n" for entry in self._file_entries)

    def is_persisted(self) -> bool:
        return self._persist

    def get_cwd(self) -> str:
        return self._cwd

    def get_session_dir(self) -> str:
        return self._session_dir

    def uses_default_session_dir(self) -> bool:
        return self._session_dir == _get_default_session_dir_path(self._cwd)

    def get_session_id(self) -> str:
        return self._session_id

    def get_session_file(self) -> str | None:
        return self._session_file

    def _persist_entry(self, entry: dict[str, Any]) -> None:
        if not self._persist or not self._session_file:
            return

        has_assistant = any(
            e.get("type") == "message" and getattr(e.get("message"), "role", None) == "assistant"
            for e in self._file_entries
        )
        if not has_assistant:
            if self._flushed:
                with open(self._session_file, "a", encoding="utf-8", newline="") as handle:
                    handle.write(_dump_json(_entry_to_wire(entry)) + "\n")
            else:
                # Mark as not flushed so when assistant arrives, all entries get written
                self._flushed = False
            return

        if not self._flushed:
            with open(self._session_file, "x", encoding="utf-8", newline="") as handle:
                handle.writelines(_dump_json(_entry_to_wire(e)) + "\n" for e in self._file_entries)
            self._flushed = True
        else:
            with open(self._session_file, "a", encoding="utf-8", newline="") as handle:
                handle.write(_dump_json(_entry_to_wire(entry)) + "\n")

    def _append_entry(self, entry: dict[str, Any]) -> None:
        self._file_entries.append(entry)
        self._by_id[entry["id"]] = entry
        self._leaf_id = entry["id"]
        self._persist_entry(entry)

    def _new_entry_base(self, entry_type: str) -> dict[str, Any]:
        return {
            "type": entry_type,
            "id": _generate_id(self._by_id),
            "parentId": self._leaf_id,
            "timestamp": _iso_now(),
        }

    # -- appends -----------------------------------------------------------------

    def append_message(self, message: Any) -> str:
        """Append a message as child of current leaf, then advance leaf. Returns entry id.
        Does not allow writing CompactionSummaryMessage and BranchSummaryMessage directly:
        those are top-level entries appended via append_compaction()/branch_with_summary()."""
        with self._lock:
            entry = self._new_entry_base("message")
            entry["message"] = message
            self._append_entry(entry)
            return entry["id"]

    def append_thinking_level_change(self, thinking_level: str) -> str:
        with self._lock:
            entry = self._new_entry_base("thinking_level_change")
            entry["thinkingLevel"] = thinking_level
            self._append_entry(entry)
            return entry["id"]

    def append_model_change(self, provider: str, model_id: str) -> str:
        with self._lock:
            entry = self._new_entry_base("model_change")
            entry["provider"] = provider
            entry["modelId"] = model_id
            self._append_entry(entry)
            return entry["id"]

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: Any = None,
        from_hook: bool | None = None,
        usage: Any = None,
    ) -> str:
        with self._lock:
            entry = self._new_entry_base("compaction")
            entry["summary"] = summary
            entry["firstKeptEntryId"] = first_kept_entry_id
            entry["tokensBefore"] = tokens_before
            if details is not None:
                entry["details"] = details
            if usage is not None:
                entry["usage"] = usage
            if from_hook is not None:
                entry["fromHook"] = from_hook
            self._append_entry(entry)
            return entry["id"]

    def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        """Append a custom entry (for extensions) as child of current leaf."""
        with self._lock:
            entry = self._new_entry_base("custom")
            entry["customType"] = custom_type
            if data is not None:
                entry["data"] = data
            self._append_entry(entry)
            return entry["id"]

    def append_session_info(self, name: str) -> str:
        """Append a session info entry (e.g., display name). Returns entry id."""
        import re

        with self._lock:
            sanitized_name = re.sub(r"[\r\n]+", " ", name).strip()
            entry = self._new_entry_base("session_info")
            entry["name"] = sanitized_name
            self._append_entry(entry)
            return entry["id"]

    def get_session_name(self) -> str | None:
        """Get the current session name from the latest session_info entry, if any."""
        with self._lock:
            # Walk entries in reverse to find the latest session_info entry.
            # Empty names explicitly clear the session title.
            for entry in reversed(self.get_entries()):
                if entry.get("type") == "session_info":
                    raw_name = entry.get("name")
                    return (raw_name.strip() if isinstance(raw_name, str) else "") or None
            return None

    def append_custom_message_entry(self, custom_type: str, content: Any, display: bool, details: Any = None) -> str:
        """Append a custom message entry (for extensions) that participates in LLM context."""
        with self._lock:
            entry = self._new_entry_base("custom_message")
            entry["customType"] = custom_type
            entry["content"] = content
            entry["display"] = display
            if details is not None:
                entry["details"] = details
            self._append_entry(entry)
            return entry["id"]

    # -- tree traversal ----------------------------------------------------------

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def get_leaf_entry(self) -> dict[str, Any] | None:
        with self._lock:
            return self._by_id.get(self._leaf_id) if self._leaf_id else None

    def get_entry(self, entry_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._by_id.get(entry_id)

    def get_children(self, parent_id: str) -> list[dict[str, Any]]:
        """Get all direct children of an entry."""
        with self._lock:
            return [entry for entry in self._by_id.values() if entry.get("parentId") == parent_id]

    def get_label(self, entry_id: str) -> str | None:
        """Get the label for an entry, if any."""
        with self._lock:
            return self._labels_by_id.get(entry_id)

    def append_label_change(self, target_id: str, label: str | None) -> str:
        """Set or clear a label on an entry. Labels are user-defined markers for
        bookmarking/navigation. Pass None or empty string to clear the label."""
        with self._lock:
            if target_id not in self._by_id:
                raise Exception(f"Entry {target_id} not found")
            entry = self._new_entry_base("label")
            entry["targetId"] = target_id
            if label is not None:
                entry["label"] = label
            self._append_entry(entry)
            if label:
                self._labels_by_id[target_id] = label
                self._label_timestamps_by_id[target_id] = entry["timestamp"]
            else:
                self._labels_by_id.pop(target_id, None)
                self._label_timestamps_by_id.pop(target_id, None)
            return entry["id"]

    def get_branch(self, from_id: str | None = None) -> list[dict[str, Any]]:
        """Walk from entry to root, returning all entries in path order.
        Includes all entry types (messages, compaction, model changes, etc.)."""
        with self._lock:
            path: list[dict[str, Any]] = []
            start_id = from_id if from_id is not None else self._leaf_id
            current = self._by_id.get(start_id) if start_id else None
            while current is not None:
                path.append(current)
                parent_id = current.get("parentId")
                current = self._by_id.get(parent_id) if parent_id else None
            path.reverse()
            return path

    def build_context_entries(self) -> list[dict[str, Any]]:
        """Build the active, compaction-aware entry list for context/rendering."""
        with self._lock:
            return build_context_entries(self.get_entries(), self._leaf_id, self._by_id)

    def build_session_context(self) -> SessionContext:
        """Build the session context (what gets sent to the LLM)."""
        with self._lock:
            return build_session_context(self.get_entries(), self._leaf_id, self._by_id)

    def get_header(self) -> dict[str, Any] | None:
        with self._lock:
            return next((entry for entry in self._file_entries if entry.get("type") == "session"), None)

    def get_entries(self) -> list[dict[str, Any]]:
        """Get all session entries (excludes header). Returns a shallow copy."""
        with self._lock:
            return [entry for entry in self._file_entries if entry.get("type") != "session"]

    def get_tree(self) -> list[SessionTreeNode]:
        """Get the session as a tree structure. A well-formed session has exactly one
        root (first entry with parentId null). Orphaned entries are also roots."""
        with self._lock:
            entries = self.get_entries()
            node_map: dict[str, SessionTreeNode] = {}
            roots: list[SessionTreeNode] = []

            for entry in entries:
                node_map[entry["id"]] = SessionTreeNode(
                    entry=entry,
                    children=[],
                    label=self._labels_by_id.get(entry["id"]),
                    label_timestamp=self._label_timestamps_by_id.get(entry["id"]),
                )

            for entry in entries:
                node = node_map[entry["id"]]
                parent_id = entry.get("parentId")
                if parent_id is None or parent_id == entry["id"]:
                    roots.append(node)
                else:
                    parent = node_map.get(parent_id)
                    if parent is not None:
                        parent.children.append(node)
                    else:
                        # Orphan - treat as root
                        roots.append(node)

            # Sort children by timestamp (oldest first, newest at bottom).
            # Iterative to avoid recursion limits on deep trees.
            stack = list(roots)
            while stack:
                node = stack.pop()
                node.children.sort(key=lambda child: _iso_to_epoch_ms(child.entry.get("timestamp")))
                stack.extend(node.children)

            return roots

    # -- branching ---------------------------------------------------------------

    def branch(self, branch_from_id: str) -> None:
        """Start a new branch from an earlier entry. Moves the leaf pointer only."""
        with self._lock:
            if branch_from_id not in self._by_id:
                raise Exception(f"Entry {branch_from_id} not found")
            self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        """Reset the leaf pointer to null (before any entries)."""
        with self._lock:
            self._leaf_id = None

    def branch_with_summary(
        self,
        branch_from_id: str | None,
        summary: str,
        details: Any = None,
        from_hook: bool | None = None,
        usage: Any = None,
    ) -> str:
        """Start a new branch with a summary of the abandoned path."""
        with self._lock:
            if branch_from_id is not None and branch_from_id not in self._by_id:
                raise Exception(f"Entry {branch_from_id} not found")
            self._leaf_id = branch_from_id
            entry: dict[str, Any] = {
                "type": "branch_summary",
                "id": _generate_id(self._by_id),
                "parentId": branch_from_id,
                "timestamp": _iso_now(),
                "fromId": branch_from_id if branch_from_id is not None else "root",
                "summary": summary,
            }
            if details is not None:
                entry["details"] = details
            if usage is not None:
                entry["usage"] = usage
            if from_hook is not None:
                entry["fromHook"] = from_hook
            self._append_entry(entry)
            return entry["id"]

    def create_branched_session(self, leaf_id: str) -> str | None:
        """Create a new session file containing only the path from root to the given
        leaf. Returns the new session file path, or None if not persisting."""
        with self._lock:
            previous_session_file = self._session_file
            path = self.get_branch(leaf_id)
            if not path:
                raise Exception(f"Entry {leaf_id} not found")

            # Filter out label entries from the path - recreated from the resolved map.
            # Because labels are real tree entries, later entries can be children of
            # labels; removing labels requires re-chaining the retained path.
            path_without_labels: list[dict[str, Any]] = []
            path_parent_id: str | None = None
            for entry in path:
                if entry.get("type") == "label":
                    continue
                copied = dict(entry)
                copied["parentId"] = path_parent_id
                path_without_labels.append(copied)
                path_parent_id = entry["id"]

            new_session_id = _create_session_id()
            timestamp = _iso_now()
            file_timestamp = timestamp.replace(":", "-").replace(".", "-")
            new_session_file = os.path.join(self.get_session_dir(), f"{file_timestamp}_{new_session_id}.jsonl")

            header: dict[str, Any] = {
                "type": "session",
                "version": CURRENT_SESSION_VERSION,
                "id": new_session_id,
                "timestamp": timestamp,
                "cwd": self._cwd,
            }
            if self._persist and previous_session_file is not None:
                header["parentSession"] = previous_session_file

            # Collect labels for entries in the path
            path_entry_ids = {entry["id"] for entry in path_without_labels}
            labels_to_write = [
                (target_id, label, self._label_timestamps_by_id[target_id])
                for target_id, label in self._labels_by_id.items()
                if target_id in path_entry_ids
            ]

            label_entries: list[dict[str, Any]] = []
            parent_id = path_without_labels[-1]["id"] if path_without_labels else None
            for target_id, label, label_timestamp in labels_to_write:
                label_entry: dict[str, Any] = {
                    "type": "label",
                    "id": _generate_id(path_entry_ids),
                    "parentId": parent_id,
                    "timestamp": label_timestamp,
                    "targetId": target_id,
                    "label": label,
                }
                path_entry_ids.add(label_entry["id"])
                label_entries.append(label_entry)
                parent_id = label_entry["id"]

            self._file_entries = [header, *path_without_labels, *label_entries]
            self._session_id = new_session_id
            if self._persist:
                self._session_file = new_session_file
            self._build_index()

            if self._persist:
                # Only write the file now if it contains an assistant message.
                # Otherwise defer to _persist_entry(), which creates the file on the
                # first assistant response, matching the new_session() contract and
                # avoiding the duplicate-header bug when the no-assistant guard later
                # resets flushed to False.
                has_assistant = any(
                    e.get("type") == "message" and getattr(e.get("message"), "role", None) == "assistant"
                    for e in self._file_entries
                )
                if has_assistant:
                    self._rewrite_file()
                    self._flushed = True
                else:
                    self._flushed = False
                return new_session_file

            # In-memory mode: replace current session with the path + labels
            return None

    # -- constructors ------------------------------------------------------------

    @staticmethod
    def create(cwd: str, session_dir: str | None = None, options: dict[str, Any] | None = None) -> SessionManager:
        """Create a new session."""
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        return SessionManager(cwd, directory, None, True, options, _internal=True)

    @staticmethod
    def open(path: str, session_dir: str | None = None, cwd_override: str | None = None) -> SessionManager:
        """Open a specific session file."""
        resolved_path = resolve_path(path)
        header: dict[str, Any] | None = None
        preloaded_file_entries: list[dict[str, Any]] | None = None
        if cwd_override is None and os.path.exists(resolved_path):
            try:
                header = _read_session_header(resolved_path)
            except SessionHeaderScanLimitError:
                # The bounded scan is only a discovery optimization. A full load remains
                # authoritative for legacy files with very large headers or prefixes.
                preloaded_file_entries = _load_wire_entries_from_file(resolved_path)
                first_entry = preloaded_file_entries[0] if preloaded_file_entries else None
                header = first_entry if first_entry and first_entry.get("type") == "session" else None
        cwd = cwd_override
        if cwd is None and header is not None:
            cwd = _get_session_header_cwd(header)
        if cwd is None:
            cwd = os.getcwd()
        # If no session_dir provided, derive from file's parent directory
        directory = normalize_path(session_dir) if session_dir else os.path.dirname(resolved_path)
        return SessionManager(cwd, directory, resolved_path, True, None, preloaded_file_entries, _internal=True)

    @staticmethod
    def continue_recent(cwd: str, session_dir: str | None = None) -> SessionManager:
        """Continue the most recent session, or create new if none."""
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        filter_cwd = session_dir is not None and directory != _get_default_session_dir_path(cwd)
        most_recent = find_most_recent_session(directory, cwd if filter_cwd else None)
        if most_recent:
            return SessionManager(cwd, directory, most_recent, True, _internal=True)
        return SessionManager(cwd, directory, None, True, _internal=True)

    @staticmethod
    def in_memory(cwd: str | None = None, options: dict[str, Any] | None = None) -> SessionManager:
        """Create an in-memory session (no file persistence)."""
        return SessionManager(cwd if cwd is not None else os.getcwd(), "", None, False, options, _internal=True)

    @staticmethod
    def fork_from(
        source_path: str,
        target_cwd: str,
        session_dir: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> SessionManager:
        """Fork a session from another project directory into the current project.
        Creates a new session in the target cwd with the full history from the source."""
        resolved_source_path = resolve_path(source_path)
        resolved_target_cwd = resolve_path(target_cwd)
        source_entries = _load_wire_entries_from_file(resolved_source_path)
        if not source_entries:
            raise Exception(f"Cannot fork: source session file is empty or invalid: {resolved_source_path}")

        source_header = next((entry for entry in source_entries if entry.get("type") == "session"), None)
        if source_header is None:
            raise Exception(f"Cannot fork: source session has no header: {resolved_source_path}")

        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(resolved_target_cwd)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # Create new session file with new ID but forked content
        session_id = (options or {}).get("id")
        if session_id is not None:
            assert_valid_session_id(session_id)
        new_session_id = session_id if session_id is not None else _create_session_id()
        timestamp = _iso_now()
        file_timestamp = timestamp.replace(":", "-").replace(".", "-")
        new_session_file = os.path.join(directory, f"{file_timestamp}_{new_session_id}.jsonl")

        # Write new header pointing to source as parent, with updated cwd
        new_header = {
            "type": "session",
            "version": CURRENT_SESSION_VERSION,
            "id": new_session_id,
            "timestamp": timestamp,
            "cwd": resolved_target_cwd,
            "parentSession": resolved_source_path,
        }
        with open(new_session_file, "x", encoding="utf-8", newline="") as handle:
            handle.write(_dump_json(new_header) + "\n")
            # Copy all non-header entries from source
            for entry in source_entries:
                if entry.get("type") != "session":
                    handle.write(_dump_json(entry) + "\n")

        return SessionManager(resolved_target_cwd, directory, new_session_file, True, _internal=True)

    @staticmethod
    async def list(
        cwd: str,
        session_dir: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SessionInfo]:
        """List all sessions for a directory."""
        directory = normalize_path(session_dir) if session_dir else get_default_session_dir(cwd)
        filter_cwd = session_dir is not None and directory != _get_default_session_dir_path(cwd)
        resolved_cwd = resolve_path(cwd)
        sessions = [
            session
            for session in await _list_sessions_from_dir(directory, on_progress)
            if not filter_cwd or _session_cwd_matches(session.cwd, resolved_cwd)
        ]
        sessions.sort(key=lambda session: session.modified, reverse=True)
        return sessions

    @staticmethod
    async def list_all(
        session_dir: str | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[SessionInfo]:
        """List all sessions across all project directories."""
        custom_session_dir = normalize_path(session_dir) if session_dir else None
        if custom_session_dir:
            sessions = await _list_sessions_from_dir(custom_session_dir, on_progress)
            sessions.sort(key=lambda session: session.modified, reverse=True)
            return sessions

        sessions_dir = get_sessions_dir()

        try:
            if not os.path.exists(sessions_dir):
                return []
            dir_names = await tonio.spawn_blocking(os.listdir, sessions_dir)
            dirs = [
                os.path.join(sessions_dir, name)
                for name in dir_names
                if os.path.isdir(os.path.join(sessions_dir, name))
            ]

            # Count total files first for accurate progress
            all_files: list[str] = []
            for directory in dirs:
                try:
                    names = await tonio.spawn_blocking(os.listdir, directory)
                    all_files.extend(os.path.join(directory, name) for name in names if name.endswith(".jsonl"))
                except Exception:  # noqa: S112 - skip unreadable project dirs like pi
                    continue

            loaded = 0
            loaded_guard = threading.Lock()
            total_files = len(all_files)

            def on_loaded() -> None:
                nonlocal loaded
                with loaded_guard:
                    loaded += 1
                    current = loaded
                if on_progress is not None:
                    on_progress(current, total_files)

            results = await _build_session_infos_with_concurrency(all_files, on_loaded)
            sessions = [info for info in results if info is not None]
            sessions.sort(key=lambda session: session.modified, reverse=True)
            return sessions
        except Exception:
            return []
