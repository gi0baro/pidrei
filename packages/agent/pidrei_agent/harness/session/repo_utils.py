"""Session repository helpers (port of pi `harness/session/repo-utils.ts`)."""

from datetime import UTC, datetime
from typing import Any

from pidrei_ai.utils.uuid import uuidv7

from ..types import FileError, Result, SessionError, SessionStorage, SessionTreeEntry
from .session import Session


def create_session_id() -> str:
    return uuidv7()


def create_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def to_session(storage: SessionStorage) -> Session:
    return Session(storage)


def get_file_system_result_or_throw[TValue](result: Result[TValue, FileError], message: str) -> TValue:
    if not result.ok:
        code = "not_found" if result.error.code == "not_found" else "storage"
        raise SessionError(code, f"{message}: {result.error.message}", result.error)
    return result.value


async def get_entries_to_fork(
    storage: SessionStorage,
    entry_id: str | None = None,
    position: str | None = None,
) -> list[SessionTreeEntry]:
    if not entry_id:
        return await storage.get_entries()
    target: Any = await storage.get_entry(entry_id)
    if target is None:
        raise SessionError("invalid_fork_target", f"Entry {entry_id} not found")
    if (position if position is not None else "before") == "at":
        effective_leaf_id = target.id
    else:
        if target.type != "message" or getattr(target.message, "role", None) != "user":
            raise SessionError("invalid_fork_target", f"Entry {entry_id} is not a user message")
        effective_leaf_id = target.parent_id
    return await storage.get_path_to_root_or_compaction(effective_leaf_id)
