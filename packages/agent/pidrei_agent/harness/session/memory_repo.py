"""In-memory session repository (port of pi `harness/session/memory-repo.ts`)."""

import threading

from ..types import SessionError, SessionMetadata
from .memory_storage import InMemorySessionStorage
from .repo_utils import create_session_id, create_timestamp, get_entries_to_fork, to_session
from .session import Session


class InMemorySessionRepo:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    async def create(self, id: str | None = None) -> Session:
        metadata = SessionMetadata(id=id if id is not None else create_session_id(), created_at=create_timestamp())
        storage = InMemorySessionStorage(metadata=metadata)
        session = to_session(storage)
        with self._lock:
            self._sessions[metadata.id] = session
        return session

    async def open(self, metadata: SessionMetadata) -> Session:
        session = self._sessions.get(metadata.id)
        if session is None:
            raise SessionError("not_found", f"Session not found: {metadata.id}")
        return session

    async def list(self) -> list[SessionMetadata]:
        with self._lock:
            sessions = list(self._sessions.values())
        return [await session.get_metadata() for session in sessions]

    async def delete(self, metadata: SessionMetadata) -> None:
        with self._lock:
            self._sessions.pop(metadata.id, None)

    async def fork(
        self,
        source_metadata: SessionMetadata,
        entry_id: str | None = None,
        position: str | None = None,
        id: str | None = None,
    ) -> Session:
        source = await self.open(source_metadata)
        forked_entries = await get_entries_to_fork(source.get_storage(), entry_id, position)
        metadata = SessionMetadata(id=id if id is not None else create_session_id(), created_at=create_timestamp())
        storage = InMemorySessionStorage(metadata=metadata, entries=forked_entries)
        session = to_session(storage)
        with self._lock:
            self._sessions[metadata.id] = session
        return session
