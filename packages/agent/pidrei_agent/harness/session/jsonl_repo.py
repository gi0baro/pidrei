"""JSONL session repository (port of pi `harness/session/jsonl-repo.ts`)."""

import re
from datetime import datetime

from ..types import JsonlSessionMetadata, SessionError, to_error
from .jsonl_storage import JsonlSessionStorage, load_jsonl_session_metadata
from .repo_utils import (
    create_session_id,
    create_timestamp,
    get_entries_to_fork,
    get_file_system_result_or_throw,
    to_session,
)
from .session import Session


def _encode_cwd(cwd: str) -> str:
    stripped = re.sub(r"^[/\\]", "", cwd)
    return f"--{re.sub(r'[/\\:]', '-', stripped)}--"


class JsonlSessionRepo:
    def __init__(self, fs, sessions_root: str):
        self._fs = fs
        self._sessions_root_input = sessions_root
        self._sessions_root: str | None = None

    async def _get_sessions_root(self) -> str:
        if self._sessions_root is None:
            self._sessions_root = get_file_system_result_or_throw(
                await self._fs.absolute_path(self._sessions_root_input),
                f"Failed to resolve sessions root {self._sessions_root_input}",
            )
        return self._sessions_root

    async def _get_session_dir(self, cwd: str) -> str:
        return get_file_system_result_or_throw(
            await self._fs.join_path([await self._get_sessions_root(), _encode_cwd(cwd)]),
            f"Failed to resolve session directory for {cwd}",
        )

    async def _create_session_file_path(self, cwd: str, session_id: str, timestamp: str) -> str:
        return get_file_system_result_or_throw(
            await self._fs.join_path(
                [await self._get_session_dir(cwd), f"{re.sub(r'[:.]', '-', timestamp)}_{session_id}.jsonl"]
            ),
            f"Failed to resolve session file path for {session_id}",
        )

    async def create(
        self,
        cwd: str,
        id: str | None = None,
        parent_session_path: str | None = None,
        metadata: dict | None = None,
    ) -> Session:
        session_id = id if id is not None else create_session_id()
        created_at = create_timestamp()
        session_dir = await self._get_session_dir(cwd)
        get_file_system_result_or_throw(
            await self._fs.create_dir(session_dir, recursive=True),
            f"Failed to create session directory {session_dir}",
        )
        file_path = await self._create_session_file_path(cwd, session_id, created_at)
        storage = await JsonlSessionStorage.create(
            self._fs,
            file_path,
            cwd=cwd,
            session_id=session_id,
            parent_session_path=parent_session_path,
            metadata=metadata,
        )
        return to_session(storage)

    async def open(self, metadata: JsonlSessionMetadata) -> Session:
        if not get_file_system_result_or_throw(
            await self._fs.exists(metadata.path), f"Failed to check session {metadata.path}"
        ):
            raise SessionError("not_found", f"Session not found: {metadata.path}")
        storage = await JsonlSessionStorage.open(self._fs, metadata.path)
        return to_session(storage)

    async def list(self, cwd: str | None = None) -> list[JsonlSessionMetadata]:
        dirs = [await self._get_session_dir(cwd)] if cwd else await self._list_session_dirs()
        sessions: list[JsonlSessionMetadata] = []
        for directory in dirs:
            if not get_file_system_result_or_throw(
                await self._fs.exists(directory), f"Failed to check session directory {directory}"
            ):
                continue
            files = [
                file
                for file in get_file_system_result_or_throw(
                    await self._fs.list_dir(directory), f"Failed to list sessions in {directory}"
                )
                if file.kind != "directory" and file.name.endswith(".jsonl")
            ]
            for file in files:
                try:
                    sessions.append(await load_jsonl_session_metadata(self._fs, file.path))
                except Exception as error:
                    cause = to_error(error)
                    if not (isinstance(cause, SessionError) and cause.code == "invalid_session"):
                        raise cause from error
        sessions.sort(key=lambda entry: datetime.fromisoformat(entry.created_at), reverse=True)
        return sessions

    async def delete(self, metadata: JsonlSessionMetadata) -> None:
        get_file_system_result_or_throw(
            await self._fs.remove(metadata.path, force=True), f"Failed to delete session {metadata.path}"
        )

    async def fork(
        self,
        source_metadata: JsonlSessionMetadata,
        cwd: str,
        entry_id: str | None = None,
        position: str | None = None,
        id: str | None = None,
        parent_session_path: str | None = None,
        metadata: dict | None = None,
    ) -> Session:
        source = await self.open(source_metadata)
        forked_entries = await get_entries_to_fork(source.get_storage(), entry_id, position)
        session_id = id if id is not None else create_session_id()
        created_at = create_timestamp()
        session_dir = await self._get_session_dir(cwd)
        get_file_system_result_or_throw(
            await self._fs.create_dir(session_dir, recursive=True),
            f"Failed to create session directory {session_dir}",
        )
        storage = await JsonlSessionStorage.create(
            self._fs,
            await self._create_session_file_path(cwd, session_id, created_at),
            cwd=cwd,
            session_id=session_id,
            parent_session_path=parent_session_path if parent_session_path is not None else source_metadata.path,
            metadata=metadata if metadata is not None else source_metadata.metadata,
        )
        for entry in forked_entries:
            await storage.append_entry(entry)
        return to_session(storage)

    async def _list_session_dirs(self) -> list[str]:
        sessions_root = await self._get_sessions_root()
        if not get_file_system_result_or_throw(
            await self._fs.exists(sessions_root), f"Failed to check sessions root {sessions_root}"
        ):
            return []
        entries = get_file_system_result_or_throw(
            await self._fs.list_dir(sessions_root), f"Failed to list sessions root {sessions_root}"
        )
        return [entry.path for entry in entries if entry.kind == "directory"]
