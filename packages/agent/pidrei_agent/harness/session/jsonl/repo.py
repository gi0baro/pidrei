"""JSONL v4 session repository (port of pi `session/jsonl/repo.ts`).

Sessions live under a coding-agent-compatible root: one `--<encoded cwd>--`
directory per working directory, one `<created-at>_<id>.jsonl` file per session.
"""

import re
import time
from dataclasses import replace
from datetime import UTC, datetime

from pidrei_ai.utils.uuid import uuidv7

from ..session import Session, assert_json_serializable
from ..types import ForkOptions, SessionError
from .codec import metadata_from_header, parse_header
from .errors import file_result, invalid_file
from .storage import JsonlSessionStorage
from .types import (
    JsonlSessionCreateOptions,
    JsonlSessionListOptions,
    JsonlSessionMetadata,
    JsonlSessionRepoOptions,
    JsonlV4Header,
)


_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def _validate_session_id(id: str) -> None:
    if not _SESSION_ID_PATTERN.match(id):
        raise SessionError(
            "invalid_payload",
            "Session id must be non-empty, contain only alphanumeric characters, '-', '_', and '.', "
            "and start and end with an alphanumeric character",
        )


def session_directory_name(cwd: str) -> str:
    return f"--{re.sub(r'[/\\\\:]', '-', re.sub(r'^[/\\\\]', '', cwd))}--"


def session_file_name(created_at: int, id: str) -> str:
    # pi: new Date(createdAt).toISOString() with ':' and '.' replaced by '-'.
    moment = datetime.fromtimestamp(created_at / 1000, tz=UTC)
    timestamp = f"{moment.strftime('%Y-%m-%dT%H:%M:%S')}.{created_at % 1000:03d}Z".replace(":", "-").replace(".", "-")
    return f"{timestamp}_{id}.jsonl"


class JsonlSessionRepo:
    def __init__(self, options: JsonlSessionRepoOptions):
        self._fs = options.fs
        self._sessions_root_input = options.sessions_root
        self._root: str | None = None

    async def create(self, options: JsonlSessionCreateOptions) -> Session:
        header, path = await self._prepare_create(options)
        return Session(await JsonlSessionStorage.create(self._fs, path, header))

    async def open(self, metadata: JsonlSessionMetadata) -> Session:
        return Session(await self._load_storage(metadata))

    async def list(self, options: JsonlSessionListOptions | None = None) -> list[JsonlSessionMetadata]:
        return await self._list_direct(options if options is not None else JsonlSessionListOptions())

    async def delete(self, metadata: JsonlSessionMetadata) -> None:
        file_result(await self._fs.remove(metadata.path, force=True), f"Failed to delete session {metadata.path}")

    async def fork(
        self,
        source: JsonlSessionMetadata,
        options: ForkOptions,
        create: JsonlSessionCreateOptions | None = None,
    ) -> Session:
        source_storage = await self._load_storage(source)
        create = create if create is not None else JsonlSessionCreateOptions(cwd=source.cwd)
        parent_session_id = create.parent_session_id if create.parent_session_id is not None else source.id
        header, path = await self._prepare_create(replace(create, parent_session_id=parent_session_id))
        return Session(await source_storage.fork(path, header, options))

    async def _load_storage(self, metadata: JsonlSessionMetadata) -> JsonlSessionStorage:
        if not file_result(await self._fs.exists(metadata.path), f"Failed to check session {metadata.path}"):
            raise SessionError("not_found", f"Session not found: {metadata.id}")
        storage = await JsonlSessionStorage.load(self._fs, metadata.path)
        loaded_metadata = await storage.get_metadata()
        if loaded_metadata.id != metadata.id:
            raise SessionError("invalid_entry", f"Session id does not match header: {metadata.id}")
        return storage

    async def _prepare_create(self, options: JsonlSessionCreateOptions) -> tuple[JsonlV4Header, str]:
        id = options.id if options.id is not None else uuidv7()
        _validate_session_id(id)
        cwd = file_result(await self._fs.absolute_path(options.cwd), f"Failed to resolve session cwd {options.cwd}")
        if await self._session_id_exists(id, cwd):
            raise SessionError("already_exists", f"Session already exists: {id}")

        created_at = int(time.time() * 1000)
        session_directory = await self._session_directory(cwd)
        path = file_result(
            await self._fs.join_path([session_directory, session_file_name(created_at, id)]),
            f"Failed to resolve path for session {id}",
        )
        if options.metadata is not None:
            assert_json_serializable(options.metadata)
        header = JsonlV4Header(
            id=id,
            created_at=created_at,
            cwd=cwd,
            parent_session_id=options.parent_session_id,
            metadata=options.metadata,
        )
        file_result(await self._fs.create_dir(session_directory, recursive=True), "Failed to create sessions directory")
        return header, path

    async def _list_direct(self, options: JsonlSessionListOptions) -> list[JsonlSessionMetadata]:
        directories = await self._session_directories(options.cwd)
        metadata: list[JsonlSessionMetadata] = []
        for directory in directories:
            entries = file_result(await self._fs.list_dir(directory), f"Failed to list sessions directory {directory}")
            files = [entry for entry in entries if entry.kind != "directory" and entry.name.endswith(".jsonl")]
            for file in files:
                content = file_result(
                    await self._fs.read_text_file(file.path), f"Failed to read session header {file.path}"
                )
                first_line = content.split("\n", 1)[0]
                if not first_line:
                    raise invalid_file(file.path, 1, "is missing a header")
                metadata.append(metadata_from_header(parse_header(first_line, file.path), file.path, file.mtime_ms))
        return sorted(metadata, key=lambda item: item.modified_at, reverse=True)

    async def _session_id_exists(self, id: str, cwd: str) -> bool:
        suffix = f"_{id}.jsonl"
        directory = await self._session_directory(cwd)
        if not file_result(await self._fs.exists(directory), f"Failed to check sessions directory {directory}"):
            return False
        files = file_result(await self._fs.list_dir(directory), f"Failed to list sessions directory {directory}")
        return any(entry.kind != "directory" and entry.name.endswith(suffix) for entry in files)

    async def _session_directories(self, cwd: str | None) -> list[str]:
        root = await self._resolve_root()
        if cwd is not None:
            resolved_cwd = file_result(await self._fs.absolute_path(cwd), f"Failed to resolve session cwd {cwd}")
            directory = await self._session_directory(resolved_cwd)
            exists = file_result(await self._fs.exists(directory), f"Failed to check sessions directory {directory}")
            return [directory] if exists else []
        if not file_result(await self._fs.exists(root), f"Failed to check sessions directory {root}"):
            return []
        entries = file_result(await self._fs.list_dir(root), f"Failed to list sessions directory {root}")
        return [entry.path for entry in entries if entry.kind in ("directory", "symlink")]

    async def _session_directory(self, cwd: str) -> str:
        return file_result(
            await self._fs.join_path([await self._resolve_root(), session_directory_name(cwd)]),
            f"Failed to resolve sessions directory for {cwd}",
        )

    async def _resolve_root(self) -> str:
        if self._root is None:
            self._root = file_result(
                await self._fs.absolute_path(self._sessions_root_input),
                f"Failed to resolve sessions root {self._sessions_root_input}",
            )
        return self._root
