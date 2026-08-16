"""JSONL v4 backend contracts (port of pi `session/jsonl/types.ts`)."""

from dataclasses import dataclass
from typing import Literal, Protocol

from ...types import CancelToken, FileError, FileInfo, Result
from ..types import JsonValue, SessionCreateOptions, SessionMetadata


class JsonlSessionRepoFileSystem(Protocol):
    """The `FileSystem` subset the JSONL backend needs (pi's Pick<FileSystem, ...>)."""

    async def absolute_path(self, path: str, cancel: CancelToken | None = None) -> Result[str, FileError]: ...
    async def join_path(self, parts: list[str], cancel: CancelToken | None = None) -> Result[str, FileError]: ...
    async def read_text_file(self, path: str, cancel: CancelToken | None = None) -> Result[str, FileError]: ...
    async def read_text_lines(
        self, path: str, max_lines: int | None = None, cancel: CancelToken | None = None
    ) -> Result[list[str], FileError]: ...
    async def write_file(
        self, path: str, content: str | bytes, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...
    async def append_file(
        self, path: str, content: str | bytes, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...
    async def rename_file(
        self, source_path: str, destination_path: str, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...
    async def file_info(self, path: str, cancel: CancelToken | None = None) -> Result[FileInfo, FileError]: ...
    async def list_dir(self, path: str, cancel: CancelToken | None = None) -> Result[list[FileInfo], FileError]: ...
    async def exists(self, path: str, cancel: CancelToken | None = None) -> Result[bool, FileError]: ...
    async def create_dir(
        self, path: str, recursive: bool = True, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...
    async def remove(
        self, path: str, recursive: bool = False, force: bool = False, cancel: CancelToken | None = None
    ) -> Result[None, FileError]: ...


@dataclass(slots=True, kw_only=True)
class JsonlSessionRepoOptions:
    fs: JsonlSessionRepoFileSystem
    # Root containing coding-agent-compatible cwd-encoded session directories.
    sessions_root: str


@dataclass(slots=True, kw_only=True)
class JsonlSessionMetadata(SessionMetadata):
    cwd: str
    path: str
    # Filesystem modification time as milliseconds since Unix epoch.
    modified_at: float
    source_format: Literal[3, 4]
    # Present only when a v3 parent path could not be resolved to a session id.
    legacy_parent_session_path: str | None = None
    # Opaque application-owned metadata.
    metadata: dict[str, JsonValue] | None = None


@dataclass(slots=True, kw_only=True)
class JsonlSessionCreateOptions(SessionCreateOptions):
    cwd: str = ""
    metadata: dict[str, JsonValue] | None = None


@dataclass(slots=True, kw_only=True)
class JsonlSessionListOptions:
    cwd: str | None = None


@dataclass(slots=True, kw_only=True)
class JsonlV4Header:
    id: str
    created_at: int
    cwd: str
    parent_session_id: str | None = None
    # Preserved only when a v3 parent path could not be resolved to a session id.
    legacy_parent_session_path: str | None = None
    metadata: dict[str, JsonValue] | None = None
    kind: Literal["header"] = "header"
    version: Literal[4] = 4
