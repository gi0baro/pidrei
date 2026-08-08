"""Session search contracts and the scanning fallback (port of pi `session/search.ts`)."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from ..types import FileError, Result
from .jsonl.codec import _entry_to_wire
from .session import Session
from .types import EntryQuery, SessionError, SessionMetadata


@dataclass(slots=True, kw_only=True)
class SessionSearchOptions:
    text: str
    cwd: str | None = None


@dataclass(slots=True, kw_only=True)
class SessionSearchHit:
    metadata: SessionMetadata
    entry_id: str
    timestamp: str
    snippet: str | None = None
    score: float | None = None


class SessionSearch(Protocol):
    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]: ...


def get_file_system_result_or_throw[TValue](result: Result[TValue, FileError], message: str) -> TValue:
    if not result.ok:
        code = "not_found" if result.error.code == "not_found" else "storage"
        raise SessionError(code, f"{message}: {result.error.message}", result.error)
    return result.value


class _ScanningSessionSearchSource(Protocol):
    async def list(self, options: Any = None) -> list[SessionMetadata]: ...
    async def open(self, metadata: SessionMetadata) -> Session: ...


def _to_iso(timestamp_ms: int) -> str:
    moment = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    return f"{moment.strftime('%Y-%m-%dT%H:%M:%S')}.{timestamp_ms % 1000:03d}Z"


class ScanningSessionSearch:
    def __init__(self, source: _ScanningSessionSearchSource):
        self._source = source

    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]:
        normalized_text = options.text.strip().lower()
        if not normalized_text:
            return []
        hits: list[SessionSearchHit] = []
        for metadata in await self._source.list():
            cwd = getattr(metadata, "cwd", None)
            if options.cwd is not None and cwd != options.cwd:
                continue
            session = await self._source.open(metadata)
            for entry in await session.find_entries(EntryQuery(order="oldestFirst")):
                payload = json.dumps(_entry_to_wire(entry, provisioned=False), ensure_ascii=False)
                if normalized_text not in payload.lower():
                    continue
                hits.append(
                    SessionSearchHit(
                        metadata=metadata,
                        entry_id=entry.id,
                        timestamp=_to_iso(entry.timestamp),
                        snippet=payload,
                    )
                )
        return hits


def create_scanning_session_search(source: _ScanningSessionSearchSource) -> ScanningSessionSearch:
    return ScanningSessionSearch(source)
