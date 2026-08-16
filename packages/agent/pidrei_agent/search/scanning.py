"""Scanning session search (port of pi `agent/src/search/scanning.ts`).

The `SessionSearchOptions`/`SessionSearchHit` contracts live here too (pi keeps
them in `search/index.ts`, but Python cannot split a type-only import cycle);
`search/__init__.py` re-exports everything under pi's surface.

A "readable" is any object with `get_metadata`/`find_entries`/`get_label` —
`Session` and the storage backends both qualify. pi's `JSON.stringify(entry)`
default projection maps to the JSONL wire form of the entry, the port's
JSON.stringify stand-in for session entries.
"""

import json
from collections.abc import AsyncIterable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pidrei_ai.utils.cancel import CancelToken

from ..harness.session.jsonl.codec import _entry_to_wire
from ..harness.session.types import Entry, EntryCursor, EntryQuery, SessionMetadata


@dataclass(slots=True, kw_only=True)
class SessionSearchOptions:
    # Restrict results to specific canonical entry types.
    entry_types: list[str] | None = None
    # Maximum number of hits to return.
    limit: int | None = None
    # Cancellation, e.g. search-as-you-type (pi: AbortSignal).
    cancel: CancelToken | None = None


@dataclass(slots=True, kw_only=True)
class SessionSearchHit:
    # Logical identifier of the session that owns the entry.
    session_id: str
    # Logical identifier of the entry within that session.
    entry_id: str


@dataclass(slots=True, kw_only=True)
class SessionSearchCandidate:
    entry_id: str
    seq: int
    type: str
    timestamp: int
    text: str
    fields: dict[str, Any] | None = None


@dataclass(slots=True, kw_only=True)
class ScanningSessionSearchHit(SessionSearchHit):
    timestamp: int
    snippet: str


def _default_search_text(_metadata: SessionMetadata, entry: Entry, label: str | None) -> str:
    serialized = json.dumps(_entry_to_wire(entry, provisioned=False), ensure_ascii=False)
    return serialized if label is None else f"{serialized} {label}"


async def _scan_readable_entries(
    readable: Any,
    metadata: SessionMetadata,
    *,
    project_text: Callable[[SessionMetadata, Entry, str | None], str] | None = None,
    page_size: int | None = None,
    entry_types: list[str] | None = None,
) -> AsyncIterable[SessionSearchCandidate]:
    projector = project_text if project_text is not None else _default_search_text
    effective_page_size = page_size if page_size is not None else 100
    after_seq = 0
    entry_type_set = None if entry_types is None else set(entry_types)
    while True:
        entries = await readable.find_entries(
            EntryQuery(
                order="oldestFirst",
                limit=effective_page_size,
                cursor=EntryCursor(after_seq=after_seq),
                type=entry_types[0] if entry_types is not None and len(entry_types) == 1 else None,
            )
        )
        if not entries:
            break
        for entry in entries:
            if entry_type_set is not None and entry.type not in entry_type_set:
                continue
            label = await readable.get_label(entry.id)
            yield SessionSearchCandidate(
                entry_id=entry.id,
                seq=entry.seq,
                type=entry.type,
                timestamp=entry.timestamp,
                text=projector(metadata, entry, label),
                fields=None if label is None else {"label": label},
            )
        after_seq = entries[-1].seq if entries else after_seq
        if len(entries) < effective_page_size:
            break


async def scanning_entries(
    readable: Any,
    *,
    project_text: Callable[[SessionMetadata, Entry, str | None], str] | None = None,
    page_size: int | None = None,
) -> AsyncIterable[SessionSearchCandidate]:
    metadata = await readable.get_metadata()
    async for candidate in _scan_readable_entries(readable, metadata, project_text=project_text, page_size=page_size):
        yield candidate


async def _array_source(readables: Iterable[Any]) -> AsyncIterable[Any]:
    for readable in readables:
        yield readable


def _readables_for(source: Any, options: Any) -> AsyncIterable[Any]:
    return source(options) if callable(source) else _array_source(source)


def _default_match(query_text: str, candidate: SessionSearchCandidate) -> bool:
    return query_text in candidate.text.lower()


def _throw_if_aborted(cancel: CancelToken | None) -> None:
    if cancel is not None:
        cancel.raise_if_cancelled()


def _create_default_scanning_hit(
    metadata: SessionMetadata, candidate: SessionSearchCandidate
) -> ScanningSessionSearchHit:
    return ScanningSessionSearchHit(
        session_id=metadata.id,
        entry_id=candidate.entry_id,
        timestamp=candidate.timestamp,
        snippet=candidate.text,
    )


class _ScanningSessionSearch:
    def __init__(
        self,
        source: Any,
        *,
        project_text: Callable[[SessionMetadata, Entry, str | None], str] | None = None,
        page_size: int | None = None,
        source_options: Callable[[str, SessionSearchOptions], Any] | None = None,
        match: Callable[[str, SessionSearchCandidate, SessionMetadata], bool] | None = None,
        create_hit: Callable[[SessionMetadata, SessionSearchCandidate], SessionSearchHit] | None = None,
    ):
        self._source = source
        self._project_text = project_text
        self._page_size = page_size
        self._source_options = source_options
        self._match = match
        self._create_hit = create_hit if create_hit is not None else _create_default_scanning_hit

    async def search(self, text: str, options: SessionSearchOptions | None = None) -> AsyncIterable[SessionSearchHit]:
        search_options = options if options is not None else SessionSearchOptions()
        normalized_text = text.strip().lower()
        if not normalized_text or (search_options.limit is not None and search_options.limit <= 0):
            return
        if search_options.entry_types is not None and len(search_options.entry_types) == 0:
            return
        hit_count = 0
        seen_session_ids: set[str] = set()
        entry_type_set = None if search_options.entry_types is None else set(search_options.entry_types)
        source_options = (
            self._source_options(normalized_text, search_options) if self._source_options is not None else None
        )
        async for readable in _readables_for(self._source, source_options):
            _throw_if_aborted(search_options.cancel)
            metadata = await readable.get_metadata()
            if metadata.id in seen_session_ids:
                raise RuntimeError(f"Duplicate sessionId: {metadata.id}")
            seen_session_ids.add(metadata.id)
            async for candidate in _scan_readable_entries(
                readable,
                metadata,
                project_text=self._project_text,
                page_size=self._page_size,
                entry_types=search_options.entry_types,
            ):
                _throw_if_aborted(search_options.cancel)
                if entry_type_set is not None and candidate.type not in entry_type_set:
                    continue
                matches = (
                    self._match(normalized_text, candidate, metadata)
                    if self._match is not None
                    else _default_match(normalized_text, candidate)
                )
                if not matches:
                    continue
                yield self._create_hit(metadata, candidate)
                hit_count += 1
                if search_options.limit is not None and hit_count >= search_options.limit:
                    return


def create_scanning_session_search(source: Any, **options: Any) -> _ScanningSessionSearch:
    """`source` is a list of readables or a callable `(source_options) -> AsyncIterable[readable]`."""
    return _ScanningSessionSearch(source, **options)
