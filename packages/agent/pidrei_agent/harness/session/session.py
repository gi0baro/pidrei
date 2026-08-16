"""Session tree API over a `SessionStorage` backend (port of pi `session/session.ts`).

`Session` binds the "main" lane; `view(lane)` returns a `SessionTree` bound to
another lane without caching its leaf. Durable payloads are validated before any
storage mutation: entries and records must be JSON-representable — port-typed
dataclasses (entries, messages, usage) count as plain objects, everything else
(arbitrary class instances, non-finite numbers, cycles) is rejected exactly
where pi rejects non-plain JS values.
"""

import dataclasses
import math
from dataclasses import replace
from typing import Any, NoReturn

from pidrei_ai.utils.uuid import uuidv7

from .state import assert_valid_cursor, assert_valid_limit
from .types import (
    BranchQuery,
    CustomEntry,
    Entry,
    EntryQuery,
    IdGenerator,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    MessageEntry,
    OperationStartedRecord,
    RecordQuery,
    SessionError,
    SessionMetadata,
    SessionStats,
    SessionStorage,
    SessionTree,
)


def _invalid_payload(reason: str) -> NoReturn:
    raise SessionError("invalid_payload", f"Durable payload {reason}")


def assert_json_serializable(value: Any) -> None:
    active: set[int] = set()
    stack: list[tuple[str, Any]] = [("value", value)]
    while stack:
        op, candidate = stack.pop()
        if op == "exit":
            active.discard(id(candidate))
            continue
        if candidate is None or isinstance(candidate, str | bool):
            continue
        if isinstance(candidate, int):
            continue
        if isinstance(candidate, float):
            if not math.isfinite(candidate):
                _invalid_payload("contains a non-finite number")
            continue
        if id(candidate) in active:
            _invalid_payload("contains a cycle")
        if isinstance(candidate, list):
            active.add(id(candidate))
            stack.append(("exit", candidate))
            for item in reversed(candidate):
                stack.append(("value", item))
            continue
        if isinstance(candidate, dict):
            active.add(id(candidate))
            stack.append(("exit", candidate))
            for key in reversed(candidate.keys()):
                if not isinstance(key, str):
                    _invalid_payload("contains a non-string key")
                stack.append(("value", candidate[key]))
            continue
        if dataclasses.is_dataclass(candidate) and not isinstance(candidate, type):
            # Port-typed vocabulary (entries, messages, usage, intents) stands in
            # for pi's plain JS objects; their fields are validated recursively.
            active.add(id(candidate))
            stack.append(("exit", candidate))
            for field in reversed(dataclasses.fields(candidate)):
                stack.append(("value", getattr(candidate, field.name)))
            continue
        _invalid_payload(f"contains {type(candidate).__name__}")


class _Uuidv7IdGenerator:
    def next(self) -> str:
        return uuidv7()


class Session:
    def __init__(self, storage: SessionStorage, id_generator: IdGenerator | None = None):
        self._storage = storage
        self.id_generator: IdGenerator = id_generator if id_generator is not None else _Uuidv7IdGenerator()

    async def get_metadata(self) -> SessionMetadata:
        return await self._storage.get_metadata()

    def view(self, lane: str) -> SessionTree:
        if lane == "main":
            return self
        return _LaneView(self, lane)

    async def get_leaf_id(self) -> str | None:
        return await self._get_leaf_id_for_lane("main")

    async def get_entry(self, id: str) -> Entry | None:
        return await self._storage.get_entry(id)

    async def get_stats(self) -> SessionStats:
        return await self._storage.get_stats()

    async def get_name(self) -> str | None:
        return await self._storage.get_name()

    async def set_name(self, name: str | None) -> None:
        await self._storage.set_name(name)

    async def get_label(self, target_id: str) -> str | None:
        return await self._storage.get_label(target_id)

    async def set_label(self, target_id: str, label: str | None) -> None:
        await self._storage.set_label(target_id, label)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return await self._query_entries(query)

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None:
        results = await self._query_entries(query, result_limit=1)
        return results[0] if results else None

    async def find_entries_on_branch(self, query: BranchQuery | None = None) -> list[Entry]:
        return await self._query_branch_entries("main", query)

    async def find_entry_on_branch(self, query: BranchQuery | None = None) -> Entry | None:
        results = await self._query_branch_entries("main", query, result_limit=1)
        return results[0] if results else None

    async def append_message(self, message: Any) -> str:
        return await self._append_message_to_lane("main", message)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._append_custom_entry_to_lane("main", custom_type, data)

    async def get_lanes(self) -> list[LanePointer]:
        return await self._storage.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        await self._storage.create_lane(lane, at)

    async def move_lane(self, lane: str, to: str | None) -> None:
        await self._storage.move_lane(lane, to)

    async def append_entry[TEntry: Entry](self, entry: TEntry, lane: str) -> TEntry:
        return await self._commit_entry(entry, lane)

    async def append_record[TRecord: LaneRecord](self, record: TRecord) -> TRecord:
        return await self._commit_record(record)

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        return await self._query_records(query)

    async def find_open_operations(self, lane: str, limit: int | None = None) -> list[OperationStartedRecord]:
        assert_valid_limit(limit)
        return await self._storage.find_open_operations(lane, limit)

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        options = options if options is not None else LogOptions()
        assert_valid_limit(options.limit)
        assert_valid_cursor(options.after_seq)
        return await self._storage.get_log(options)

    async def _get_leaf_id_for_lane(self, lane: str) -> str | None:
        """Returns the lane's current leaf, or None when empty. Raises when the lane does not exist."""
        for pointer in await self.get_lanes():
            if pointer.lane == lane:
                return pointer.leaf_id
        raise SessionError("invalid_lane", f"Lane not found: {lane}")

    async def _query_entries(self, query: EntryQuery | None, result_limit: int | None = None) -> list[Entry]:
        query = query if query is not None else EntryQuery()
        assert_valid_limit(query.limit)
        assert_valid_cursor(query.cursor.after_seq if query.cursor is not None else None)
        if result_limit is None:
            result_limit = query.limit
        return await self._storage.find_entries(
            query if result_limit == query.limit else replace(query, limit=result_limit)
        )

    async def _query_branch_entries(
        self, default_lane: str, query: BranchQuery | None, result_limit: int | None = None
    ) -> list[Entry]:
        """Queries from `query.start` toward the root, defaulting to the lane's current leaf.

        `result_limit` lets single-entry queries cap results without changing the caller's query.
        """
        query = query if query is not None else BranchQuery()
        assert_valid_limit(query.limit)
        assert_valid_cursor(query.cursor.after_seq if query.cursor is not None else None)
        if result_limit is None:
            result_limit = query.limit
        start = query.start if query.start is not None else await self._get_leaf_id_for_lane(default_lane)
        if start is None:
            return []
        return await self._storage.find_entries_on_branch(replace(query, start=start, limit=result_limit))

    async def _query_records(self, query: RecordQuery | None) -> list[LaneRecord]:
        query = query if query is not None else RecordQuery()
        assert_valid_limit(query.limit)
        assert_valid_cursor(query.after_seq)
        if query.operation_kind is not None and query.type != "operation_started":
            raise SessionError("invalid_query", 'operationKind requires type "operation_started"')
        return await self._storage.find_records(query)

    async def _append_message_to_lane(self, lane: str, message: Any) -> str:
        entry = await self._commit_entry(MessageEntry(id=self.id_generator.next(), message=message), lane)
        return entry.id

    async def _append_custom_entry_to_lane(self, lane: str, custom_type: str, data: Any) -> str:
        entry = await self._commit_entry(
            CustomEntry(id=self.id_generator.next(), custom_type=custom_type, data=data), lane
        )
        return entry.id

    async def _commit_entry[TEntry: Entry](self, entry: TEntry, lane: str) -> TEntry:
        assert_json_serializable(entry)
        return await self._storage.append_entry(entry, lane)

    async def _commit_record[TRecord: LaneRecord](self, record: TRecord) -> TRecord:
        assert_json_serializable(record)
        return await self._storage.append_record(record)


class _LaneView:
    """SessionTree bound to a non-main lane; resolves the leaf per call."""

    def __init__(self, session: Session, lane: str):
        self._session = session
        self._lane = lane

    async def get_leaf_id(self) -> str | None:
        return await self._session._get_leaf_id_for_lane(self._lane)

    async def get_entry(self, id: str) -> Entry | None:
        return await self._session.get_entry(id)

    async def get_stats(self) -> SessionStats:
        return await self._session.get_stats()

    async def get_name(self) -> str | None:
        return await self._session.get_name()

    async def set_name(self, name: str | None) -> None:
        await self._session.set_name(name)

    async def get_label(self, target_id: str) -> str | None:
        return await self._session.get_label(target_id)

    async def set_label(self, target_id: str, label: str | None) -> None:
        await self._session.set_label(target_id, label)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return await self._session._query_entries(query)

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None:
        results = await self._session._query_entries(query, result_limit=1)
        return results[0] if results else None

    async def find_entries_on_branch(self, query: BranchQuery | None = None) -> list[Entry]:
        return await self._session._query_branch_entries(self._lane, query)

    async def find_entry_on_branch(self, query: BranchQuery | None = None) -> Entry | None:
        results = await self._session._query_branch_entries(self._lane, query, result_limit=1)
        return results[0] if results else None

    async def append_message(self, message: Any) -> str:
        return await self._session._append_message_to_lane(self._lane, message)

    async def append_custom_entry(self, custom_type: str, data: Any = None) -> str:
        return await self._session._append_custom_entry_to_lane(self._lane, custom_type, data)
