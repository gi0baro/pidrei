"""Validated in-memory session state and its mutation log (port of pi `session/state.ts`).

`SessionState` is the single write-path for every backend: mutations carry the
next sequence number and are validated before they land, so a state rebuilt
from a persisted log and a live one agree by construction.
"""

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Literal, NoReturn

from .types import (
    BranchQuery,
    Entry,
    EntryLogItem,
    EntryOrder,
    EntryQuery,
    ForkOptions,
    LabelFactLogItem,
    LaneLogItem,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    NameFactLogItem,
    OperationStartedRecord,
    RecordLogItem,
    RecordQuery,
    SessionError,
    SessionStats,
)


@dataclass(slots=True, kw_only=True)
class EntryMutation:
    entry: Entry
    lane: str | None = None
    kind: Literal["entry"] = "entry"


@dataclass(slots=True, kw_only=True)
class RecordMutation:
    record: LaneRecord
    kind: Literal["record"] = "record"


@dataclass(slots=True, kw_only=True)
class LaneMutation:
    seq: int
    lane: str
    leaf_id: str | None
    kind: Literal["lane"] = "lane"


@dataclass(slots=True, kw_only=True)
class NameFactMutation:
    seq: int
    name: str
    kind: Literal["fact"] = "fact"
    fact: Literal["name"] = "name"


@dataclass(slots=True, kw_only=True)
class LabelFactMutation:
    seq: int
    target_id: str
    label: str | None
    kind: Literal["fact"] = "fact"
    fact: Literal["label"] = "label"


type SessionMutation = EntryMutation | RecordMutation | LaneMutation | NameFactMutation | LabelFactMutation

type InvalidMutation = Callable[[str], NoReturn]


def _invalid_mutation(message: str) -> NoReturn:
    raise SessionError("invalid_entry", f"Invalid session mutation: {message}")


def assert_valid_limit(limit: int | None) -> None:
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise SessionError("invalid_query", "limit must be a positive integer")


def assert_valid_cursor(after_seq: int | None) -> None:
    if after_seq is not None and (not isinstance(after_seq, int) or isinstance(after_seq, bool) or after_seq < 0):
        raise SessionError("invalid_query", "cursor sequence must be a non-negative integer")


def _ordered[T](items: list[T], order: EntryOrder | None) -> Iterator[T]:
    if order == "oldestFirst":
        yield from items
        return
    yield from reversed(items)


class SessionState:
    def __init__(self) -> None:
        self._sequence = 0
        self._used_ids: set[str] = set()
        self._entries: list[Entry] = []
        self._entries_by_id: dict[str, Entry] = {}
        self._records: list[LaneRecord] = []
        self._open_operations_by_lane: dict[str, dict[str, OperationStartedRecord]] = {}
        self._lanes: dict[str, str | None] = {"main": None}
        self._log: list[LogItem] = []
        self._stats = SessionStats()
        self._name: str | None = None
        self._labels: dict[str, str] = {}

    @property
    def next_sequence(self) -> int:
        return self._sequence + 1

    def get_lanes(self) -> list[LanePointer]:
        return [LanePointer(lane=lane, leaf_id=leaf_id) for lane, leaf_id in self._lanes.items()]

    def require_lane(self, lane: str) -> str | None:
        if lane not in self._lanes:
            raise SessionError("invalid_lane", f"Lane not found: {lane}")
        return self._lanes[lane]

    def validate_new_lane(self, lane: str) -> None:
        if lane in self._lanes:
            raise SessionError("already_exists", f"Lane already exists: {lane}")

    def validate_target(self, target_id: str | None) -> None:
        if target_id is not None and target_id not in self._entries_by_id:
            raise SessionError("not_found", f"Entry not found: {target_id}")

    def validate_unused_id(self, id: str) -> None:
        if id in self._used_ids:
            raise SessionError("already_exists", f"Session id already exists: {id}")

    def apply_mutation(self, mutation: SessionMutation, invalid: InvalidMutation = _invalid_mutation) -> None:
        if mutation.kind == "entry":
            seq = mutation.entry.seq
        elif mutation.kind == "record":
            seq = mutation.record.seq
        else:
            seq = mutation.seq
        if seq != self._sequence + 1:
            invalid(f"has non-consecutive seq {seq}")

        if mutation.kind == "entry":
            entry = mutation.entry
            if entry.id in self._used_ids:
                invalid(f"contains duplicate id {entry.id}")
            if mutation.lane is not None:
                if mutation.lane not in self._lanes:
                    invalid(f"references missing lane {mutation.lane}")
                if entry.parent_id != self._lanes[mutation.lane]:
                    invalid("does not chain to the lane leaf")
            if entry.parent_id is not None and entry.parent_id not in self._entries_by_id:
                invalid(f"references missing parent {entry.parent_id}")
            self._sequence = seq
            self._used_ids.add(entry.id)
            self._entries.append(entry)
            self._entries_by_id[entry.id] = entry
            if mutation.lane is not None:
                self._lanes[mutation.lane] = entry.id
            self._log.append(EntryLogItem(seq=seq, entry=entry))
            if entry.type == "message":
                self._stats.message_count += 1
        elif mutation.kind == "record":
            record = mutation.record
            if record.lane not in self._lanes:
                invalid(f"references missing lane {record.lane}")
            if record.id in self._used_ids:
                invalid(f"contains duplicate id {record.id}")
            self._sequence = seq
            self._used_ids.add(record.id)
            self._records.append(record)
            if record.type == "operation_started":
                self._open_operations_by_lane.setdefault(record.lane, {})[record.id] = record
            elif record.type == "operation_finished":
                self._open_operations_by_lane.get(record.lane, {}).pop(record.run_id, None)
            self._log.append(RecordLogItem(seq=seq, record=record))
            if record.type == "usage":
                self._stats.cached_tokens += record.usage.cache_read
                self._stats.uncached_tokens += record.usage.input + record.usage.cache_write
                self._stats.total_tokens += record.usage.total_tokens
                self._stats.cost_total += record.usage.cost.total
        elif mutation.kind == "lane":
            if mutation.leaf_id is not None and mutation.leaf_id not in self._entries_by_id:
                invalid(f"references missing lane target {mutation.leaf_id}")
            self._sequence = seq
            self._lanes[mutation.lane] = mutation.leaf_id
            self._log.append(LaneLogItem(seq=seq, lane=mutation.lane, leaf_id=mutation.leaf_id))
        elif mutation.fact == "name":
            self._sequence = seq
            self._name = mutation.name
            self._log.append(NameFactLogItem(seq=seq, name=mutation.name))
        else:
            if mutation.target_id not in self._entries_by_id:
                invalid(f"references missing label target {mutation.target_id}")
            self._sequence = seq
            if mutation.label is None:
                self._labels.pop(mutation.target_id, None)
            else:
                self._labels[mutation.target_id] = mutation.label
            self._log.append(LabelFactLogItem(seq=seq, target_id=mutation.target_id, label=mutation.label))

    def get_entry(self, id: str) -> Entry | None:
        return self._entries_by_id.get(id)

    def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        query = query if query is not None else EntryQuery()
        assert_valid_limit(query.limit)
        assert_valid_cursor(query.cursor.after_seq if query.cursor is not None else None)
        results: list[Entry] = []
        for entry in _ordered(self._entries, query.order):
            if not self._matches_entry_query(entry, query):
                continue
            results.append(entry)
            if len(results) == query.limit:
                break
        return results

    def find_entries_on_branch(self, query: BranchQuery) -> list[Entry]:
        assert_valid_limit(query.limit)
        assert_valid_cursor(query.cursor.after_seq if query.cursor is not None else None)
        if query.start is None:
            raise SessionError("invalid_query", "start is required")
        results: list[Entry] = []
        if query.order == "oldestFirst":
            for entry in reversed(list(self._walk_to_root(query.start))):
                reached_bound = entry.id == query.stop_at_id or entry.type == query.stop_at_type
                if self._matches_entry_query(entry, query):
                    results.append(entry)
                if reached_bound or len(results) == query.limit:
                    break
        else:
            for entry in self._walk_to_root(query.start, stop_at_id=query.stop_at_id, stop_at_type=query.stop_at_type):
                if self._matches_entry_query(entry, query):
                    results.append(entry)
                if len(results) == query.limit:
                    break
        return results

    def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        query = query if query is not None else RecordQuery()
        assert_valid_limit(query.limit)
        assert_valid_cursor(query.after_seq)
        results: list[LaneRecord] = []
        for record in _ordered(self._records, query.order):
            if not self._matches_record_query(record, query):
                continue
            results.append(record)
            if len(results) == query.limit:
                break
        return results

    def find_open_operations(self, lane: str, limit: int | None = None) -> list[OperationStartedRecord]:
        assert_valid_limit(limit)
        open_operations_by_id = self._open_operations_by_lane.get(lane)
        open_operations = list(reversed(open_operations_by_id.values())) if open_operations_by_id else []
        return open_operations if limit is None else open_operations[:limit]

    def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        options = options if options is not None else LogOptions()
        assert_valid_limit(options.limit)
        assert_valid_cursor(options.after_seq)
        results: list[LogItem] = []
        for item in self._log:
            if options.after_seq is not None and item.seq <= options.after_seq:
                continue
            results.append(item)
            if len(results) == options.limit:
                break
        return results

    def get_name(self) -> str | None:
        return self._name

    def get_label(self, id: str) -> str | None:
        return self._labels.get(id)

    def get_stats(self) -> SessionStats:
        return self._stats

    def create_fork_mutations(self, options: ForkOptions) -> list[SessionMutation]:
        copied_entries: list[Entry]
        fork_lanes: list[LanePointer]
        if options.scope == "tree":
            copied_entries = self.find_entries(EntryQuery(order="oldestFirst"))
            fork_lanes = self.get_lanes()
        else:
            selected_entry_id = options.entry_id if options.entry_id is not None else self.require_lane("main")
            target_id: str | None = None
            if selected_entry_id is not None:
                entry = self.get_entry(selected_entry_id)
                if entry is None or entry.type != "message":
                    raise SessionError(
                        "invalid_fork_target", f"Fork target is not a message entry: {selected_entry_id}"
                    )
                position = options.position
                if position is None:
                    position = "at" if options.entry_id is None else "before"
                target_id = entry.id if position == "at" else entry.parent_id
            copied_entries = (
                []
                if target_id is None
                else self.find_entries_on_branch(BranchQuery(start=target_id, order="oldestFirst"))
            )
            fork_lanes = [LanePointer(lane="main", leaf_id=target_id)]

        mutations: list[SessionMutation] = []
        sequence = 1
        for source_entry in copied_entries:
            mutations.append(EntryMutation(entry=replace(copy.deepcopy(source_entry), seq=sequence)))
            sequence += 1
        for pointer in fork_lanes:
            mutations.append(LaneMutation(seq=sequence, lane=pointer.lane, leaf_id=pointer.leaf_id))
            sequence += 1
        if self._name is not None:
            mutations.append(NameFactMutation(seq=sequence, name=self._name))
            sequence += 1
        for entry in copied_entries:
            label = self._labels.get(entry.id)
            if label is not None:
                mutations.append(LabelFactMutation(seq=sequence, target_id=entry.id, label=label))
                sequence += 1
        return mutations

    def _walk_to_root(
        self, start: str | None, *, stop_at_id: str | None = None, stop_at_type: str | None = None
    ) -> Iterator[Entry]:
        if start is None:
            return
        visited: set[str] = set()
        current = self._entries_by_id.get(start)
        if current is None:
            raise SessionError("not_found", f"Entry not found: {start}")
        while current is not None:
            if current.id in visited:
                raise SessionError("invalid_entry", f"Session branch contains a cycle at {current.id}")
            visited.add(current.id)
            yield current
            if current.id == stop_at_id or current.type == stop_at_type or current.parent_id is None:
                break
            parent_id = current.parent_id
            current = self._entries_by_id.get(parent_id)
            if current is None:
                raise SessionError("invalid_entry", f"Entry not found: {parent_id}")

    def _matches_entry_query(self, entry: Entry, query: EntryQuery) -> bool:
        if query.type is not None and entry.type != query.type:
            return False
        if query.custom_type is not None and (entry.type != "custom" or entry.custom_type != query.custom_type):
            return False
        if query.cursor is not None:
            if query.order == "oldestFirst":
                return entry.seq > query.cursor.after_seq
            return entry.seq < query.cursor.after_seq
        return True

    def _matches_record_query(self, record: LaneRecord, query: RecordQuery) -> bool:
        if query.lane is not None and record.lane != query.lane:
            return False
        if query.type is not None and record.type != query.type:
            return False
        if query.run_id is not None:
            if record.type == "operation_started":
                if record.id != query.run_id:
                    return False
            elif getattr(record, "run_id", None) != query.run_id:
                return False
        if query.operation_kind is not None and (
            record.type != "operation_started" or record.intent.kind != query.operation_kind
        ):
            return False
        return not (query.after_seq is not None and record.seq <= query.after_seq)
