"""JSONL v4 session storage (port of pi `session/jsonl/storage.ts`).

One append-only file per session: a header line followed by one mutation per
line, all sharing the session's monotonic sequence. Writes are serialized on a
FIFO tail so concurrent appends commit in call order, and every mutation is
appended to disk before it lands in the in-memory state.
"""

import copy
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace

import tonio.colored as tonio

from ..state import (
    EntryMutation,
    LabelFactMutation,
    LaneMutation,
    NameFactMutation,
    RecordMutation,
    SessionMutation,
    SessionState,
)
from ..types import (
    BranchQuery,
    Entry,
    EntryQuery,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogItem,
    LogOptions,
    OperationStartedRecord,
    RecordQuery,
    SessionError,
    SessionStats,
)
from .codec import encode_header, encode_mutation, metadata_from_header, parse_header, parse_mutation
from .errors import JsonlDecodeError, file_result, invalid_file
from .types import JsonlSessionMetadata, JsonlSessionRepoFileSystem, JsonlV4Header


def _now_ms() -> int:
    return int(time.time() * 1000)


_PARSE_CHUNK_LINES = 256


def _parse_mutation_chunk(lines: list[str]) -> list:
    return [parse_mutation(line) for line in lines]


async def publish_file_atomically(
    fs: JsonlSessionRepoFileSystem,
    destination_path: str,
    populate: Callable[[str], Awaitable[None]],
) -> None:
    """Build a complete sibling temporary file, then atomically rename it over the destination.

    The populate callback must create or overwrite `temp_path` with the complete
    file. The destination is untouched until the rename commits, so a process
    crash while populating can leave only the ignored `.tmp` file behind.

    Raises when population or rename fails. On failure, temporary-file removal
    is best-effort and the original error is preserved. Callers must serialize
    publications to the same destination because they share its deterministic
    `.tmp` path.
    """
    temp_path = f"{destination_path}.tmp"
    try:
        await populate(temp_path)
        file_result(
            await fs.rename_file(temp_path, destination_path), f"Failed to publish staged file {destination_path}"
        )
    except BaseException:
        await fs.remove(temp_path, force=True)
        raise


class JsonlSessionStorage:
    def __init__(self, fs: JsonlSessionRepoFileSystem, metadata: JsonlSessionMetadata):
        self._fs = fs
        self._metadata = copy.deepcopy(metadata)
        self._state = SessionState()
        self._tail: tonio.Event | None = None
        self._tail_guard = threading.Lock()

    @staticmethod
    async def create(fs: JsonlSessionRepoFileSystem, path: str, header: JsonlV4Header) -> JsonlSessionStorage:
        file_result(await fs.write_file(path, encode_header(header)), f"Failed to initialize session {path}")
        file_info = file_result(await fs.file_info(path), f"Failed to read session metadata {path}")
        return JsonlSessionStorage(fs, metadata_from_header(header, path, file_info.mtime_ms))

    @staticmethod
    async def load(fs: JsonlSessionRepoFileSystem, path: str) -> JsonlSessionStorage:
        content = file_result(await fs.read_text_file(path), f"Failed to read session {path}")
        physical_lines = content.split("\n")
        if physical_lines and physical_lines[-1] == "":
            physical_lines.pop()
        if not physical_lines or not physical_lines[0]:
            raise invalid_file(path, 1, JsonlDecodeError("schema", "is missing a header"))
        header_result = parse_header(physical_lines[0])
        if not header_result.ok:
            raise invalid_file(path, 1, header_result.error)
        file_info = file_result(await fs.file_info(path), f"Failed to read session metadata {path}")
        storage = JsonlSessionStorage(fs, metadata_from_header(header_result.value, path, file_info.mtime_ms))
        # Decoding is pure and per-line, so it runs off the runtime in
        # parallel chunks; only the in-order apply stays here.
        mutation_lines = physical_lines[1:]
        chunks = [
            mutation_lines[start : start + _PARSE_CHUNK_LINES]
            for start in range(0, len(mutation_lines), _PARSE_CHUNK_LINES)
        ]
        parsed = [
            result
            for chunk_results in await tonio.map_blocking(_parse_mutation_chunk, chunks)
            for result in chunk_results
        ]
        for index in range(1, len(physical_lines)):
            mutation_result = parsed[index - 1]
            if not mutation_result.ok:
                is_torn_tail = index == len(physical_lines) - 1 and mutation_result.error.kind == "syntax"
                if is_torn_tail:
                    # Drop the unacknowledged partial append by atomically publishing the valid prefix.
                    valid_prefix = "\n".join(physical_lines[:index]) + "\n"

                    async def populate(temp_path: str, valid_prefix: str = valid_prefix) -> None:
                        file_result(
                            await fs.write_file(temp_path, valid_prefix), f"Failed to stage torn-tail repair {path}"
                        )

                    await publish_file_atomically(fs, path, populate)
                    return storage
                raise invalid_file(path, index + 1, mutation_result.error)
            try:
                storage._apply_mutation(mutation_result.value)
            except SessionError as error:
                if error.code == "invalid_entry":
                    raise invalid_file(path, index + 1, error) from error
                raise
        if not content.endswith("\n"):
            file_result(await fs.append_file(path, "\n"), f"Failed to repair unterminated session tail {path}")
        return storage

    async def fork(self, path: str, header: JsonlV4Header, options: ForkOptions) -> JsonlSessionStorage:
        mutations = self._state.create_fork_mutations(options)

        async def populate(temp_path: str) -> None:
            target_storage = await JsonlSessionStorage.create(self._fs, temp_path, header)
            for mutation in mutations:
                await target_storage._append_mutation(mutation)
                target_storage._apply_mutation(mutation)

        await publish_file_atomically(self._fs, path, populate)
        return await JsonlSessionStorage.load(self._fs, path)

    async def drain(self) -> None:
        tail = self._tail
        if tail is not None:
            await tail.wait()

    async def get_metadata(self) -> JsonlSessionMetadata:
        return copy.deepcopy(self._metadata)

    async def get_lanes(self) -> list[LanePointer]:
        return self._state.get_lanes()

    async def create_lane(self, lane: str, at: str | None) -> None:
        async def operation() -> None:
            self._state.validate_new_lane(lane)
            self._state.validate_target(at)
            mutation = LaneMutation(seq=self._state.next_sequence, lane=lane, leaf_id=at)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)

        await self._enqueue(operation)

    async def move_lane(self, lane: str, to: str | None) -> None:
        async def operation() -> None:
            self._state.require_lane(lane)
            self._state.validate_target(to)
            mutation = LaneMutation(seq=self._state.next_sequence, lane=lane, leaf_id=to)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)

        await self._enqueue(operation)

    async def append_entry[TEntry: Entry](self, new_entry: TEntry, lane: str) -> TEntry:
        async def operation() -> TEntry:
            parent_id = self._state.require_lane(lane)
            self._state.validate_unused_id(new_entry.id)
            entry = replace(
                copy.deepcopy(new_entry), parent_id=parent_id, seq=self._state.next_sequence, timestamp=_now_ms()
            )
            mutation = EntryMutation(lane=lane, entry=entry)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)
            return copy.deepcopy(entry)

        return await self._enqueue(operation)

    async def append_record[TRecord: LaneRecord](self, new_record: TRecord) -> TRecord:
        async def operation() -> TRecord:
            self._state.require_lane(new_record.lane)
            self._state.validate_unused_id(new_record.id)
            open_operations = self._state.find_open_operations(new_record.lane, limit=1)
            if new_record.type == "operation_started" and open_operations:
                raise SessionError(
                    "storage", f"Lane {new_record.lane} already has an open operation {open_operations[0].id}"
                )
            record = replace(copy.deepcopy(new_record), seq=self._state.next_sequence, timestamp=_now_ms())
            mutation = RecordMutation(record=record)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)
            return copy.deepcopy(record)

        return await self._enqueue(operation)

    async def get_entry(self, id: str) -> Entry | None:
        entry = self._state.get_entry(id)
        return None if entry is None else copy.deepcopy(entry)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        return copy.deepcopy(self._state.find_entries(query))

    async def find_entries_on_branch(self, query: BranchQuery) -> list[Entry]:
        return copy.deepcopy(self._state.find_entries_on_branch(query))

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        return copy.deepcopy(self._state.find_records(query))

    async def find_open_operations(self, lane: str, limit: int | None = None) -> list[OperationStartedRecord]:
        return copy.deepcopy(self._state.find_open_operations(lane, limit))

    async def get_log(self, options: LogOptions | None = None) -> list[LogItem]:
        return copy.deepcopy(self._state.get_log(options))

    async def get_name(self) -> str | None:
        return self._state.get_name()

    async def set_name(self, name: str | None) -> None:
        async def operation() -> None:
            mutation = NameFactMutation(seq=self._state.next_sequence, name=name)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)

        await self._enqueue(operation)

    async def get_label(self, id: str) -> str | None:
        return self._state.get_label(id)

    async def set_label(self, id: str, label: str | None) -> None:
        async def operation() -> None:
            self._state.validate_target(id)
            mutation = LabelFactMutation(seq=self._state.next_sequence, target_id=id, label=label)
            await self._append_mutation(mutation)
            self._apply_mutation(mutation)

        await self._enqueue(operation)

    async def get_stats(self) -> SessionStats:
        return copy.deepcopy(self._state.get_stats())

    async def _enqueue[T](self, operation: Callable[[], Awaitable[T]]) -> T:
        # FIFO chain: operations commit in call order and a failed predecessor
        # never blocks or fails its successors. The reservation is guarded
        # because "no await between reading and replacing the tail" is only
        # atomic on a single-threaded loop — tonio runs tasks on real threads,
        # where two callers can read the same tail and then run concurrently.
        # Same shape as `with_file_mutation_queue`'s `queues_guard`.
        with self._tail_guard:
            previous, mine = self._tail, tonio.Event()
            self._tail = mine
        try:
            if previous is not None:
                await previous.wait()
            return await operation()
        finally:
            mine.set()

    async def _append_mutation(self, mutation: SessionMutation) -> None:
        file_result(
            await self._fs.append_file(self._metadata.path, encode_mutation(mutation)),
            f"Failed to append session {self._metadata.path}",
        )

    def _apply_mutation(self, mutation: SessionMutation) -> None:
        self._state.apply_mutation(mutation)
