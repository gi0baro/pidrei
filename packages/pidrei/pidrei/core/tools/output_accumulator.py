"""Mirror of pi coding-agent src/core/tools/output-accumulator.ts."""

import codecs
import os
import threading
from dataclasses import dataclass, replace

import tonio.colored as tonio

from ...utils.output_files import OutputFileResult, TempFileWriter, create_output_file_stream
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail


def _byte_length(text: str) -> int:
    return len(text.encode("utf-8", "surrogatepass"))


class OutputAccumulator:
    """Incrementally tracks streaming output with bounded memory.

    Appends decode chunks with a streaming UTF-8 decoder, keeps only a decoded
    tail for display snapshots, and opens a temp file when the full output
    needs to be preserved.
    """

    def __init__(
        self, *, max_lines: int | None = None, max_bytes: int | None = None, temp_file_prefix: str | None = None
    ):
        self._max_lines = max_lines if max_lines is not None else DEFAULT_MAX_LINES
        self._max_bytes = max_bytes if max_bytes is not None else DEFAULT_MAX_BYTES
        self._max_rolling_bytes = max(self._max_bytes * 2, 1)
        self._temp_file_prefix = temp_file_prefix if temp_file_prefix is not None else "pidrei-output"
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")

        self._raw_chunks: list[bytes] = []
        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self._total_raw_bytes = 0
        self._total_decoded_bytes = 0
        self._completed_lines = 0
        self._total_lines = 0
        self._current_line_bytes = 0
        self._has_open_line = False
        self._finished = False

        self._temp_file: TempFileWriter | None = None
        # stdout and stderr readers append concurrently (pi's single thread
        # serialized them); the lock never suspends, so it is safe on workers.
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        with self._lock:
            if self._finished:
                raise Exception("Cannot append to a finished output accumulator")

            self._total_raw_bytes += len(data)
            self._append_decoded_text(self._decoder.decode(data))

            if self._temp_file is not None or self._should_use_temp_file():
                self._ensure_temp_file().write(data)
            elif data:
                self._raw_chunks.append(data)

    def finish(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._append_decoded_text(self._decoder.decode(b"", True))
            if self._should_use_temp_file():
                self._ensure_temp_file()

    def snapshot(self, *, persist_if_truncated: bool = False):
        with self._lock:
            return self._snapshot_locked(persist_if_truncated=persist_if_truncated)

    def _snapshot_locked(self, *, persist_if_truncated: bool):
        tail_truncation = truncate_tail(self._get_retained_text(), max_lines=self._max_lines, max_bytes=self._max_bytes)
        truncated = self._total_lines > self._max_lines or self._total_decoded_bytes > self._max_bytes
        truncated_by = (
            (tail_truncation.truncated_by or ("bytes" if self._total_decoded_bytes > self._max_bytes else "lines"))
            if truncated
            else None
        )
        truncation = replace(
            tail_truncation,
            truncated=truncated,
            truncated_by=truncated_by,
            total_lines=self._total_lines,
            total_bytes=self._total_decoded_bytes,
            max_lines=self._max_lines,
            max_bytes=self._max_bytes,
        )

        if persist_if_truncated and truncation.truncated:
            self._ensure_temp_file()

        temp_file = self._temp_file
        return _OutputSnapshot(
            content=truncation.content,
            truncation=truncation,
            full_output_file=temp_file.result if temp_file is not None else None,
        )

    async def close_temp_file(self) -> None:
        """Flush and close the temp file. Never raises: a failed file shows up as
        `failed` in later snapshots."""
        with self._lock:
            temp_file = self._temp_file
        if temp_file is not None:
            await temp_file.close()

    async def read_full_output(self, max_bytes: int) -> FullOutput:
        """The complete output, for callers that can take more than the display
        snapshot. Call after `finish()` and `close_temp_file()`. Output longer
        than `max_bytes` raw bytes keeps its first and last `max_bytes // 2`
        bytes around an omission marker. When the temp file could not be saved,
        only the retained tail is left: up to its last `max_bytes` bytes, behind
        the same marker when output is missing."""
        with self._lock:
            full_output_file = self._temp_file.result if self._temp_file is not None else None
            if full_output_file is None:
                raw = b"".join(self._raw_chunks)
            elif full_output_file.kind == "failed":
                return self._read_retained_output(max_bytes)
        if full_output_file is None:
            return FullOutput(content=raw.decode("utf-8", "replace"), truncated=False)
        return await tonio.spawn_blocking(_read_full_output_blocking, full_output_file.path, max_bytes)

    def get_last_line_bytes(self) -> int:
        with self._lock:
            return self._current_line_bytes

    def _append_decoded_text(self, text: str) -> None:
        if not text:
            return

        bytes_count = _byte_length(text)
        self._total_decoded_bytes += bytes_count
        self._tail_text += text
        self._tail_bytes += bytes_count
        if self._tail_bytes > self._max_rolling_bytes * 2:
            self._trim_tail()

        newlines = text.count("\n")
        if newlines == 0:
            self._current_line_bytes += bytes_count
            self._has_open_line = True
        else:
            self._completed_lines += newlines
            tail = text[text.rfind("\n") + 1 :]
            self._current_line_bytes = _byte_length(tail)
            self._has_open_line = len(tail) > 0
        self._total_lines = self._completed_lines + (1 if self._has_open_line else 0)

    def _trim_tail(self) -> None:
        buffer = self._tail_text.encode("utf-8", "replace")
        if len(buffer) <= self._max_rolling_bytes:
            self._tail_bytes = len(buffer)
            return

        start = len(buffer) - self._max_rolling_bytes
        while start < len(buffer) and (buffer[start] & 0xC0) == 0x80:
            start += 1

        self._tail_starts_at_line_boundary = (
            self._tail_starts_at_line_boundary if start == 0 else buffer[start - 1] == 0x0A
        )
        self._tail_text = buffer[start:].decode("utf-8", "replace")
        self._tail_bytes = _byte_length(self._tail_text)

    def _get_retained_text(self) -> str:
        """The retained tail, for display and as the fallback of `read_full_output()`.
        Clipping drops the partial first line of the tail. When the tail is only the
        end of one over-long line (plus blank lines), nothing is left, so it keeps
        the end of that line instead."""
        clipped_text = self._get_snapshot_text()
        return self._tail_text if len(clipped_text.strip()) == 0 else clipped_text

    def _read_retained_output(self, max_bytes: int) -> FullOutput:
        """The retained tail within `max_bytes`, marked as truncated only when output is missing."""
        data = self._get_retained_text().encode("utf-8", "surrogatepass")
        if len(data) > max_bytes:
            start = len(data) - max_bytes
            while start < len(data) and (data[start] & 0xC0) == 0x80:
                start += 1
            data = data[start:]
        tail = data.decode("utf-8", "replace")
        omitted = self._total_decoded_bytes - len(data)
        if omitted == 0:
            return FullOutput(content=tail, truncated=False)
        return FullOutput(content=f"[... {omitted} bytes omitted ...]\n\n{tail}", truncated=True)

    def _get_snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text

        first_newline = self._tail_text.find("\n")
        return self._tail_text if first_newline == -1 else self._tail_text[first_newline + 1 :]

    def _should_use_temp_file(self) -> bool:
        return (
            self._total_raw_bytes > self._max_bytes
            or self._total_decoded_bytes > self._max_bytes
            or self._total_lines > self._max_lines
        )

    def _ensure_temp_file(self) -> TempFileWriter:
        if self._temp_file is not None:
            return self._temp_file
        # `TempFileWriter.write` is a channel send, so this stays callable from
        # the sync streaming path — pi's `createWriteStream` behaves the same.
        temp_file = create_output_file_stream(self._temp_file_prefix, ".log")
        self._temp_file = temp_file
        for chunk in self._raw_chunks:
            temp_file.write(chunk)
        self._raw_chunks = []
        return temp_file


@dataclass(slots=True, frozen=True)
class FullOutput:
    content: str
    # Whether `content` omits part of the output.
    truncated: bool


def _read_full_output_blocking(path: str, max_bytes: int) -> FullOutput:
    with open(path, "rb") as file:
        size = os.fstat(file.fileno()).st_size
        if size <= max_bytes:
            return FullOutput(content=file.read().decode("utf-8", "replace"), truncated=False)
        head_bytes = max_bytes // 2
        tail_bytes = max_bytes - head_bytes
        head = file.read(head_bytes)
        file.seek(size - tail_bytes)
        tail = file.read(tail_bytes)
    # Cut at character boundaries: the incremental decode holds back an
    # incomplete trailing sequence, and the tail skips leading continuation bytes.
    head_text = codecs.getincrementaldecoder("utf-8")("replace").decode(head)
    tail_start = 0
    while tail_start < len(tail) and (tail[tail_start] & 0xC0) == 0x80:
        tail_start += 1
    tail_text = tail[tail_start:].decode("utf-8", "replace")
    omitted = size - head_bytes - tail_bytes
    return FullOutput(content=f"{head_text}\n\n[... {omitted} bytes omitted ...]\n\n{tail_text}", truncated=True)


class _OutputSnapshot:
    __slots__ = ("content", "full_output_file", "truncation")

    def __init__(self, *, content: str, truncation: TruncationResult, full_output_file: OutputFileResult | None):
        self.content = content
        self.truncation = truncation
        # Set once the full output goes to a file. While output streams in, `saved` means the file is being written.
        self.full_output_file = full_output_file
