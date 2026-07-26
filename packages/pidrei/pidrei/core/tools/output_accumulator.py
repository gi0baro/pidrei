"""Mirror of pi coding-agent src/core/tools/output-accumulator.ts."""

import codecs
import os
import secrets
import tempfile
from dataclasses import replace
from typing import BinaryIO

from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail


def _default_temp_file_path(prefix: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{prefix}-{secrets.token_hex(8)}.log")


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

        self._temp_file_path: str | None = None
        self._temp_file: BinaryIO | None = None

    @property
    def full_output_path(self) -> str | None:
        return self._temp_file_path

    def append(self, data: bytes) -> None:
        if self._finished:
            raise Exception("Cannot append to a finished output accumulator")

        self._total_raw_bytes += len(data)
        self._append_decoded_text(self._decoder.decode(data))

        if self._temp_file is not None or self._should_use_temp_file():
            self._ensure_temp_file()
            if self._temp_file is not None:
                self._temp_file.write(data)
        elif data:
            self._raw_chunks.append(data)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        self._append_decoded_text(self._decoder.decode(b"", True))
        if self._should_use_temp_file():
            self._ensure_temp_file()

    def snapshot(self, *, persist_if_truncated: bool = False):
        tail_truncation = truncate_tail(self._get_snapshot_text(), max_lines=self._max_lines, max_bytes=self._max_bytes)
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

        return _OutputSnapshot(content=truncation.content, truncation=truncation, full_output_path=self._temp_file_path)

    async def close_temp_file(self) -> None:
        if self._temp_file is None:
            return
        temp_file = self._temp_file
        self._temp_file = None
        temp_file.close()

    def get_last_line_bytes(self) -> int:
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

    def _ensure_temp_file(self) -> None:
        if self._temp_file_path is not None:
            return
        self._temp_file_path = _default_temp_file_path(self._temp_file_prefix)
        self._temp_file = open(self._temp_file_path, "wb")  # noqa: SIM115
        for chunk in self._raw_chunks:
            self._temp_file.write(chunk)
        self._raw_chunks = []


class _OutputSnapshot:
    __slots__ = ("content", "full_output_path", "truncation")

    def __init__(self, *, content: str, truncation: TruncationResult, full_output_path: str | None):
        self.content = content
        self.truncation = truncation
        self.full_output_path = full_output_path
