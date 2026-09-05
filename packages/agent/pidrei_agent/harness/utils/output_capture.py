"""Bounded shell-output capture (port of pi `harness/utils/output-capture.ts`).

Maintains and publishes one bounded shell-output view. Writes received while
publication is rate-limited collapse into the latest view. Small changes remain
responsive; complete window turnovers purchase a proportionally longer delay.
The first update after idle and an explicit final flush are immediate.

Chunks arrive from the execution env's stdout and stderr reader tasks, so the
capture state sits behind a reentrant lock (pi's single thread serializes the
`data` events for free).
"""

import codecs
import json
import re
import threading
from collections.abc import Callable
from dataclasses import asdict

from pidrei_ai.utils.cancel import CancelToken

from ..types import (
    ShellOutputAppend,
    ShellOutputCaptureOptions,
    ShellOutputMetadata,
    ShellOutputMetadataUpdate,
    ShellOutputReplace,
    ShellOutputSlide,
    ShellOutputTruncation,
    ShellOutputUpdate,
    ShellOutputView,
)
from .adaptive_publisher import AdaptivePublisher
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, truncate_head, truncate_tail, utf8_byte_length


OUTPUT_MIN_EMIT_INTERVAL_MS = 100
OUTPUT_TARGET_BYTES_PER_SECOND = 100 * 1024

_INVALID_SHELL_OUTPUT_RE = re.compile("[\x00-\x08\x0b-\x1f￹-￻]")


def sanitize_shell_output(text: str) -> str:
    return _INVALID_SHELL_OUTPUT_RE.sub("", text)


def _count_newlines(text: str) -> int:
    return text.count("\n")


def _trim_to_last_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8", errors="replace")


def _trim_to_first_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    end = max_bytes
    while end > 0 and (encoded[end] & 0xC0) == 0x80:
        end -= 1
    return encoded[:end].decode("utf-8", errors="replace")


def _encoded_size(update: ShellOutputUpdate) -> int:
    return utf8_byte_length(json.dumps(asdict(update), ensure_ascii=False, separators=(",", ":")))


def _suffix_prefix_overlap(before: str, after: str, scan: int) -> int:
    if not before or not after or scan == 0:
        return 0
    tail = before[len(before) - scan :] if len(before) > scan else before
    for probe_length in (min(64, len(after)), 1):
        probe = after[:probe_length]
        candidates = 0
        index = tail.find(probe)
        while index != -1:
            candidates += 1
            if candidates > 8:
                break
            overlap_length = len(tail) - index
            if overlap_length <= len(after) and tail[index:] == after[:overlap_length]:
                return overlap_length
            index = tail.find(probe, index + 1)
        if probe_length == 1:
            break
    return 0


def _update_from(previous: ShellOutputView | None, current: ShellOutputView) -> ShellOutputUpdate:
    if previous is None:
        return ShellOutputReplace(output=current)
    metadata = current.metadata
    if current.text == previous.text:
        return ShellOutputMetadataUpdate(metadata=metadata)
    if len(current.text) > len(previous.text) and current.text.startswith(previous.text):
        return ShellOutputAppend(text=current.text[len(previous.text) :], metadata=metadata)
    shared = _suffix_prefix_overlap(
        previous.text, current.text, min(len(previous.text), len(current.text), current.truncation.max_bytes * 2)
    )
    if shared > 0:
        return ShellOutputSlide(drop=len(previous.text) - shared, text=current.text[shared:], metadata=metadata)
    return ShellOutputReplace(output=current)


def apply_shell_output_update(current: ShellOutputView | None, update: ShellOutputUpdate) -> ShellOutputView:
    """Fold one incremental update into the accumulated view."""
    current_text = current.text if current is not None else ""
    if update.kind == "replace":
        return update.output
    if update.kind == "append":
        return ShellOutputView(
            text=f"{current_text}{update.text}",
            truncation=update.metadata.truncation,
            spill_path=update.metadata.spill_path,
            last_line_bytes=update.metadata.last_line_bytes,
        )
    if update.kind == "slide":
        return ShellOutputView(
            text=f"{current_text[update.drop :]}{update.text}",
            truncation=update.metadata.truncation,
            spill_path=update.metadata.spill_path,
            last_line_bytes=update.metadata.last_line_bytes,
        )
    return ShellOutputView(
        text=current_text,
        truncation=update.metadata.truncation,
        spill_path=update.metadata.spill_path,
        last_line_bytes=update.metadata.last_line_bytes,
    )


class OutputCapture:
    def __init__(
        self,
        options: ShellOutputCaptureOptions | None,
        cancel: CancelToken | None,
        *,
        on_update: Callable[[ShellOutputUpdate, CancelToken | None], None] | None,
        on_error: Callable[[Exception], None],
    ) -> None:
        self._max_bytes = options.limits.max_bytes if options is not None else DEFAULT_MAX_BYTES
        self._max_lines = options.limits.max_lines if options is not None else DEFAULT_MAX_LINES
        self._retain = options.limits.retain if options is not None else "tail"
        self._cancel = cancel
        self._on_update = on_update
        if isinstance(self._max_bytes, bool) or not isinstance(self._max_bytes, (int, float)) or self._max_bytes <= 0:
            raise TypeError("Output maxBytes must be a positive finite number")
        if isinstance(self._max_lines, bool) or not isinstance(self._max_lines, int) or self._max_lines <= 0:
            raise TypeError("Output maxLines must be a positive integer")

        self._lock = threading.RLock()
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""
        self._buffer_bytes = 0
        self._total_bytes = 0
        self._newlines = 0
        self._ends_with_newline = True
        self._current_line_bytes = 0
        self._spill_path: str | None = None
        self._disposed = False
        self._publisher: AdaptivePublisher[ShellOutputView, ShellOutputUpdate] = AdaptivePublisher(
            snapshot=self.snapshot,
            update=_update_from,
            measure=_encoded_size,
            publish=self._deliver,
            on_error=on_error,
            min_interval_ms=OUTPUT_MIN_EMIT_INTERVAL_MS,
            target_bytes_per_second=OUTPUT_TARGET_BYTES_PER_SECOND,
        )

    def _deliver(self, update: ShellOutputUpdate) -> None:
        if self._on_update is not None:
            self._on_update(update, self._cancel)

    @property
    def truncated(self) -> bool:
        with self._lock:
            return self._total_bytes > self._max_bytes or self._total_lines() > self._max_lines

    def push(self, chunk: str | bytes) -> None:
        with self._lock:
            if self._disposed:
                return
            if isinstance(chunk, str):
                self._append_text(self._decoder.decode(b"", True))
                self._append_text(chunk)
                return
            self._append_text(self._decoder.decode(chunk))

    def finish(self) -> None:
        with self._lock:
            if self._disposed:
                return
            self._append_text(self._decoder.decode(b"", True))

    def set_spill_path(self, path: str) -> None:
        with self._lock:
            if self._disposed or self._spill_path == path:
                return
            self._spill_path = path
            self._publisher.mark_dirty()
            self.flush()

    def snapshot(self) -> ShellOutputView:
        with self._lock:
            if self._retain == "head":
                retained = truncate_head(self._buffer, max_lines=self._max_lines, max_bytes=self._max_bytes)
            else:
                retained = truncate_tail(self._buffer, max_lines=self._max_lines, max_bytes=self._max_bytes)
            total_lines = self._total_lines()
            truncated = self.truncated
            return ShellOutputView(
                text=sanitize_shell_output(retained.content),
                truncation=ShellOutputTruncation(
                    truncated=truncated,
                    truncated_by=("lines" if total_lines > self._max_lines else "bytes") if truncated else None,
                    total_lines=total_lines,
                    total_bytes=self._total_bytes,
                    output_lines=retained.output_lines,
                    output_bytes=retained.output_bytes,
                    last_line_partial=retained.last_line_partial,
                    first_line_exceeds_limit=retained.first_line_exceeds_limit,
                    max_lines=retained.max_lines,
                    max_bytes=retained.max_bytes,
                ),
                spill_path=self._spill_path,
                last_line_bytes=self._current_line_bytes if retained.last_line_partial else None,
            )

    def flush(self) -> None:
        with self._lock:
            if self._disposed:
                return
            self._publisher.flush(True)

    def dispose(self) -> None:
        with self._lock:
            self._publisher.dispose()
            self._disposed = True

    def _append_text(self, text: str) -> None:
        if text == "":
            return
        text_bytes = utf8_byte_length(text)
        self._total_bytes += text_bytes
        self._newlines += _count_newlines(text)
        self._ends_with_newline = text.endswith("\n")
        last_newline = text.rfind("\n")
        self._current_line_bytes = (
            self._current_line_bytes + text_bytes if last_newline == -1 else utf8_byte_length(text[last_newline + 1 :])
        )
        self._buffer += text
        self._buffer_bytes += text_bytes

        guard = self._max_bytes * 2
        if self._buffer_bytes > guard * 2:
            self._buffer = (
                _trim_to_last_utf8_bytes(self._buffer, guard)
                if self._retain == "tail"
                else _trim_to_first_utf8_bytes(self._buffer, guard)
            )
            self._buffer_bytes = utf8_byte_length(self._buffer)
        self._publisher.mark_dirty()

    def _total_lines(self) -> int:
        return self._newlines + (0 if self._ends_with_newline or self._total_bytes == 0 else 1)


def metadata_of(view: ShellOutputView) -> ShellOutputMetadata:
    return view.metadata
