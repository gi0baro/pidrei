"""Mirror of pi coding-agent src/core/tools/read.ts."""

import codecs
import dataclasses
import math
import os
import stat
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs
from tonio.colored.exceptions import CancelledError

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import ImageContent, JsonSchemaConstrainedSampling, ModelImageResizeOptions, TextContent

from ...utils.image_process import process_image
from ...utils.mime import detect_supported_image_mime_type_from_file_blocking
from ..extensions.types import ToolDefinition
from .path_utils import resolve_read_path_blocking
from .renderers.read import read_renderers
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
    utf8_byte_length,
)


READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to read (relative or absolute)"},
        "offset": {"type": "number", "description": "Line number to start reading from (1-indexed)"},
        "limit": {"type": "number", "description": "Maximum number of lines to read"},
    },
    "required": ["path"],
}

# Result for programmatic callers such as codemode scripts: the text for text
# files, and an image block for images that codemode's `image()` accepts. `note`
# is the text that goes with the image, such as resize hints. Property
# descriptions are left out so the type stays on one line in tool descriptions.
READ_OUTPUT_SCHEMA = {
    "anyOf": [
        {"type": "string"},
        {
            "type": "object",
            "properties": {
                "type": {"const": "image", "type": "string"},
                "data": {"type": "string"},
                "mimeType": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["type", "data", "mimeType", "note"],
        },
    ]
}

READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION: dict[str, Any] = {
    "snippet": "Read file contents",
    "guidelines": ("Use read to examine files instead of cat or sed.",),
}


@dataclass(slots=True)
class ReadToolDetails:
    truncation: TruncationResult | None = None


class LocalReadOperations:
    async def read_file(self, absolute_path: str) -> bytes:
        return await fs.Path(absolute_path).read_bytes()

    async def access(self, absolute_path: str) -> None:
        def check_blocking() -> None:
            with open(absolute_path, "rb"):
                pass

        await tonio.spawn_blocking(check_blocking)

    async def detect_image_mime_type(self, absolute_path: str) -> str | None:
        return await tonio.spawn_blocking(detect_supported_image_mime_type_from_file_blocking, absolute_path)


def _to_read_output(content: list[TextContent | ImageContent]) -> str | dict[str, str]:
    """The image block and its note, or the text for text files and images that could not be processed."""
    text = next((block.text for block in content if isinstance(block, TextContent)), "")
    image = next((block for block in content if isinstance(block, ImageContent)), None)
    if image is None:
        return text
    return {"type": "image", "data": image.data, "mimeType": image.mime_type, "note": text}


def _get_non_vision_image_note(model: Any) -> str | None:
    if model is None or "image" in model.input:
        return None
    return "[Current model does not support images. The image will be omitted from this request.]"


def _throw_if_aborted(cancel: Any) -> None:
    if cancel is not None and cancel.cancelled:
        raise Exception("Operation aborted")


# The chunk size of the default local read (Node's fs.createReadStream reads 64 KB chunks).
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class _SelectedLines:
    """The lines a read shows, with the file's line count; line counts follow `text.split("\\n")`."""

    total_file_lines: int
    truncation: TruncationResult
    # Bytes of the first selected line, for the notice when it alone exceeds the byte limit.
    first_line_bytes: int


def _is_integer(value: Any) -> bool:
    # JS `Number.isInteger`: integral floats count.
    return isinstance(value, int) or (isinstance(value, float) and value.is_integer())


def _is_streamable_range(offset: Any, limit: Any) -> bool:
    """Whether a range can be streamed. Other values (fractional, negative limit) keep the whole-file path,
    whose list slicing defines what they select, for example limit -1 as all lines but the last.
    """
    return (offset is None or _is_integer(offset)) and (limit is None or (_is_integer(limit) and limit >= 0))


def _select_lines_from_buffer(buffer: bytes, start_line: int, limit: Any) -> _SelectedLines:
    """Select lines from a whole file in memory: for operations without create_read_stream and unusual ranges."""
    all_lines = buffer.decode("utf-8", "replace").split("\n")
    end_line = None if limit is None else min(start_line + int(limit), len(all_lines))
    truncation = truncate_head("\n".join(all_lines[start_line:end_line]))
    first_line_bytes = utf8_byte_length(all_lines[start_line]) if truncation.first_line_exceeds_limit else 0
    return _SelectedLines(total_file_lines=len(all_lines), truncation=truncation, first_line_bytes=first_line_bytes)


class _LineSelector:
    """pi's selectLinesFromStream, fed one chunk at a time. Only lines truncate_head can show are stored;
    the rest are counted, so memory does not grow with the file. Synchronous: it runs wherever the
    chunks are read (inside the local read's pool job, or per chunk of a custom stream).
    """

    def __init__(self, start_line: int, limit: int | None) -> None:
        self._start_line = start_line
        self._end_line = math.inf if limit is None else start_line + limit
        # Lossy UTF-8 that keeps a byte order mark decodes like bytes.decode("utf-8", "replace").
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._stored_lines: list[str] = []
        self._stored_bytes = 0
        self._stored_past_caps = False
        self._line = 0
        self._selected_lines = 0
        self._selected_bytes = 0
        self._first_line_bytes = 0
        self._last_line_empty = True
        # The selected line being read: its text, while it is stored, and its size.
        self._pieces: list[str] | None = []
        self._line_bytes = 0
        self._line_has_text = False

    def feed(self, chunk: bytes) -> None:
        self._select(self._decoder.decode(chunk))

    def finish(self) -> _SelectedLines:
        self._select(self._decoder.decode(b"", final=True))
        if self._start_line <= self._line < self._end_line:
            self._finish_line()
        truncation = dataclasses.replace(
            truncate_head("\n".join(self._stored_lines)),
            # truncate_head's counts for the joined selected lines, which drop a trailing empty line.
            total_lines=0 if self._selected_bytes == 0 else self._selected_lines - (1 if self._last_line_empty else 0),
            total_bytes=self._selected_bytes,
        )
        return _SelectedLines(
            total_file_lines=self._line + 1, truncation=truncation, first_line_bytes=self._first_line_bytes
        )

    def _append_to_line(self, text: str, start: int, end: int) -> None:
        self._line_has_text = True
        if self._pieces is None:
            return
        piece = text[start:end]
        # A line over the byte cap is never shown, and a first line over it is only measured: never store
        # all of a 1 GB line.
        if self._line_bytes <= DEFAULT_MAX_BYTES:
            self._pieces.append(piece)
        self._line_bytes += utf8_byte_length(piece)

    def _finish_line(self) -> None:
        if self._pieces is not None:
            if self._selected_lines == 0:
                self._first_line_bytes = self._line_bytes
            # Storing continues one line past the caps: truncate_head ignores a trailing empty line, so the
            # line after the cap must be there for it to decide truncation as for the full text.
            if len(self._stored_lines) > DEFAULT_MAX_LINES or self._stored_bytes > DEFAULT_MAX_BYTES:
                self._stored_past_caps = True
            text = "".join(self._pieces)
            self._stored_bytes += utf8_byte_length(text) + (1 if self._stored_lines else 0)
            self._stored_lines.append(text)
        self._selected_lines += 1
        self._last_line_empty = not self._line_has_text
        self._pieces = None if self._stored_past_caps else []
        self._line_bytes = 0
        self._line_has_text = False

    def _select(self, text: str) -> None:
        position = 0
        # Start of the selected text in this piece, measured as a whole: the joined lines' size.
        selection_start = -1
        while True:
            selected = self._start_line <= self._line < self._end_line
            if selected and selection_start == -1:
                selection_start = position
            newline = text.find("\n", position)
            end = len(text) if newline == -1 else newline
            if selected and end > position:
                self._append_to_line(text, position, end)
            if newline == -1:
                break
            if selected:
                self._finish_line()
                if self._line + 1 == self._end_line:
                    self._selected_bytes += utf8_byte_length(text[selection_start:newline])
                    selection_start = -1
            self._line += 1
            position = newline + 1
        if selection_start != -1:
            self._selected_bytes += utf8_byte_length(text[selection_start:])


def _select_local_lines_blocking(absolute_path: str, start_line: int, limit: int | None, cancel: Any) -> _SelectedLines:
    """The default local read, whole on the pool: the file the way fs.readFile reads it. A regular file is
    read up to its size when opened, so text appended during the read is not counted and a growing file
    still ends; a file reporting size 0 (such as /proc files) and other file types are read until EOF.
    The handle is closed here on every exit, so a caller that stops waiting leaves nothing to close.
    """
    selector = _LineSelector(start_line, limit)
    with open(absolute_path, "rb") as file:
        info = os.fstat(file.fileno())
        remaining = info.st_size if stat.S_ISREG(info.st_mode) and info.st_size > 0 else None
        while remaining is None or remaining > 0:
            chunk = file.read(_READ_CHUNK_BYTES if remaining is None else min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                break
            if remaining is not None:
                remaining -= len(chunk)
            _throw_if_aborted(cancel)
            selector.feed(chunk)
    return selector.finish()


async def _close_stream_detached(stream: AsyncGenerator[bytes]) -> None:
    try:
        await stream.aclose()
    except Exception:
        pass


async def _select_lines_from_stream(
    stream: AsyncGenerator[bytes], start_line: int, limit: int | None, cancel: Any
) -> _SelectedLines:
    """Select lines from a custom `create_read_stream`. Stops reading when the token is cancelled.

    pi's `for await` closes the stream when the loop is left early; Python's `async for` does not, so
    the stream is closed here on every early exit.
    """
    selector = _LineSelector(start_line, limit)
    try:
        async for chunk in stream:
            _throw_if_aborted(cancel)
            selector.feed(chunk)
    except CancelledError:
        # Nothing awaited on a cancelled chain runs: the close goes on its own coroutine.
        tonio.spawn.without_tracking(_close_stream_detached(stream))
        raise
    except Exception:
        await stream.aclose()
        raise
    return selector.finish()


def create_read_tool_definition(
    cwd: str,
    *,
    auto_resize_images: bool = True,
    resize_options: ModelImageResizeOptions | None = None,
    operations: Any = None,
) -> ToolDefinition:
    """`resize_options` is the fallback resize profile when the execution
    context has no model metadata.

    Custom `operations` may provide an optional `create_read_stream(absolute_path, *, cancel)` returning
    an async generator of `bytes` chunks: text files are then read through it, keeping only the shown
    lines, and `read` closes it with `aclose()` when it stops before the end. Without it, `read_file`
    loads the whole file. The default local read is one blocking-pool job instead of a stream."""
    ops = operations if operations is not None else LocalReadOperations()

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, ctx=None):
        path = params["path"]
        offset = params.get("offset")
        limit = params.get("limit")
        _throw_if_aborted(cancel)

        absolute_path = await tonio.spawn_blocking(
            resolve_read_path_blocking, path, (ctx.cwd if ctx is not None else None) or cwd
        )
        _throw_if_aborted(cancel)
        # Check if file exists and is readable.
        await ops.access(absolute_path)
        _throw_if_aborted(cancel)
        detect = getattr(ops, "detect_image_mime_type", None)
        mime_type = await detect(absolute_path) if detect is not None else None
        model = getattr(ctx, "model", None)
        non_vision_image_note = _get_non_vision_image_note(model)
        details: ReadToolDetails | None = None

        if mime_type:
            # Read image as binary.
            buffer = await ops.read_file(absolute_path)
            model_limits = model.input_limits if model is not None else None
            model_resize = (
                model_limits.images.resize if model_limits is not None and model_limits.images is not None else None
            )
            image_resize = model_resize if model_resize is not None else resize_options
            processed = await tonio.spawn_blocking(
                lambda: process_image(
                    buffer, mime_type, auto_resize_images=auto_resize_images, resize_options=image_resize
                )
            )
            if not processed.ok:
                text_note = f"Read image file [{mime_type}]\n{processed.message}"
                if non_vision_image_note:
                    text_note += f"\n{non_vision_image_note}"
                content: list[TextContent | ImageContent] = [TextContent(text=text_note)]
            else:
                text_note = f"Read image file [{processed.mime_type}]"
                if processed.hints:
                    text_note += "\n" + "\n".join(processed.hints)
                if non_vision_image_note:
                    text_note += f"\n{non_vision_image_note}"
                content = [
                    TextContent(text=text_note),
                    ImageContent(data=processed.data, mime_type=processed.mime_type),
                ]
        else:
            # Read text content. Convert offset from 1-indexed input to 0-indexed line access.
            start_line = max(0, int(offset) - 1) if offset else 0
            start_line_display = start_line + 1
            create_read_stream = getattr(ops, "create_read_stream", None)
            if not _is_streamable_range(offset, limit):
                selected = _select_lines_from_buffer(await ops.read_file(absolute_path), start_line, limit)
            elif operations is None:
                selected = await tonio.spawn_blocking(
                    _select_local_lines_blocking,
                    absolute_path,
                    start_line,
                    None if limit is None else int(limit),
                    cancel,
                )
            elif create_read_stream is not None:
                selected = await _select_lines_from_stream(
                    create_read_stream(absolute_path, cancel=cancel),
                    start_line,
                    None if limit is None else int(limit),
                    cancel,
                )
            else:
                selected = _select_lines_from_buffer(await ops.read_file(absolute_path), start_line, limit)
            _throw_if_aborted(cancel)
            total_file_lines = selected.total_file_lines
            truncation = selected.truncation
            # Check if offset is out of bounds.
            if start_line >= total_file_lines:
                raise Exception(f"Offset {offset} is beyond end of file ({total_file_lines} lines total)")
            # A user-specified limit is honored before truncation.
            user_limited_lines = None if limit is None else min(start_line + int(limit), total_file_lines) - start_line
            if truncation.first_line_exceeds_limit:
                # First line alone exceeds the byte limit. Point the model at a bash fallback.
                first_line_size = format_size(selected.first_line_bytes)
                output_text = (
                    f"[Line {start_line_display} is {first_line_size}, exceeds {format_size(DEFAULT_MAX_BYTES)} "
                    f"limit. Use bash: sed -n '{start_line_display}p' {path} | head -c {DEFAULT_MAX_BYTES}]"
                )
                details = ReadToolDetails(truncation=truncation)
            elif truncation.truncated:
                # Truncation occurred. Build an actionable continuation notice.
                end_line_display = start_line_display + truncation.output_lines - 1
                next_offset = end_line_display + 1
                output_text = truncation.content
                if truncation.truncated_by == "lines":
                    output_text += (
                        f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines}. "
                        f"Use offset={next_offset} to continue.]"
                    )
                else:
                    output_text += (
                        f"\n\n[Showing lines {start_line_display}-{end_line_display} of {total_file_lines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Use offset={next_offset} to continue.]"
                    )
                details = ReadToolDetails(truncation=truncation)
            elif user_limited_lines is not None and start_line + user_limited_lines < total_file_lines:
                # User-specified limit stopped early, but the file still has more content.
                remaining = total_file_lines - (start_line + user_limited_lines)
                next_offset = start_line + user_limited_lines + 1
                output_text = (
                    f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                )
            else:
                # No truncation and no remaining user-limited content.
                output_text = truncation.content
            content = [TextContent(text=output_text)]

        _throw_if_aborted(cancel)
        return AgentToolResult(content=content, details=details, structured_content=_to_read_output(content))

    return ToolDefinition(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
            "Images are sent as attachments. For text files, output is truncated to "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            "Use offset/limit for large files. When you need the full file, continue with offset until complete."
        ),
        prompt_snippet=READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
        prompt_guidelines=list(READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"]),
        parameters=READ_SCHEMA,
        output_schema=READ_OUTPUT_SCHEMA,
        constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
        execute=execute,
        render_call=read_renderers.render_call,
        render_result=read_renderers.render_result,
    )


def create_read_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_read_tool_definition(cwd, **options))
