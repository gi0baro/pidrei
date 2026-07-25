"""Mirror of pi coding-agent src/core/tools/read.ts (execute path; renderers Phase 4)."""

from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import ImageContent, TextContent

from ...utils.image_process import process_image
from ...utils.mime import detect_supported_image_mime_type_from_file
from ..extensions.types import ToolDefinition
from .path_utils import resolve_read_path
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


@dataclass(slots=True)
class ReadToolDetails:
    truncation: TruncationResult | None = None


class LocalReadOperations:
    async def read_file(self, absolute_path: str) -> bytes:
        def read() -> bytes:
            with open(absolute_path, "rb") as f:
                return f.read()

        return await tonio.spawn_blocking(read)

    async def access(self, absolute_path: str) -> None:
        def check() -> None:
            with open(absolute_path, "rb"):
                pass

        await tonio.spawn_blocking(check)

    async def detect_image_mime_type(self, absolute_path: str) -> str | None:
        return await tonio.spawn_blocking(detect_supported_image_mime_type_from_file, absolute_path)


def _get_non_vision_image_note(model: Any) -> str | None:
    if model is None or "image" in model.input:
        return None
    return "[Current model does not support images. The image will be omitted from this request.]"


def _throw_if_aborted(cancel: Any) -> None:
    if cancel is not None and cancel.cancelled:
        raise Exception("Operation aborted")


def create_read_tool_definition(
    cwd: str,
    *,
    auto_resize_images: bool = True,
    operations: Any = None,
) -> ToolDefinition:
    ops = operations if operations is not None else LocalReadOperations()

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, ctx=None):
        path = params["path"]
        offset = params.get("offset")
        limit = params.get("limit")
        _throw_if_aborted(cancel)

        absolute_path = await tonio.spawn_blocking(resolve_read_path, path, cwd)
        _throw_if_aborted(cancel)
        # Check if file exists and is readable.
        await ops.access(absolute_path)
        _throw_if_aborted(cancel)
        detect = getattr(ops, "detect_image_mime_type", None)
        mime_type = await detect(absolute_path) if detect is not None else None
        non_vision_image_note = _get_non_vision_image_note(getattr(ctx, "model", None))
        details: ReadToolDetails | None = None

        if mime_type:
            # Read image as binary.
            buffer = await ops.read_file(absolute_path)
            processed = await tonio.spawn_blocking(
                lambda: process_image(buffer, mime_type, auto_resize_images=auto_resize_images)
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
            # Read text content.
            buffer = await ops.read_file(absolute_path)
            text_content = buffer.decode("utf-8", "replace")
            all_lines = text_content.split("\n")
            total_file_lines = len(all_lines)
            # Apply offset if specified. Convert from 1-indexed input to 0-indexed array access.
            start_line = max(0, int(offset) - 1) if offset else 0
            start_line_display = start_line + 1
            # Check if offset is out of bounds.
            if start_line >= len(all_lines):
                raise Exception(f"Offset {offset} is beyond end of file ({len(all_lines)} lines total)")
            user_limited_lines: int | None = None
            # If limit is specified by the user, honor it first. Otherwise truncate_head decides.
            if limit is not None:
                end_line = min(start_line + int(limit), len(all_lines))
                selected_content = "\n".join(all_lines[start_line:end_line])
                user_limited_lines = end_line - start_line
            else:
                selected_content = "\n".join(all_lines[start_line:])
            # Apply truncation, respecting both line and byte limits.
            truncation = truncate_head(selected_content)
            if truncation.first_line_exceeds_limit:
                # First line alone exceeds the byte limit. Point the model at a bash fallback.
                first_line_size = format_size(utf8_byte_length(all_lines[start_line]))
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
            elif user_limited_lines is not None and start_line + user_limited_lines < len(all_lines):
                # User-specified limit stopped early, but the file still has more content.
                remaining = len(all_lines) - (start_line + user_limited_lines)
                next_offset = start_line + user_limited_lines + 1
                output_text = (
                    f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
                )
            else:
                # No truncation and no remaining user-limited content.
                output_text = truncation.content
            content = [TextContent(text=output_text)]

        _throw_if_aborted(cancel)
        return AgentToolResult(content=content, details=details)

    return ToolDefinition(
        name="read",
        label="read",
        description=(
            "Read the contents of a file. Supports text files and images (jpg, png, gif, webp, bmp). "
            "Images are sent as attachments. For text files, output is truncated to "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            "Use offset/limit for large files. When you need the full file, continue with offset until complete."
        ),
        prompt_snippet="Read file contents",
        prompt_guidelines=["Use read to examine files instead of cat or sed."],
        parameters=READ_SCHEMA,
        execute=execute,
    )


def create_read_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_read_tool_definition(cwd, **options))
