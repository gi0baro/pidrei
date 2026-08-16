"""Mirror of pi coding-agent src/core/tools/read.ts."""

import os
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import ImageContent, TextContent
from pidrei_tui import Text

from ...config import get_readme_path
from ...modes.interactive.components.keybinding_hints import key_hint, key_text
from ...modes.interactive.theme import get_language_from_path, highlight_code
from ...utils.image_process import process_image
from ...utils.mime import detect_supported_image_mime_type_from_file
from ...utils.paths import format_path_relative_to_cwd_or_absolute
from ..experimental import get_experimental_tool_sampling
from ..extensions.types import ToolDefinition
from .path_utils import resolve_read_path, resolve_to_cwd
from .render_utils import get_text_output, render_tool_path, replace_tabs, str_or_none
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

READ_TOOL_SYSTEM_PROMPT_CONTRIBUTION: dict[str, Any] = {
    "snippet": "Read file contents",
    "guidelines": ("Use read to examine files instead of cat or sed.",),
}


@dataclass(slots=True)
class ReadToolDetails:
    truncation: TruncationResult | None = None


_COMPACT_RESOURCE_FILE_NAMES = {"AGENTS.override.md", "AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"}


def _format_read_line_range(args: dict | None, theme) -> str:
    args = args or {}
    if args.get("offset") is None and args.get("limit") is None:
        return ""
    start_line = args.get("offset") if args.get("offset") is not None else 1
    end_line = start_line + args["limit"] - 1 if args.get("limit") is not None else ""
    return theme.fg("warning", f":{start_line}{f'-{end_line}' if end_line else ''}")


def _format_read_call(args: dict | None, theme, cwd: str) -> str:
    args = args or {}
    path_display = render_tool_path(str_or_none(args.get("file_path", args.get("path"))), theme, cwd)
    return f"{theme.fg('toolTitle', theme.bold('read'))} {path_display}{_format_read_line_range(args, theme)}"


def _to_posix_path(path: str) -> str:
    return path.replace(os.sep, "/")


def _get_pi_docs_classification(absolute_path: str) -> dict | None:
    package_root = os.path.dirname(get_readme_path())
    relative_path = os.path.relpath(os.path.abspath(absolute_path), os.path.abspath(package_root))
    if relative_path == ".":
        relative_path = ""
    if (
        relative_path == ""
        or relative_path == ".."
        or relative_path.startswith(".." + os.sep)
        or os.path.isabs(relative_path)
    ):
        return None

    label = _to_posix_path(relative_path)
    if label == "README.md" or label.startswith(("docs/", "examples/")):
        return {"kind": "docs", "label": label}
    return None


def _get_compact_read_classification(args: dict | None, cwd: str) -> dict | None:
    raw_path = str_or_none((args or {}).get("file_path", (args or {}).get("path")))
    if not raw_path:
        return None

    absolute_path = resolve_to_cwd(raw_path, cwd)
    file_name = os.path.basename(absolute_path)
    if file_name == "SKILL.md":
        return {"kind": "skill", "label": os.path.basename(os.path.dirname(absolute_path)) or file_name}

    docs_classification = _get_pi_docs_classification(absolute_path)
    if docs_classification:
        return docs_classification

    if file_name in _COMPACT_RESOURCE_FILE_NAMES:
        return {"kind": "resource", "label": format_path_relative_to_cwd_or_absolute(absolute_path, cwd)}

    return None


def _format_compact_read_call(classification: dict, args: dict | None, theme) -> str:
    expand_hint = theme.fg("dim", f" ({key_text('app.tools.expand')} to expand)")
    if classification["kind"] == "skill":
        return (
            theme.fg("customMessageLabel", "\x1b[1m[skill]\x1b[22m ")
            + theme.fg("customMessageText", classification["label"])
            + _format_read_line_range(args, theme)
            + expand_hint
        )

    return (
        theme.fg("toolTitle", theme.bold(f"read {classification['kind']}"))
        + " "
        + theme.fg("accent", classification["label"])
        + _format_read_line_range(args, theme)
        + expand_hint
    )


def _trim_trailing_empty_lines(lines: list) -> list:
    end = len(lines)
    while end > 0 and lines[end - 1] == "":
        end -= 1
    return lines[:end]


def _details_truncation(details) -> TruncationResult | None:
    if details is None:
        return None
    if isinstance(details, dict):
        return details.get("truncation")
    return getattr(details, "truncation", None)


def _format_read_result(args, result, options, theme, show_images, _cwd, is_error) -> str:
    if not options.get("expanded") and not is_error:
        return ""

    args = args or {}
    raw_path = str_or_none(args.get("file_path", args.get("path")))
    output = get_text_output(result, show_images)
    lang = get_language_from_path(raw_path) if not is_error and raw_path else None
    rendered_lines = highlight_code(replace_tabs(output), lang) if lang else output.split("\n")
    lines = _trim_trailing_empty_lines(rendered_lines)
    max_lines = len(lines) if options.get("expanded") else 10
    display_lines = lines[:max_lines]
    remaining = len(lines) - max_lines
    body = "\n".join(
        replace_tabs(line) if lang else theme.fg("toolOutput", replace_tabs(line)) for line in display_lines
    )
    text = f"\n{body}"
    if remaining > 0:
        text += (
            theme.fg("muted", f"\n... ({remaining} more lines,")
            + " "
            + key_hint("app.tools.expand", "to expand")
            + theme.fg("muted", ")")
        )

    truncation = _details_truncation(result.get("details") if isinstance(result, dict) else result.details)
    if truncation is not None and truncation.truncated:
        if truncation.first_line_exceeds_limit:
            max_bytes = truncation.max_bytes if truncation.max_bytes is not None else DEFAULT_MAX_BYTES
            text += "\n" + theme.fg("warning", f"[First line exceeds {format_size(max_bytes)} limit]")
        elif truncation.truncated_by == "lines":
            max_lines_limit = truncation.max_lines if truncation.max_lines is not None else DEFAULT_MAX_LINES
            text += "\n" + theme.fg(
                "warning",
                f"[Truncated: showing {truncation.output_lines} of {truncation.total_lines} lines "
                f"({max_lines_limit} line limit)]",
            )
        else:
            max_bytes = truncation.max_bytes if truncation.max_bytes is not None else DEFAULT_MAX_BYTES
            text += "\n" + theme.fg(
                "warning",
                f"[Truncated: {truncation.output_lines} lines shown ({format_size(max_bytes)} limit)]",
            )
    return text


class LocalReadOperations:
    async def read_file(self, absolute_path: str) -> bytes:
        return await fs.Path(absolute_path).read_bytes()

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

    def render_call(args, theme, context):
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        classification = None if context["expanded"] else _get_compact_read_classification(args, context["cwd"])
        text.set_text(
            _format_compact_read_call(classification, args, theme)
            if classification
            else _format_read_call(args, theme, context["cwd"])
        )
        return text

    def render_result(result, options, theme, context):
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(
            _format_read_result(
                context["args"], result, options, theme, context["showImages"], context["cwd"], context["isError"]
            )
        )
        return text

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
        constrained_sampling=get_experimental_tool_sampling(),
        execute=execute,
        render_call=render_call,
        render_result=render_result,
    )


def create_read_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_read_tool_definition(cwd, **options))
