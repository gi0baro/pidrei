"""Mirror of pi coding-agent src/core/tools/write.ts."""

import os
from typing import Any

from tonio.colored import fs

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Container, Text

from ...modes.interactive.components.keybinding_hints import key_hint
from ...modes.interactive.theme import get_language_from_path, highlight_code
from ..extensions.types import ToolDefinition
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_to_cwd
from .render_utils import normalize_display_text, render_tool_path, replace_tabs, str_or_none
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition


class WriteCallRenderComponent(Text):
    """Text component carrying the streaming highlight cache."""

    def __init__(self) -> None:
        super().__init__("", 0, 0)
        self.cache: dict | None = None


WRITE_PARTIAL_FULL_HIGHLIGHT_LINES = 50


def _highlight_single_line(line: str, lang: str) -> str:
    highlighted = highlight_code(line, lang)
    return highlighted[0] if highlighted else ""


def _refresh_write_highlight_prefix(cache: dict) -> None:
    prefix_count = min(WRITE_PARTIAL_FULL_HIGHLIGHT_LINES, len(cache["normalizedLines"]))
    if prefix_count == 0:
        return
    prefix_source = "\n".join(cache["normalizedLines"][:prefix_count])
    prefix_highlighted = highlight_code(prefix_source, cache["lang"])
    for i in range(prefix_count):
        if i < len(prefix_highlighted):
            cache["highlightedLines"][i] = prefix_highlighted[i]
        else:
            cache["highlightedLines"][i] = _highlight_single_line(cache["normalizedLines"][i] or "", cache["lang"])


def _rebuild_write_highlight_cache_full(raw_path: str | None, file_content: str) -> dict | None:
    lang = get_language_from_path(raw_path) if raw_path else None
    if not lang:
        return None
    display_content = normalize_display_text(file_content)
    normalized = replace_tabs(display_content)
    return {
        "rawPath": raw_path,
        "lang": lang,
        "rawContent": file_content,
        "normalizedLines": normalized.split("\n"),
        "highlightedLines": highlight_code(normalized, lang),
    }


def _update_write_highlight_cache_incremental(
    cache: dict | None, raw_path: str | None, file_content: str
) -> dict | None:
    lang = get_language_from_path(raw_path) if raw_path else None
    if not lang:
        return None
    if cache is None:
        return _rebuild_write_highlight_cache_full(raw_path, file_content)
    if cache["lang"] != lang or cache["rawPath"] != raw_path:
        return _rebuild_write_highlight_cache_full(raw_path, file_content)
    if not file_content.startswith(cache["rawContent"]):
        return _rebuild_write_highlight_cache_full(raw_path, file_content)
    if len(file_content) == len(cache["rawContent"]):
        return cache

    delta_raw = file_content[len(cache["rawContent"]) :]
    delta_normalized = replace_tabs(normalize_display_text(delta_raw))
    cache["rawContent"] = file_content
    if not cache["normalizedLines"]:
        cache["normalizedLines"].append("")
        cache["highlightedLines"].append("")

    segments = delta_normalized.split("\n")
    cache["normalizedLines"][-1] += segments[0]
    cache["highlightedLines"][-1] = _highlight_single_line(cache["normalizedLines"][-1], cache["lang"])
    for segment in segments[1:]:
        cache["normalizedLines"].append(segment)
        cache["highlightedLines"].append(_highlight_single_line(segment, cache["lang"]))
    _refresh_write_highlight_prefix(cache)
    return cache


def _trim_trailing_empty_lines(lines: list) -> list:
    end = len(lines)
    while end > 0 and lines[end - 1] == "":
        end -= 1
    return lines[:end]


def _format_write_call(args: dict | None, options: dict, theme, cache: dict | None, cwd: str) -> str:
    args = args or {}
    raw_path = str_or_none(args.get("file_path", args.get("path")))
    file_content = str_or_none(args.get("content"))
    path_display = render_tool_path(raw_path, theme, cwd)
    text = f"{theme.fg('toolTitle', theme.bold('write'))} {path_display}"

    if file_content is None:
        text += "\n\n" + theme.fg("error", "[invalid content arg - expected string]")
    elif file_content:
        lang = get_language_from_path(raw_path) if raw_path else None
        if lang:
            rendered_lines = (
                cache["highlightedLines"]
                if cache is not None
                else highlight_code(replace_tabs(normalize_display_text(file_content)), lang)
            )
        else:
            rendered_lines = normalize_display_text(file_content).split("\n")
        lines = _trim_trailing_empty_lines(rendered_lines)
        total_lines = len(lines)
        max_lines = len(lines) if options.get("expanded") else 10
        display_lines = lines[:max_lines]
        remaining = len(lines) - max_lines
        body = "\n".join(line if lang else theme.fg("toolOutput", replace_tabs(line)) for line in display_lines)
        text += f"\n\n{body}"
        if remaining > 0:
            text += (
                theme.fg("muted", f"\n... ({remaining} more lines, {total_lines} total,")
                + " "
                + key_hint("app.tools.expand", "to expand")
                + theme.fg("muted", ")")
            )

    return text


def _format_write_result(result, is_error: bool, theme) -> str | None:
    if not is_error:
        return None
    content = result["content"] if isinstance(result, dict) else result.content
    output = "\n".join(
        (c.get("text") if isinstance(c, dict) else getattr(c, "text", None)) or ""
        for c in content
        if (c.get("type") if isinstance(c, dict) else getattr(c, "type", None)) == "text"
    )
    if not output:
        return None
    return "\n" + theme.fg("error", output)


WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path to the file to write (relative or absolute)"},
        "content": {"type": "string", "description": "Content to write to the file"},
    },
    "required": ["path", "content"],
}


class LocalWriteOperations:
    async def write_file(self, absolute_path: str, content: str) -> None:
        await fs.Path(absolute_path).write_text(content, encoding="utf-8", newline="")

    async def mkdir(self, dir: str) -> None:
        await fs.Path(dir).mkdir(parents=True, exist_ok=True)


def _js_string_length(text: str) -> int:
    """JS String.length (UTF-16 code units) for the byte-count message parity."""
    return sum(2 if ord(char) > 0xFFFF else 1 for char in text)


def create_write_tool_definition(cwd: str, *, operations: Any = None) -> ToolDefinition:
    ops = operations if operations is not None else LocalWriteOperations()

    async def execute(_tool_call_id, params, cancel=None, _on_update=None, _ctx=None):
        path = params["path"]
        content = params["content"]
        absolute_path = resolve_to_cwd(path, cwd)
        dir = os.path.dirname(absolute_path)

        async def run():
            # Do not release the mutation queue while an in-flight filesystem
            # operation may still finish: check cancellation after each await.
            def throw_if_aborted() -> None:
                if cancel is not None and cancel.cancelled:
                    raise Exception("Operation aborted")

            throw_if_aborted()
            # Create parent directories if needed.
            await ops.mkdir(dir)
            throw_if_aborted()

            # Write the file contents.
            await ops.write_file(absolute_path, content)
            throw_if_aborted()

            return AgentToolResult(
                content=[TextContent(text=f"Successfully wrote {_js_string_length(content)} bytes to {path}")],
                details=None,
            )

        return await with_file_mutation_queue(absolute_path, run)

    def render_call(args, theme, context):
        args = args or {}
        raw_path = str_or_none(args.get("file_path", args.get("path")))
        file_content = str_or_none(args.get("content"))
        component = (
            context["lastComponent"]
            if isinstance(context.get("lastComponent"), WriteCallRenderComponent)
            else WriteCallRenderComponent()
        )
        if file_content is not None:
            if context["argsComplete"]:
                component.cache = _rebuild_write_highlight_cache_full(raw_path, file_content)
            else:
                component.cache = _update_write_highlight_cache_incremental(component.cache, raw_path, file_content)
        else:
            component.cache = None
        component.set_text(
            _format_write_call(
                args,
                {"expanded": context["expanded"], "isPartial": context["isPartial"]},
                theme,
                component.cache,
                context["cwd"],
            )
        )
        return component

    def render_result(result, _options, theme, context):
        output = _format_write_result(result, context["isError"], theme)
        if not output:
            component = context["lastComponent"] if isinstance(context.get("lastComponent"), Container) else Container()
            component.clear()
            return component
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(output)
        return text

    return ToolDefinition(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. "
            "Automatically creates parent directories."
        ),
        prompt_snippet="Create or overwrite files",
        prompt_guidelines=["Use write only for new files or complete rewrites."],
        parameters=WRITE_SCHEMA,
        execute=execute,
        render_call=render_call,
        render_result=render_result,
    )


def create_write_tool(cwd: str, **options) -> WrappedDefinitionTool:
    return wrap_tool_definition(create_write_tool_definition(cwd, **options))
