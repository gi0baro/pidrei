"""Mirror of pi coding-agent src/core/tools/renderers/read.ts.

Presentation for the read tool.

Renderers live apart from the implementation so a process that only displays
tool output does not load the execution path or its parameter schema.
`read.py` spreads these into its definition, so the tool's public shape is
unchanged.
"""

import os

from pidrei_tui import Text

from ....config import get_readme_path
from ....modes.interactive.components.keybinding_hints import key_hint, key_text
from ....modes.interactive.theme import get_language_from_path, highlight_code
from ....utils.paths import format_path_relative_to_cwd_or_absolute
from ..path_utils import resolve_to_cwd
from ..render_utils import get_text_output, render_tool_path, replace_tabs, str_or_none
from ..truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size
from .types import ToolRenderers


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


def _render_call(args, theme, context):
    text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
    classification = None if context["expanded"] else _get_compact_read_classification(args, context["cwd"])
    text.set_text(
        _format_compact_read_call(classification, args, theme)
        if classification
        else _format_read_call(args, theme, context["cwd"])
    )
    return text


def _render_result(result, options, theme, context):
    text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
    text.set_text(
        _format_read_result(
            context["args"], result, options, theme, context["showImages"], context["cwd"], context["isError"]
        )
    )
    return text


read_renderers = ToolRenderers(render_call=_render_call, render_result=_render_result)
