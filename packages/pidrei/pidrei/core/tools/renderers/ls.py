"""Mirror of pi coding-agent src/core/tools/renderers/ls.ts.

Presentation for the ls tool.

Renderers live apart from the implementation so a process that only displays
tool output does not load the execution path or its parameter schema.
`ls.py` spreads these into its definition, so the tool's public shape is
unchanged.
"""

from pidrei_tui import Text

from ....modes.interactive.components.keybinding_hints import key_hint
from ..render_utils import get_text_output, render_tool_path, str_or_none
from ..truncate import DEFAULT_MAX_BYTES, format_size
from .types import ToolRenderers


def _format_ls_call(args: dict | None, theme, cwd: str) -> str:
    args = args or {}
    limit = args.get("limit")
    path_display = render_tool_path(str_or_none(args.get("path")), theme, cwd, {"emptyFallback": "."})
    text = f"{theme.fg('toolTitle', theme.bold('ls'))} {path_display}"
    if limit is not None:
        text += theme.fg("toolOutput", f" (limit {limit})")
    return text


def _format_ls_result(result, options: dict, theme, show_images: bool) -> str:
    output = get_text_output(result, show_images).strip()
    text = ""
    if output:
        lines = output.split("\n")
        max_lines = len(lines) if options.get("expanded") else 20
        display_lines = lines[:max_lines]
        remaining = len(lines) - max_lines
        text += "\n" + "\n".join(theme.fg("toolOutput", line) for line in display_lines)
        if remaining > 0:
            text += (
                theme.fg("muted", f"\n... ({remaining} more lines,")
                + " "
                + key_hint("app.tools.expand", "to expand")
                + theme.fg("muted", ")")
            )

    details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
    entry_limit = getattr(details, "entry_limit_reached", None) if details is not None else None
    truncation = getattr(details, "truncation", None) if details is not None else None
    if entry_limit or (truncation is not None and truncation.truncated):
        warnings = []
        if entry_limit:
            warnings.append(f"{entry_limit} entries limit")
        if truncation is not None and truncation.truncated:
            max_bytes = truncation.max_bytes if truncation.max_bytes is not None else DEFAULT_MAX_BYTES
            warnings.append(f"{format_size(max_bytes)} limit")
        text += "\n" + theme.fg("warning", f"[Truncated: {', '.join(warnings)}]")
    return text


def _render_call(args, theme, context):
    text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
    text.set_text(_format_ls_call(args, theme, context["cwd"]))
    return text


def _render_result(result, options, theme, context):
    text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
    text.set_text(_format_ls_result(result, options, theme, context["showImages"]))
    return text


ls_renderers = ToolRenderers(render_call=_render_call, render_result=_render_result)
