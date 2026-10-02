"""Mirror of pi coding-agent src/extensions/codemode/renderer.ts.

Presentation for the codemode tool.

The call shows the script; the result lists the nested tool calls with their
status as they run and the cost of its model calls, followed by the script
output without the "Script completed" header. Nested calls are not separate
tool rows because they never reach the model as tool calls.

Live results carry `CodemodeToolDetails`; results read back from a session file
carry its wire form, a dict with camelCase keys. Both render the same.
"""

import re
from typing import Any

from pidrei_tui import Container, Spacer, Text

from ...core.tools.render_utils import get_text_output, replace_tabs, str_or_none
from ...core.tools.renderers.types import ToolRenderers
from ...modes.interactive.components.keybinding_hints import key_hint
from ...modes.interactive.components.visual_truncate import VisualLinePreview
from ...modes.interactive.theme import highlight_code, theme


_CODE_PREVIEW_LINES = 10
_CALL_PREVIEW_COUNT = 8
_OUTPUT_PREVIEW_LINES = 5
_COLLAPSED_ARGS_CHARS = 80
_SCRIPT_HEADER = re.compile(r"^Script (completed|failed)\nWall time [\d.]+ seconds\nOutput:\n$")


def _field(value: Any, key: str, attribute: str | None = None) -> Any:
    """A field of details or of a nested call: a dict key (wire form) or an
    attribute (live)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, attribute or key, None)


def _expand_hint(hidden: int, noun: str) -> str:
    return (
        theme.fg("muted", f"... ({hidden} more {noun},")
        + f" {key_hint('app.tools.expand', 'to expand')}"
        + theme.fg("muted", ")")
    )


def _format_duration(ms: float | None) -> str:
    if ms is None:
        return ""
    return f"{round(ms)}ms" if ms < 1000 else f"{ms / 1000:.1f}s"


def _format_cost(cost: float) -> str:
    """Cents for larger amounts, two significant digits for the fractions of a
    cent classifier calls cost."""
    if cost >= 0.01:
        return f"${cost:.2f}"
    # JS `toPrecision(2)`: two significant digits. The exponent is taken after
    # rounding to them, so 0.0099999 gives 0.010.
    digits = max(0, 1 - int(f"{cost:.1e}".split("e")[1])) if cost else 1
    return f"${cost:.{digits}f}"


def _status_icon(status: str) -> str:
    match status:
        case "running":
            return theme.fg("warning", "…")
        case "ok":
            return theme.fg("success", "✓")
        case "error":
            return theme.fg("error", "✗")
        case _:
            return theme.fg("muted", "⊘")


def _format_call(call: Any, expanded: bool) -> str:
    args = _field(call, "args") or ""
    if not expanded and len(args) > _COLLAPSED_ARGS_CHARS:
        args = f"{args[: _COLLAPSED_ARGS_CHARS - 3]}..."
    duration = _format_duration(_field(call, "durationMs", "duration_ms"))
    cost = _field(call, "cost")
    error = _field(call, "error")
    line = f"{_status_icon(_field(call, 'status'))} {theme.fg('toolTitle', _field(call, 'name'))}"
    if args:
        line += f" {theme.fg('muted', args)}"
    if duration:
        line += f" {theme.fg('dim', duration)}"
    if cost:
        line += f" {theme.fg('dim', _format_cost(cost))}"
    if expanded and error:
        line += "\n    " + theme.fg("error", "\n    ".join(error.split("\n")))
    return line


def _render_call(args, _theme, context):
    # The code includes the `# @options:` line, so options show as part of the script.
    code = str_or_none(args.get("code") if isinstance(args, dict) else None)
    title = theme.fg("toolTitle", theme.bold("codemode"))
    component = context["lastComponent"] if isinstance(context.get("lastComponent"), Container) else Container()
    component.clear()
    if code is None:
        component.add_child(Text(f"{title} {theme.fg('error', '[invalid arg]')}", 0, 0))
        return component
    component.add_child(Text(title, 0, 0))
    if code:
        highlighted = "\n".join(highlight_code(replace_tabs(code.replace("\r", "").rstrip()), "python"))
        component.add_child(
            Text(highlighted, 0, 0)
            if context["expanded"]
            else VisualLinePreview(
                text=highlighted,
                max_visual_lines=_CODE_PREVIEW_LINES,
                keep="start",
                format_hint=lambda hidden: _expand_hint(hidden, "lines"),
            )
        )
    return component


def _render_result(result, options, _theme, context):
    component = context["lastComponent"] if isinstance(context.get("lastComponent"), Container) else Container()
    component.clear()
    expanded = bool(options.get("expanded"))
    details = result.get("details") if isinstance(result, dict) else result.details
    calls = list(_field(details, "calls") or [])
    if calls:
        shown = calls if expanded else calls[-_CALL_PREVIEW_COUNT:]
        lines = [_format_call(call, expanded) for call in shown]
        if len(shown) < len(calls):
            lines.insert(
                0,
                theme.fg("muted", f"... ({len(calls) - len(shown)} earlier calls,")
                + f" {key_hint('app.tools.expand', 'to expand')}"
                + theme.fg("muted", ")"),
            )
        # Collapsed rows hide earlier calls, so the total covers every call.
        priced = [cost for call in calls if (cost := _field(call, "cost"))]
        if len(priced) > 1:
            lines.append(theme.fg("muted", f"Model calls: {_format_cost(sum(priced))}"))
        component.add_child(Spacer(1))
        component.add_child(Text("\n".join(lines), 0, 0))

    # Drop the "Script completed\nWall time ...\nOutput:\n" header. Rejected
    # input (invalid options) has no header.
    content = list(result["content"] if isinstance(result, dict) else result.content)
    first = content[0] if content else None
    first_text = _field(first, "text") if _field(first, "type") == "text" else None
    has_header = isinstance(first_text, str) and _SCRIPT_HEADER.match(first_text) is not None
    output = (
        ""
        if options.get("isPartial")
        else get_text_output({"content": content[1:] if has_header else content}, context["showImages"]).strip()
    )
    if output:
        color = "error" if context["isError"] else "toolOutput"
        styled = "\n".join(theme.fg(color, line) for line in replace_tabs(output).split("\n"))
        component.add_child(Spacer(1))
        if expanded:
            component.add_child(Text(styled, 0, 0))
        else:
            # Limit wrapped lines, not logical ones: script output is often one long JSON line.
            component.add_child(
                VisualLinePreview(
                    text=styled,
                    max_visual_lines=_OUTPUT_PREVIEW_LINES,
                    keep="start",
                    format_hint=lambda hidden: _expand_hint(hidden, "lines"),
                )
            )
            # The collapsed preview hides the truncation notice at the end, so name the file here.
            full_output_path = _field(details, "fullOutputPath", "full_output_path")
            if full_output_path:
                component.add_child(Text(theme.fg("muted", f"Full output: {full_output_path}"), 0, 0))
    return component


CODEMODE_RENDERERS = ToolRenderers(render_call=_render_call, render_result=_render_result)
