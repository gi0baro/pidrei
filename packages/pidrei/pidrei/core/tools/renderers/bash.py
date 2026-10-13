"""Mirror of pi coding-agent src/core/tools/renderers/bash.ts.

Presentation for the shell tools.

Renderers live apart from the implementation so a process that only displays
tool output does not load the execution path or its parameter schema.
`bash.py` spreads these into the shell tool definition, so the tool's public
shape is unchanged.
"""

from pidrei_tui import Container, Spacer, Text

from ....modes.interactive.components.keybinding_hints import key_hint
from ....modes.interactive.components.visual_truncate import VisualLinePreview
from ....modes.interactive.theme import theme
from ..render_utils import get_text_output, invalid_arg_text, start_tool_timer, str_or_none, tool_timer_text
from ..truncate import DEFAULT_MAX_BYTES, format_size
from .types import ToolRenderers


BASH_PREVIEW_LINES = 5
BASH_UPDATE_THROTTLE_S = 0.1


def _format_shell_call(args: dict | None, prompt: str) -> str:
    args = args or {}
    command = str_or_none(args.get("command"))
    timeout = args.get("timeout")
    timeout_suffix = theme.fg("muted", f" (timeout {timeout}s)") if timeout else ""
    if command is None:
        command_display = invalid_arg_text(theme)
    elif command:
        command_display = command
    else:
        command_display = theme.fg("toolOutput", "...")
    return theme.fg("toolTitle", theme.bold(f"{prompt} {command_display}")) + timeout_suffix


def _result_details(result):
    details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
    return details


def _rebuild_bash_result_render_component(
    component: Container,
    result,
    options: dict,
    show_images: bool,
    timer_text: str | None,
) -> None:
    component.clear()

    output = get_text_output(result, show_images).strip()
    details = _result_details(result)
    truncation = getattr(details, "truncation", None) if details is not None else None
    full_output_path = getattr(details, "full_output_path", None) if details is not None else None
    if (
        not options.get("isPartial")
        and truncation is not None
        and truncation.truncated
        and full_output_path
        and output.endswith("]")
    ):
        footer_start = output.rfind("\n\n[")
        if footer_start != -1 and full_output_path in output[footer_start:]:
            output = output[:footer_start].rstrip()

    if output:
        styled_output = "\n".join(theme.fg("toolOutput", line) for line in output.split("\n"))

        if options.get("expanded"):
            component.add_child(Text(f"\n{styled_output}", 0, 0))
        else:
            component.add_child(Spacer(1))
            component.add_child(
                VisualLinePreview(
                    text=styled_output,
                    max_visual_lines=BASH_PREVIEW_LINES,
                    keep="end",
                    format_hint=lambda hidden: (
                        theme.fg("muted", f"... ({hidden} earlier lines,")
                        + f" {key_hint('app.tools.expand', 'to expand')}"
                        + theme.fg("muted", ")")
                    ),
                )
            )

    if (truncation is not None and truncation.truncated) or full_output_path:
        warnings: list = []
        if full_output_path:
            warnings.append(f"Full output: {full_output_path}")
        if truncation is not None and truncation.truncated:
            if truncation.truncated_by == "lines":
                warnings.append(f"Truncated: showing {truncation.output_lines} of {truncation.total_lines} lines")
            else:
                max_bytes = truncation.max_bytes if truncation.max_bytes is not None else DEFAULT_MAX_BYTES
                warnings.append(f"Truncated: {truncation.output_lines} lines shown ({format_size(max_bytes)} limit)")
        component.add_child(Text("\n" + theme.fg("warning", f"[{'. '.join(warnings)}]"), 0, 0))

    if timer_text:
        component.add_child(Text("\n" + theme.fg("muted", timer_text), 0, 0))


def create_shell_renderers(prompt: str) -> ToolRenderers:
    def render_call(args, _theme, context):
        start_tool_timer(context)
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_shell_call(args, prompt))
        return text

    def render_result(result, options, _theme, context):
        timer_text = tool_timer_text(context, bool(options.get("isPartial")))
        component = context["lastComponent"] if isinstance(context.get("lastComponent"), Container) else Container()
        _rebuild_bash_result_render_component(component, result, options, context["showImages"], timer_text)
        component.invalidate()
        return component

    return ToolRenderers(render_call=render_call, render_result=render_result)
