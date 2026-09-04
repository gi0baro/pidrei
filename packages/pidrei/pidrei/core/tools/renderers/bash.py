"""Mirror of pi coding-agent src/core/tools/renderers/bash.ts.

Presentation for the shell tools.

Renderers live apart from the implementation so a process that only displays
tool output does not load the execution path or its parameter schema.
`bash.py` spreads these into the shell tool definition, so the tool's public
shape is unchanged.
"""

import time

from pidrei_tui import Container, Text, truncate_to_width
from pidrei_tui._timers import Interval

from ....modes.interactive.components.keybinding_hints import key_hint
from ....modes.interactive.components.visual_truncate import truncate_to_visual_lines
from ....modes.interactive.theme import theme
from ..render_utils import get_text_output, invalid_arg_text, str_or_none
from ..truncate import DEFAULT_MAX_BYTES, format_size
from .types import ToolRenderers


BASH_PREVIEW_LINES = 5
BASH_UPDATE_THROTTLE_S = 0.1


class BashResultRenderComponent(Container):
    def __init__(self) -> None:
        super().__init__()
        self.state = {"cachedWidth": None, "cachedLines": None, "cachedSkipped": None}


class _BashPreviewOutput:
    """Width-aware cached preview (pi's inline render object)."""

    def __init__(self, state: dict, styled_output: str) -> None:
        self._state = state
        self._styled_output = styled_output

    def render(self, width: int) -> list:
        state = self._state
        if state["cachedLines"] is None or state["cachedWidth"] != width:
            preview = truncate_to_visual_lines(self._styled_output, BASH_PREVIEW_LINES, width)
            state["cachedLines"] = preview["visualLines"]
            state["cachedSkipped"] = preview["skippedCount"]
            state["cachedWidth"] = width
        if state["cachedSkipped"]:
            hint = (
                theme.fg("muted", f"... ({state['cachedSkipped']} earlier lines,")
                + f" {key_hint('app.tools.expand', 'to expand')}"
                + theme.fg("muted", ")")
            )
            return ["", truncate_to_width(hint, width, "..."), *(state["cachedLines"] or [])]
        return ["", *(state["cachedLines"] or [])]

    def invalidate(self) -> None:
        self._state["cachedWidth"] = None
        self._state["cachedLines"] = None
        self._state["cachedSkipped"] = None


def _format_duration(ms: float) -> str:
    return f"{ms / 1000:.1f}s"


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
    component: BashResultRenderComponent,
    result,
    options: dict,
    show_images: bool,
    started_at: float | None,
    ended_at: float | None,
) -> None:
    state = component.state
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
            component.add_child(_BashPreviewOutput(state, styled_output))

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

    if started_at is not None:
        label = "Elapsed" if options.get("isPartial") else "Took"
        end_time = ended_at if ended_at is not None else time.time() * 1000
        component.add_child(Text("\n" + theme.fg("muted", f"{label} {_format_duration(end_time - started_at)}"), 0, 0))


def create_shell_renderers(prompt: str) -> ToolRenderers:
    def render_call(args, _theme, context):
        state = context["state"]
        if context["executionStarted"] and state.get("startedAt") is None:
            state["startedAt"] = time.time() * 1000
            state["endedAt"] = None
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_shell_call(args, prompt))
        return text

    def render_result(result, options, _theme, context):
        state = context["state"]
        if state.get("startedAt") is not None and options.get("isPartial") and not state.get("interval"):
            invalidate = context["invalidate"]

            async def tick() -> None:
                invalidate()

            state["interval"] = Interval(1000, tick)
        if not options.get("isPartial") or context["isError"]:
            if state.get("endedAt") is None:
                state["endedAt"] = time.time() * 1000
            if state.get("interval"):
                state["interval"].cancel()
                state["interval"] = None
        component = (
            context["lastComponent"]
            if isinstance(context.get("lastComponent"), BashResultRenderComponent)
            else BashResultRenderComponent()
        )
        _rebuild_bash_result_render_component(
            component, result, options, context["showImages"], state.get("startedAt"), state.get("endedAt")
        )
        component.invalidate()
        return component

    return ToolRenderers(render_call=render_call, render_result=render_result)
