"""Border Status Editor

Replaces the footer with an empty component and moves all status information
onto the editor borders: a spinner on the top border while the agent works,
model/thinking level on the bottom left, context usage, cwd and git branch on
the bottom right.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/border_status_editor.py
"""

import os

import tonio.colored as tonio

from pidrei.modes.interactive.components import CustomEditor
from pidrei_tui import truncate_to_width, visible_width


SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SPINNER_INTERVAL_S = 0.08


def fit_border(left: str, right: str, width: int, border, fill=None) -> str:
    """Lay `left` and `right` into a horizontal border line of `width` cells."""
    if fill is None:
        fill = border
    if width <= 0:
        return ""
    if width == 1:
        return border("─")

    left_text = left
    right_text = right
    fixed_width = 2
    minimum_gap = 3

    while (
        fixed_width + visible_width(left_text) + visible_width(right_text) + minimum_gap > width
        and visible_width(right_text) > 0
    ):
        right_text = truncate_to_width(right_text, max(0, visible_width(right_text) - 1), "")
    while (
        fixed_width + visible_width(left_text) + visible_width(right_text) + minimum_gap > width
        and visible_width(left_text) > 0
    ):
        left_text = truncate_to_width(left_text, max(0, visible_width(left_text) - 1), "")

    gap_width = max(0, width - fixed_width - visible_width(left_text) - visible_width(right_text))
    return f"{border('─')}{left_text}{fill('─' * gap_width)}{right_text}{border('─')}"


def format_cwd(cwd: str) -> str:
    home = os.environ.get("HOME")
    if home and cwd.startswith(home):
        return f"~{cwd[len(home) :]}"
    return cwd


def format_context(ctx) -> str:
    usage = ctx.get_context_usage()
    model = ctx.model
    context_window = (
        usage.context_window if usage is not None else (model.context_window if model is not None else None)
    )
    if not context_window or usage is None or usage.percent is None:
        return "ctx ?"
    return f"ctx {round(usage.percent)}%/{context_window / 1000:.0f}k"


class EmptyFooter:
    def render(self, _width: int) -> list[str]:
        return []

    def invalidate(self) -> None:
        pass


def extension(pi):
    state = {"is_working": False, "spinner_index": 0, "spinner_stop": None, "tui": None}

    def request_render() -> None:
        if state["tui"] is not None:
            state["tui"].request_render()

    def stop_spinner() -> None:
        if state["spinner_stop"] is not None:
            state["spinner_stop"].set()
            state["spinner_stop"] = None

    async def on_agent_start(_event, _ctx) -> None:
        state["is_working"] = True
        stop_spinner()

        # pi drives the spinner with setInterval; here it is a background task
        # that ends cooperatively through the Event.
        stop = tonio.Event()
        state["spinner_stop"] = stop

        async def spin() -> None:
            while True:
                await stop.wait(SPINNER_INTERVAL_S)
                if stop.is_set():
                    return
                state["spinner_index"] = (state["spinner_index"] + 1) % len(SPINNER_FRAMES)
                request_render()

        tonio.spawn.without_tracking(spin())
        request_render()

    async def on_agent_end(_event, _ctx) -> None:
        state["is_working"] = False
        stop_spinner()
        request_render()

    async def on_session_shutdown(_event, _ctx) -> None:
        stop_spinner()
        state["tui"] = None

    async def on_session_start(_event, ctx) -> None:
        ctx.ui.set_working_visible(False)
        ctx.ui.set_footer(lambda _ui, _theme, _data: EmptyFooter())

        branch_state = {"branch": None}

        async def refresh_branch() -> None:
            try:
                result = await pi.exec("git", ["branch", "--show-current"], cwd=ctx.cwd)
                stdout = result.stdout.strip()
            except Exception:
                stdout = ""
            branch_state["branch"] = stdout or None
            request_render()

        tonio.spawn.without_tracking(refresh_branch())

        class BorderStatusEditor(CustomEditor):
            def __init__(self, tui, theme, keybindings) -> None:
                super().__init__(tui, theme, keybindings, {"paddingX": 0})
                state["tui"] = tui

            def render(self, width: int) -> list[str]:
                lines = super().render(width)
                if len(lines) < 2:
                    return lines

                thm = ctx.ui.theme
                model = ctx.model
                model_label = f"{model.provider}/{model.id}" if model is not None else "no model"
                thinking = pi.get_thinking_level()
                top_left = (
                    thm.fg("accent", f" {SPINNER_FRAMES[state['spinner_index']]} ") if state["is_working"] else ""
                )
                branch = branch_state["branch"]
                bottom_left = thm.fg("muted", f" {model_label} · {thinking} ")
                bottom_right = thm.fg(
                    "muted",
                    f" {format_context(ctx)} · {format_cwd(ctx.cwd)}{f' ({branch})' if branch else ''} ",
                )

                lines[0] = fit_border(top_left, "", width, self.border_color)
                lines[-1] = fit_border(bottom_left, bottom_right, width, self.border_color)
                return lines

        ctx.ui.set_editor_component(lambda tui, theme, keybindings: BorderStatusEditor(tui, theme, keybindings))

    pi.on("agent_start", on_agent_start)
    pi.on("agent_end", on_agent_end)
    pi.on("session_shutdown", on_session_shutdown)
    pi.on("session_start", on_session_start)
