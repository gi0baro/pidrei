"""Titlebar Spinner

Shows a braille spinner animation in the terminal title while the agent is
working, via `ctx.ui.set_title()`.

pi drives the animation with `setInterval`; here it is a cooperative tonio
task that ticks until a cancel event is set (the same pattern as
`pidrei_tui`'s internal timers), stopped on `agent_settled` and
`session_shutdown`.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/titlebar_spinner.py
"""

import os

import tonio.colored as tonio

from pidrei.config import APP_TITLE


BRAILLE_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
FRAME_INTERVAL_S = 0.08


def extension(pi):
    state: dict = {"cancel": None}

    def make_title(ctx, frame: str | None = None) -> str:
        cwd = os.path.basename(ctx.cwd)
        session = pi.get_session_name()
        base = f"{APP_TITLE} - {session} - {cwd}" if session else f"{APP_TITLE} - {cwd}"
        return f"{frame} {base}" if frame else base

    def stop_animation(ctx) -> None:
        if state["cancel"] is not None:
            state["cancel"].set()
            state["cancel"] = None
        ctx.ui.set_title(make_title(ctx))

    def start_animation(ctx) -> None:
        stop_animation(ctx)
        cancelled = tonio.Event()
        state["cancel"] = cancelled

        async def animate() -> None:
            frame_index = 0
            while True:
                # Wake every tick, or immediately when cancelled.
                await cancelled.wait(FRAME_INTERVAL_S)
                if cancelled.is_set():
                    return
                frame = BRAILLE_FRAMES[frame_index % len(BRAILLE_FRAMES)]
                ctx.ui.set_title(make_title(ctx, frame))
                frame_index += 1

        tonio.spawn.without_tracking(animate())

    async def on_agent_start(_event, ctx) -> None:
        if not ctx.has_ui:
            return
        start_animation(ctx)

    async def on_agent_settled(_event, ctx) -> None:
        stop_animation(ctx)

    async def on_session_shutdown(_event, ctx) -> None:
        stop_animation(ctx)

    pi.on("agent_start", on_agent_start)
    pi.on("agent_settled", on_agent_settled)
    pi.on("session_shutdown", on_session_shutdown)
