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
import threading

import tonio.colored as tonio

from pidrei.config import APP_TITLE


BRAILLE_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
FRAME_INTERVAL_S = 0.08


async def extension(pi):
    state: dict = {"cancel": None}
    # Start and stop run from agent events and from session_shutdown (quit
    # does not abort the agent first), so they can overlap; the animation
    # ticks on its own task. The cancel swap and the title each one posts
    # happen under this lock, so the last title posted is the right one.
    title_guard = threading.Lock()

    def make_title(ctx, frame: str | None = None) -> str:
        cwd = os.path.basename(ctx.cwd)
        session = pi.get_session_name()
        base = f"{APP_TITLE} - {session} - {cwd}" if session else f"{APP_TITLE} - {cwd}"
        return f"{frame} {base}" if frame else base

    def stop_animation(ctx) -> None:
        with title_guard:
            cancel, state["cancel"] = state["cancel"], None
            if cancel is not None:
                cancel.set()
            ctx.ui.set_title(make_title(ctx))

    def start_animation(ctx) -> None:
        cancelled = tonio.Event()
        with title_guard:
            previous, state["cancel"] = state["cancel"], cancelled
            if previous is not None:
                previous.set()
            ctx.ui.set_title(make_title(ctx))

        async def animate() -> None:
            frame_index = 0
            while True:
                # Wake every tick, or immediately when cancelled.
                await cancelled.wait(FRAME_INTERVAL_S)
                with title_guard:
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
