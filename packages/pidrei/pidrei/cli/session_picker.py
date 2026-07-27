"""Mirror of pi coding-agent src/cli/session-picker.ts.

TUI session selector for the --resume flag.
"""

import os
from collections.abc import Awaitable, Callable
from typing import Any

import tonio.colored as tonio

from pidrei_tui import set_keybindings

from ..core.keybindings import KeybindingsManager
from ..modes.interactive.components.session_selector import SessionSelectorComponent
from .startup_ui import create_startup_tui, start_startup_tui


async def select_session(
    current_sessions_loader: Callable[[Callable[[int, int], None] | None], Awaitable[list[Any]]],
    all_sessions_loader: Callable[[Callable[[int, int], None] | None], Awaitable[list[Any]]],
    settings_manager: Any,
) -> str | None:
    """Show TUI session selector; returns the selected session path or None if cancelled."""
    ui = await create_startup_tui(settings_manager)
    keybindings = await KeybindingsManager.create()
    set_keybindings(keybindings)

    done = tonio.Event()
    outcome: dict = {"path": None}
    settled = False

    async def finish(path: str | None) -> None:
        nonlocal settled
        if settled:
            return
        settled = True
        outcome["path"] = path
        await ui.stop()
        done.set()

    def on_select(path: str) -> None:
        tonio.spawn.without_tracking(finish(path))

    def on_cancel() -> None:
        tonio.spawn.without_tracking(finish(None))

    def on_exit() -> None:
        async def stop_and_exit() -> None:
            await ui.stop()
            os._exit(0)

        tonio.spawn.without_tracking(stop_and_exit())

    selector = SessionSelectorComponent(
        current_sessions_loader,
        all_sessions_loader,
        on_select,
        on_cancel,
        on_exit,
        lambda: ui.request_render(),
        {"showRenameHint": False, "keybindings": keybindings},
    )

    ui.add_child(selector)
    ui.set_focus(selector.get_session_list())
    await start_startup_tui(ui, settings_manager)
    await done.wait(None)
    return outcome["path"]
