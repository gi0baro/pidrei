"""Port of pi coding-agent src/cli/config-selector.ts.

The TUI driver behind `pidrei config`. `ConfigSelectorComponent` was ported with
the rest of the TUI in Phase 4 and has had no caller until now — this is its
only one, in pi too.
"""

import sys
from typing import Any

import tonio.colored as tonio

from pidrei_tui import TUI, ProcessTerminal, TuiMainScreen

from ..modes.interactive.components.config_selector import ConfigSelectorComponent
from ..modes.interactive.theme import init_theme, stop_theme_watcher


async def select_config(
    *,
    resolved_paths: dict,
    settings_manager: Any,
    cwd: str,
    agent_dir: str,
    write_scope: str = "global",
    project_mode_available: bool = True,
) -> None:
    """Run the config TUI until the user closes it."""
    await init_theme(settings_manager.get_theme(), True)

    ui: TUI = TuiMainScreen(ProcessTerminal(), settings_manager.get_show_hardware_cursor(), agent_dir)
    ui.set_clear_on_shrink(settings_manager.get_clear_on_shrink())
    closed = tonio.Event()

    # `TUI.start`/`stop` are async; these callbacks are sync (the component
    # invokes them directly), so the stop is spawned. Pre-existing bug found
    # while auditing the render path: both coroutines used to be dropped
    # outright, so the config TUI never started and never restored the terminal.
    def finish() -> None:
        if not closed.is_set():
            tonio.spawn.without_tracking(ui.stop())
            stop_theme_watcher()
            closed.set()

    def exit_now() -> None:
        tonio.spawn.without_tracking(ui.stop())
        stop_theme_watcher()
        sys.exit(0)

    selector = ConfigSelectorComponent(
        resolved_paths,
        settings_manager,
        cwd,
        agent_dir,
        finish,
        exit_now,
        ui.request_render,
        ui.terminal.rows,
        write_scope,
        project_mode_available,
    )

    ui.add_child(selector)
    ui.set_focus(selector.get_resource_list())
    await ui.start()
    await closed.wait()
