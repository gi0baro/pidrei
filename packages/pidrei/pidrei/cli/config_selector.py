"""Port of pi coding-agent src/cli/config-selector.ts.

The TUI driver behind `pidrei config`. `ConfigSelectorComponent` was ported with
the rest of the TUI in Phase 4 and has had no caller until now — this is its
only one, in pi too.
"""

from typing import Any

import tonio.colored as tonio

from pidrei_tui import TUI, ProcessTerminal, TuiMainScreen, prime_capabilities

from ..core.output_guard import attach_terminal, detach_terminal
from ..modes.interactive.components.config_selector import ConfigSelectorComponent, build_canonical_path_map_blocking
from ..modes.interactive.theme import init_theme, stop_theme_watcher
from ..utils.fd_io import hard_exit
from ..utils.process import probe_tmux_hyperlinks


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
    # `ui.start()` and the render paths read the capabilities: settle them
    # (under tmux, a subprocess probe) first.
    await prime_capabilities(probe_tmux_hyperlinks)
    canonical_by_path = await tonio.spawn_blocking(build_canonical_path_map_blocking, resolved_paths)

    terminal = ProcessTerminal()
    ui: TUI = TuiMainScreen(terminal, settings_manager.get_show_hardware_cursor(), agent_dir)
    ui.set_clear_on_shrink(settings_manager.get_clear_on_shrink())
    # The terminal is the tty's one writer until it is closed below.
    await attach_terminal(terminal)
    closed = tonio.Event()

    # The component calls these from its input handling. pi stops the UI in
    # place there; here the stop is the key's completion, which the next key
    # waits for (spec/ui-island.md, stopping the UI from inside a key), and
    # `closed` is set only once the
    # terminal is restored.
    finishing = False

    async def close_terminal() -> None:
        detach_terminal()
        await ui.close()

    async def stop_and_close() -> None:
        await ui.stop()
        stop_theme_watcher()
        await close_terminal()
        closed.set()

    def finish() -> None:
        nonlocal finishing
        if finishing:
            return
        finishing = True
        ui.finish_before_next_input(tonio.spawn(stop_and_close()))

    async def stop_and_exit() -> None:
        await ui.stop()
        stop_theme_watcher()
        await close_terminal()
        hard_exit(0)

    def exit_now() -> None:
        ui.finish_before_next_input(tonio.spawn(stop_and_exit()))

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
        canonical_by_path=canonical_by_path,
    )

    ui.add_child(selector)
    ui.set_focus(selector.get_resource_list())
    await ui.start()
    await closed.wait()
