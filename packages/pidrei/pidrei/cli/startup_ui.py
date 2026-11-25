"""Mirror of pi coding-agent src/cli/startup-ui.ts."""

import contextlib
import os

import tonio.colored as tonio

from pidrei_tui import (
    TUI,
    ProcessTerminal,
    ProgramStatus,
    TuiMainScreen,
    prime_capabilities,
    set_capability_overrides,
    set_keybindings,
)

from ..config import APP_NAME, get_agent_dir
from ..core.keybindings import KeybindingsManager
from ..core.output_guard import attach_terminal, detach_terminal
from ..core.package_manager import DefaultPackageManager
from ..core.settings_manager import SettingsManager
from ..modes.interactive.components.extension_input import ExtensionInputComponent
from ..modes.interactive.components.extension_selector import ExtensionSelectorComponent
from ..modes.interactive.program_status_reporter import BlockedStatus, dialog_blocked_status
from ..modes.interactive.theme import (
    SYSTEM_THEME_NAME,
    _load_theme_from_path_blocking,
    get_terminal_theme,
    init_theme,
    mark_terminal_colors_pending,
    prime_theme_cache,
    request_terminal_colors,
    resolve_theme_setting,
    set_registered_themes,
    set_terminal_colors,
    set_theme,
)
from ..utils.process import probe_tmux_hyperlinks


# pi's first-time setup (`shouldRunFirstTimeSetup`, `showFirstTimeSetup`) is not ported: since
# pi 1.6.1 it only asks the analytics opt-in, and pidrei has no telemetry.


def _load_themes_blocking(resources: list) -> list:
    themes: list = []
    seen: set = set()
    for resource in resources:
        if not resource.enabled:
            continue
        # Startup prompts should not fail because a theme is broken. The
        # normal resource loader reports theme diagnostics later in startup.
        with contextlib.suppress(Exception):
            loaded_theme = _load_theme_from_path_blocking(resource.path)
            if loaded_theme.name:
                if loaded_theme.name in seen:
                    continue
                seen.add(loaded_theme.name)
            themes.append(loaded_theme)
    return themes


async def _load_startup_themes(settings_manager: SettingsManager) -> list:
    global_settings_manager = SettingsManager.in_memory(settings_manager.get_global_settings(), project_trusted=False)
    package_manager = DefaultPackageManager(
        cwd=os.getcwd(),
        agent_dir=get_agent_dir(),
        settings_manager=global_settings_manager,
    )
    resolved_paths = await package_manager.resolve()
    return await tonio.spawn_blocking(_load_themes_blocking, resolved_paths.themes)


async def create_startup_tui(settings_manager: SettingsManager) -> TUI:
    set_capability_overrides(settings_manager.get_terminal_capability_overrides())
    # Warm the caches that sync render/callback paths read from, so neither
    # the builtin-theme files nor the tmux capability probe is ever needed
    # from them later.
    await prime_theme_cache()
    await prime_capabilities(probe_tmux_hyperlinks)
    set_registered_themes(await _load_startup_themes(settings_manager))
    # The system theme starts in grayscale until the terminal reports its colors.
    mark_terminal_colors_pending()
    await init_theme(_resolve_startup_theme(settings_manager.get_theme_setting()))
    set_keybindings(await KeybindingsManager())
    terminal = ProcessTerminal()
    ui: TUI = TuiMainScreen(terminal, settings_manager.get_show_hardware_cursor(), get_agent_dir())
    ui.set_clear_on_shrink(settings_manager.get_clear_on_shrink())
    # The terminal is the tty's one writer until `close_startup_tui`.
    await attach_terminal(terminal)
    return ui


async def close_startup_tui(ui: TUI) -> None:
    """Once the dialog's UI has stopped: end the terminal, putting out what it
    still holds; stdout/stderr writes go straight to the fds again."""
    detach_terminal()
    await ui.close()


def _resolve_startup_theme(theme_setting: str | None) -> str:
    theme_name = resolve_theme_setting(theme_setting, get_terminal_theme())
    return theme_name if theme_name is not None else SYSTEM_THEME_NAME


async def start_startup_tui(ui: TUI, settings_manager: SettingsManager) -> None:
    await ui.start()
    theme_setting = settings_manager.get_theme_setting()

    def on_colors():
        return set_theme(_resolve_startup_theme(theme_setting))

    _query_startup_terminal_colors(ui, on_colors)


def _query_startup_terminal_colors(ui: TUI, on_colors) -> None:
    """Query the terminal's colors without waiting for them. When they
    arrive, including after the timeout, record them for the system theme and
    "" (terminal default) tokens, run ``on_colors``, and re-render. The TUI's
    terminal-event loop delivers them, one report at a time."""

    async def apply(colors: dict) -> None:
        set_terminal_colors(colors)
        await on_colors()
        ui.invalidate()
        ui.request_render()

    ui.on_terminal_colors(apply)
    request_terminal_colors(ui)


def _show_startup_dialog(ui: TUI, component, blocked: BlockedStatus, settings_manager: SettingsManager):
    """Show a startup dialog and report pidrei as blocked on it (OSC 7501). Interactive mode is not
    running yet, so its status reporter cannot do this; stopping the startup TUI clears the report.
    The terminal holds the status until it has confirmed support. Returns the start, to await."""
    ui.add_child(component)
    ui.set_focus(component)
    ui.terminal.set_program_status(
        ProgramStatus(state="blocked", app=APP_NAME, kind=blocked.kind, message=blocked.message)
    )
    return start_startup_tui(ui, settings_manager)


async def _clear_startup_tui(ui: TUI) -> None:
    ui.clear()
    ui.request_render()
    await tonio.time.sleep(0.025)


async def show_startup_selector(
    settings_manager: SettingsManager, title: str, options: list, blocked_kind: str = "question"
):
    """Show a selector over ``{"label", "value"}`` options; None on cancel."""
    ui = await create_startup_tui(settings_manager)
    done = tonio.Event()
    outcome: dict = {"value": None}
    settled = False

    def settle(result) -> bool:
        """pi's `void finish(result)` runs `finish` up to its first await in the key handler: the
        "answered once" claim. The callbacks run in the input consumer one at a time (the dialog has
        no countdown), so the claim needs no lock."""
        nonlocal settled
        if settled:
            return False
        settled = True
        outcome["value"] = result
        return True

    async def finish() -> None:
        await _clear_startup_tui(ui)
        await ui.stop()
        await close_startup_tui(ui)
        done.set()

    def on_select(option: str) -> None:
        if settle(next((entry["value"] for entry in options if entry["label"] == option), None)):
            tonio.spawn.without_tracking(finish())

    def on_cancel() -> None:
        if settle(None):
            tonio.spawn.without_tracking(finish())

    selector = ExtensionSelectorComponent(
        title,
        [option["label"] for option in options],
        on_select,
        on_cancel,
        {"tui": ui},
    )
    await _show_startup_dialog(ui, selector, dialog_blocked_status(blocked_kind, title), settings_manager)
    await done.wait(None)
    return outcome["value"]


async def show_startup_input(
    settings_manager: SettingsManager, title: str, placeholder: str | None = None, blocked_kind: str = "question"
):
    ui = await create_startup_tui(settings_manager)
    done = tonio.Event()
    outcome: dict = {"value": None}
    settled = False

    def settle(result) -> bool:
        """The prefix of pi's `void finish(result)`, run in the key handler: see
        `show_startup_selector`."""
        nonlocal settled
        if settled:
            return False
        settled = True
        outcome["value"] = result
        input_component.dispose()
        return True

    async def finish() -> None:
        await _clear_startup_tui(ui)
        await ui.stop()
        await close_startup_tui(ui)
        done.set()

    def on_submit(value: str) -> None:
        if settle(value):
            tonio.spawn.without_tracking(finish())

    def on_cancel() -> None:
        if settle(None):
            tonio.spawn.without_tracking(finish())

    input_component = ExtensionInputComponent(
        title,
        placeholder,
        on_submit,
        on_cancel,
        {"tui": ui},
    )
    await _show_startup_dialog(ui, input_component, dialog_blocked_status(blocked_kind, title), settings_manager)
    await done.wait(None)
    return outcome["value"]
