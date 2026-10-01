"""Mirror of pi coding-agent src/cli/startup-ui.ts."""

import contextlib
import os

import tonio.colored as tonio
from tonio.colored import fs, sync

from pidrei_tui import (
    TUI,
    ProcessTerminal,
    TuiMainScreen,
    prime_capabilities,
    set_capability_overrides,
    set_keybindings,
)

from ..config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, PACKAGE_NAME, get_agent_dir, get_settings_path
from ..core.experimental import are_experimental_features_enabled
from ..core.keybindings import KeybindingsManager
from ..core.output_guard import attach_terminal, detach_terminal
from ..core.package_manager import DefaultPackageManager
from ..core.settings_manager import SettingsManager
from ..modes.interactive.components.extension_input import ExtensionInputComponent
from ..modes.interactive.components.extension_selector import ExtensionSelectorComponent
from ..modes.interactive.components.first_time_setup import FirstTimeSetupComponent
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


_OFFICIAL_PACKAGE_NAME = "pidrei"
_OFFICIAL_APP_NAME = "pidrei"
_OFFICIAL_CONFIG_DIR_NAME = ".pidrei"


def _is_official_distribution(package_name: str, app_name: str, config_dir_name: str) -> bool:
    return (
        package_name == _OFFICIAL_PACKAGE_NAME
        and app_name == _OFFICIAL_APP_NAME
        and config_dir_name == _OFFICIAL_CONFIG_DIR_NAME
    )


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


async def _clear_startup_tui(ui: TUI) -> None:
    ui.clear()
    ui.request_render()
    await tonio.time.sleep(0.025)


async def should_run_first_time_setup(settings_path: str | None = None) -> bool:
    """First-time setup runs when all of these hold:

    - this is the official pidrei distribution (not a fork/rebrand)
    - experimental features are enabled (PIDREI_EXPERIMENTAL=1)
    - the default agent directory is used (no custom agent dir override)
    - setup was not completed before (settings.json does not exist)
    """
    if settings_path is None:
        settings_path = get_settings_path()
    if not _is_official_distribution(PACKAGE_NAME, APP_NAME, CONFIG_DIR_NAME):
        return False
    if not are_experimental_features_enabled():
        return False
    if os.environ.get(ENV_AGENT_DIR):
        return False
    return not await fs.Path(settings_path).exists()


async def show_startup_selector(settings_manager: SettingsManager, title: str, options: list):
    """Show a selector over ``{"label", "value"}`` options; None on cancel."""
    ui = await create_startup_tui(settings_manager)
    done = tonio.Event()
    outcome: dict = {"value": None}
    settled = False

    async def finish(result) -> None:
        nonlocal settled
        if settled:
            return
        settled = True
        outcome["value"] = result
        await _clear_startup_tui(ui)
        await ui.stop()
        await close_startup_tui(ui)
        done.set()

    def on_select(option: str) -> None:
        value = next((entry["value"] for entry in options if entry["label"] == option), None)
        tonio.spawn.without_tracking(finish(value))

    def on_cancel() -> None:
        tonio.spawn.without_tracking(finish(None))

    selector = ExtensionSelectorComponent(
        title,
        [option["label"] for option in options],
        on_select,
        on_cancel,
        {"tui": ui},
    )
    ui.add_child(selector)
    ui.set_focus(selector)
    await start_startup_tui(ui, settings_manager)
    await done.wait(None)
    return outcome["value"]


async def show_first_time_setup(settings_manager: SettingsManager) -> None:
    """Show the first-time setup dialog and persist the result."""
    ui = await create_startup_tui(settings_manager)
    done = tonio.Event()
    settled = False

    async def finish(result) -> None:
        nonlocal settled
        if settled:
            return
        settled = True
        if result:
            settings_manager.set_theme(result["theme"])
            await settings_manager.flush()
        await _clear_startup_tui(ui)
        await ui.stop()
        await close_startup_tui(ui)
        done.set()

    await ui.start()
    preview_theme = SYSTEM_THEME_NAME
    await set_theme(preview_theme)
    # pi's theme changes are synchronous; a preview and the terminal's colors
    # arriving must not interleave their loads.
    theme_lock = sync.Lock()

    async def preview(theme_name: str) -> None:
        nonlocal preview_theme
        async with theme_lock:
            preview_theme = theme_name
            await set_theme(theme_name)
        ui.request_render()

    def on_theme_preview(theme_name: str) -> None:
        # pi's preview is synchronous: the next key waits for the theme.
        ui.finish_before_next_input(tonio.spawn(preview(theme_name)))

    component = FirstTimeSetupComponent(
        {
            "onThemePreview": on_theme_preview,
            "onSubmit": lambda result: tonio.spawn.without_tracking(finish(result)),
            "onCancel": lambda: tonio.spawn.without_tracking(finish(None)),
        }
    )
    ui.add_child(component)
    ui.set_focus(component)
    ui.request_render()

    async def reapply_preview() -> None:
        async with theme_lock:
            await set_theme(preview_theme)

    # The terminal's colors regenerate the system theme; re-rendering rebuilds the dialog with it.
    _query_startup_terminal_colors(ui, reapply_preview)
    await done.wait(None)


async def show_startup_input(settings_manager: SettingsManager, title: str, placeholder: str | None = None):
    ui = await create_startup_tui(settings_manager)
    done = tonio.Event()
    outcome: dict = {"value": None}
    settled = False

    async def finish(result) -> None:
        nonlocal settled
        if settled:
            return
        settled = True
        outcome["value"] = result
        input_component.dispose()
        await _clear_startup_tui(ui)
        await ui.stop()
        await close_startup_tui(ui)
        done.set()

    input_component = ExtensionInputComponent(
        title,
        placeholder,
        lambda value: tonio.spawn.without_tracking(finish(value)),
        lambda: tonio.spawn.without_tracking(finish(None)),
        {"tui": ui},
    )
    ui.add_child(input_component)
    ui.set_focus(input_component)
    await start_startup_tui(ui, settings_manager)
    await done.wait(None)
    return outcome["value"]
