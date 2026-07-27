"""Mirror of pi coding-agent src/cli/startup-ui.ts."""

import contextlib
import os

import tonio.colored as tonio

from pidrei_tui import TUI, ProcessTerminal, get_capabilities, set_keybindings

from ..config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, PACKAGE_NAME, get_agent_dir, get_settings_path
from ..core.experimental import are_experimental_features_enabled
from ..core.keybindings import KeybindingsManager
from ..core.package_manager import DefaultPackageManager
from ..core.settings_manager import SettingsManager
from ..modes.interactive.components.extension_input import ExtensionInputComponent
from ..modes.interactive.components.extension_selector import ExtensionSelectorComponent
from ..modes.interactive.components.first_time_setup import FirstTimeSetupComponent
from ..modes.interactive.theme import (
    _load_theme_from_path_sync,
    detect_terminal_background_from_env,
    detect_terminal_theme_for_auto,
    init_theme,
    parse_auto_theme_setting,
    prime_theme_cache,
    resolve_theme_setting,
    set_registered_themes,
    set_theme,
)


_OFFICIAL_PACKAGE_NAME = "pidrei"
_OFFICIAL_APP_NAME = "pidrei"
_OFFICIAL_CONFIG_DIR_NAME = ".pidrei"


def _is_official_distribution(package_name: str, app_name: str, config_dir_name: str) -> bool:
    return (
        package_name == _OFFICIAL_PACKAGE_NAME
        and app_name == _OFFICIAL_APP_NAME
        and config_dir_name == _OFFICIAL_CONFIG_DIR_NAME
    )


def _load_themes(resources: list) -> list:
    themes: list = []
    seen: set = set()
    for resource in resources:
        if not resource.enabled:
            continue
        # Startup prompts should not fail because a theme is broken. The
        # normal resource loader reports theme diagnostics later in startup.
        with contextlib.suppress(Exception):
            loaded_theme = _load_theme_from_path_sync(resource.path)
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
    return await tonio.spawn_blocking(_load_themes, resolved_paths.themes)


async def create_startup_tui(settings_manager: SettingsManager) -> TUI:
    # Warm the caches that sync render/callback paths read from, so neither
    # the builtin-theme files nor the tmux capability probe is ever touched
    # from a runtime worker later.
    await prime_theme_cache()
    await tonio.spawn_blocking(get_capabilities)
    set_registered_themes(await _load_startup_themes(settings_manager))
    terminal_theme = detect_terminal_background_from_env()["theme"]
    await init_theme(resolve_theme_setting(settings_manager.get_theme_setting(), terminal_theme) or terminal_theme)
    set_keybindings(await KeybindingsManager.create())
    ui = TUI(ProcessTerminal(), settings_manager.get_show_hardware_cursor(), get_agent_dir())
    ui.set_clear_on_shrink(settings_manager.get_clear_on_shrink())
    return ui


async def start_startup_tui(ui: TUI, settings_manager: SettingsManager) -> None:
    await ui.start()
    tonio.spawn.without_tracking(_apply_detected_startup_theme(ui, settings_manager))


async def _apply_detected_startup_theme(ui: TUI, settings_manager: SettingsManager) -> None:
    theme_setting = settings_manager.get_theme_setting()
    if theme_setting and not parse_auto_theme_setting(theme_setting):
        return

    terminal_theme = await detect_terminal_theme_for_auto({"ui": ui, "timeoutMs": 100})
    await set_theme(resolve_theme_setting(theme_setting, terminal_theme) or terminal_theme)
    ui.invalidate()
    ui.request_render()


async def _clear_startup_tui(ui: TUI) -> None:
    ui.clear()
    ui.request_render()
    await tonio.time.sleep(0.025)


def should_run_first_time_setup(settings_path: str | None = None) -> bool:
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
    return not os.path.exists(settings_path)


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
            settings_manager.set_enable_analytics(result["shareAnalytics"])
            await settings_manager.flush()
        await _clear_startup_tui(ui)
        await ui.stop()
        done.set()

    await ui.start()
    detected_theme = await detect_terminal_theme_for_auto({"ui": ui, "timeoutMs": 100})
    await set_theme(detected_theme)

    async def on_theme_preview(theme_name: str) -> None:
        await set_theme(theme_name)
        ui.request_render()

    component = FirstTimeSetupComponent(
        {
            "detectedTheme": detected_theme,
            "onThemePreview": on_theme_preview,
            "onSubmit": lambda result: tonio.spawn.without_tracking(finish(result)),
            "onCancel": lambda: tonio.spawn.without_tracking(finish(None)),
        }
    )
    ui.add_child(component)
    ui.set_focus(component)
    ui.request_render()
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
