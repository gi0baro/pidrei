"""Mirror of pi coding-agent src/modes/interactive/theme/theme-controller.ts."""

from collections.abc import Awaitable

from .theme import (
    detect_terminal_background_from_env,
    detect_terminal_background_theme,
    detect_terminal_theme_for_auto,
    init_theme,
    parse_auto_theme_setting,
    resolve_theme_setting,
    set_theme,
    set_theme_instance as apply_theme_instance,
)


class InteractiveThemeController:
    def __init__(self, ui, settings_manager, show_error, on_changed) -> None:
        self._ui = ui
        self._settings_manager = settings_manager
        self._show_error = show_error
        self._on_changed = on_changed
        self._terminal_theme = detect_terminal_background_from_env()["theme"]
        self._auto_sync_enabled = False
        self._active_theme_name = resolve_theme_setting(
            self._settings_manager.get_theme_setting(), self._terminal_theme
        )
        self._terminal_color_scheme_unsubscribe = None
        self._bind_terminal_color_scheme_listener()

    async def rebind_tui(self) -> None:
        """Re-attach to the renderer InteractiveMode just swapped in."""
        if self._terminal_color_scheme_unsubscribe is not None:
            self._terminal_color_scheme_unsubscribe()
        self._bind_terminal_color_scheme_listener()
        await self._ui.set_terminal_color_scheme_notifications(self._auto_sync_enabled)

    def _bind_terminal_color_scheme_listener(self) -> None:
        self._terminal_color_scheme_unsubscribe = self._ui.on_terminal_color_scheme_change(self._apply_terminal_theme)

    def prime(self) -> Awaitable[None]:
        """Load the initial theme off the runtime.

        `init_theme` reads theme files, so it cannot run in `__init__`;
        `InteractiveMode.run` calls this before the first render.
        """
        return init_theme(self._active_theme_name, True)

    async def apply_from_settings(self) -> None:
        theme_setting = self._settings_manager.get_theme_setting()
        auto_theme = parse_auto_theme_setting(theme_setting)
        if auto_theme:
            self._terminal_theme = await detect_terminal_theme_for_auto({"ui": self._ui, "timeoutMs": 100})
            await self._set_auto_sync(True)
            await self._apply_theme_name(
                auto_theme["lightTheme"] if self._terminal_theme == "light" else auto_theme["darkTheme"],
                True,
            )
            return

        await self._set_auto_sync(False)
        if theme_setting is not None:
            await self._apply_theme_name(theme_setting, True)
            return

        detection = await detect_terminal_background_theme({"ui": self._ui, "timeoutMs": 100})
        self._terminal_theme = detection["theme"]
        if not (await self._apply_theme_name(detection["theme"]))["success"]:
            return
        if detection["confidence"] == "high":
            self._settings_manager.set_theme(detection["theme"])
            # pidrei's SettingsManager.flush is synchronous (pi awaits it)
            self._settings_manager.flush()

    async def set_theme_name(self, theme_name: str, show_error: bool = False) -> dict:
        await self._set_auto_sync(False)
        return await self._apply_theme_name(theme_name, show_error)

    async def set_theme_instance(self, theme_instance) -> dict:
        await self._set_auto_sync(False)
        apply_theme_instance(theme_instance)
        self._active_theme_name = "<in-memory>"
        self._notify_changed()
        return {"success": True}

    async def preview(self, theme_setting_or_name: str) -> None:
        theme_name = resolve_theme_setting(theme_setting_or_name, self._terminal_theme) or self._active_theme_name
        if not theme_name:
            return
        if (await set_theme(theme_name, True))["success"]:
            self._ui.invalidate()
            self._ui.request_render()

    def disable_auto_sync(self) -> Awaitable[None]:
        return self._set_auto_sync(False)

    def get_terminal_theme(self) -> str:
        return self._terminal_theme

    async def _apply_theme_name(self, theme_name: str, show_error: bool = False) -> dict:
        result = await set_theme(theme_name, True)
        self._active_theme_name = theme_name if result["success"] else "dark"
        self._notify_changed()
        if not result["success"] and show_error:
            self._show_error(f'Failed to load theme "{theme_name}": {result.get("error")}\nFell back to dark theme.')
        return result

    def _notify_changed(self) -> None:
        self._ui.invalidate()
        self._on_changed()

    async def _set_auto_sync(self, enabled: bool) -> None:
        if self._auto_sync_enabled == enabled:
            return
        self._auto_sync_enabled = enabled
        await self._ui.set_terminal_color_scheme_notifications(enabled)

    async def _apply_terminal_theme(self, terminal_theme: str) -> None:
        if not self._auto_sync_enabled:
            return
        self._terminal_theme = terminal_theme
        auto_theme = parse_auto_theme_setting(self._settings_manager.get_theme_setting())
        if not auto_theme:
            await self._set_auto_sync(False)
            return
        theme_name = auto_theme["lightTheme"] if terminal_theme == "light" else auto_theme["darkTheme"]
        if theme_name != self._active_theme_name:
            await self._apply_theme_name(theme_name)
