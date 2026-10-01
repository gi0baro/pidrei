"""Mirror of pi coding-agent src/modes/interactive/theme/theme-controller.ts.

Deviations (the `terminal-colors-loop` recipe):
- Terminal color reports reach the controller as the TUI's
  ``on_terminal_colors`` listener, which the TUI's one terminal-event consumer
  runs in arrival order, like the color-scheme listener. pi's query callback
  and ``onLateReply`` become that listener; ``request_terminal_colors`` only
  starts a query and returns its applied event, which
  ``wait_for_terminal_colors`` waits on.
- pi's methods are synchronous, so a terminal report never lands in the
  middle of a user's theme change. Here the theme loads are awaits, so every
  theme application (settings, selections, previews, terminal reports) runs
  under ``_apply_lock``: each one resolves and applies as a unit, as in pi.
"""

from collections.abc import Awaitable

import tonio.colored as tonio
from tonio.colored import sync

from .system_theme import SYSTEM_THEME_NAME
from .theme import (
    get_terminal_theme,
    init_theme,
    mark_terminal_colors_pending,
    parse_auto_theme_setting,
    resolve_theme_setting,
    set_terminal_color_scheme,
    set_terminal_colors,
    set_theme,
    set_theme_instance as apply_theme_instance,
)


# How long the system theme stays grayscale before falling back to palette
# indices. Terminals answer the trailing DA1 request right after the color
# replies, so this only matters for terminals that answer neither. Replies
# arriving later still apply.
TERMINAL_QUERY_TIMEOUT_MS = 100


def request_terminal_colors(ui) -> tonio.Event:
    """Query the terminal's colors. The reports reach ``ui``'s
    ``on_terminal_colors`` listeners when the query completes or times out,
    and again if the terminal answers after the timeout; a failed query
    reports no colors. Returns an event set after the first report was
    handled."""
    return ui.query_terminal_colors(timeout_ms=TERMINAL_QUERY_TIMEOUT_MS)


def _same_rgb(a: dict | None, b: dict | None) -> bool:
    return a is b or (a is not None and b is not None and a["r"] == b["r"] and a["g"] == b["g"] and a["b"] == b["b"])


def _same_terminal_colors(a: dict, b: dict) -> bool:
    if not _same_rgb(a.get("foreground"), b.get("foreground")) or not _same_rgb(
        a.get("background"), b.get("background")
    ):
        return False
    a_palette = a.get("palette")
    b_palette = b.get("palette")
    if a_palette is b_palette:
        return True
    if not a_palette or not b_palette or len(a_palette) != len(b_palette):
        return False
    return all(_same_rgb(color, b_palette[index]) for index, color in enumerate(a_palette))


def _settled_event() -> tonio.Event:
    event = tonio.Event()
    event.set()
    return event


class InteractiveThemeController:
    """Applies the theme setting and keeps it in sync with the terminal. The
    theme applies immediately, and the terminal's colors update it when they
    arrive; the system theme renders in grayscale until then. Callers that
    bake theme colors into content can wait for the colors with
    `wait_for_terminal_colors()`."""

    def __init__(self, ui, options: dict) -> None:
        """``options``: ``getSettingsManager``, ``showError``, ``onChanged``,
        optional ``initialThemeSetting`` (pi's constructor options record)."""
        self._ui = ui
        self._get_settings_manager = options["getSettingsManager"]
        self._show_error = options["showError"]
        self._on_changed = options["onChanged"]
        self._current_theme_setting = options.get("initialThemeSetting")
        # Last reported colors; a query that times out keeps them instead of
        # erasing them. Only the terminal-colors listener touches it.
        self._terminal_colors: dict | None = None
        self._auto_sync_enabled = False
        self._terminal_color_scheme_unsubscribe = None
        self._terminal_colors_unsubscribe = None
        self._apply_lock = sync.Lock()
        # Set when the latest color query completed or timed out, and its
        # colors applied.
        self._terminal_color_query = _settled_event()
        self._active_theme_name = self._resolve_theme_name()
        # The system theme starts in grayscale; color follows once the
        # terminal reports its colors.
        mark_terminal_colors_pending()
        self._bind_terminal_listeners()

    def prime(self) -> Awaitable[None]:
        """Load the initial theme off the runtime.

        `init_theme` reads theme files, so it cannot run in `__init__`;
        `InteractiveMode.run` calls this before the first render.
        """
        return init_theme(self._active_theme_name, True)

    async def rebind_tui(self) -> None:
        """Re-attach to the renderer InteractiveMode just swapped in."""
        self._unbind_terminal_listeners()
        self._bind_terminal_listeners()
        await self._ui.set_terminal_color_scheme_notifications(self._auto_sync_enabled)

    async def apply_from_settings(self) -> None:
        """Apply the theme setting now and query the terminal's colors, which
        update the theme when they arrive. Theme pairs and the system theme
        follow terminal appearance changes."""
        async with self._apply_lock:
            await self._apply_from_settings()

    async def _apply_from_settings(self) -> None:
        theme_setting = self._get_theme_setting()
        theme_name = self._resolve_theme_name()
        await self._set_auto_sync(
            parse_auto_theme_setting(theme_setting) is not None or theme_name == SYSTEM_THEME_NAME
        )
        await self._apply_theme_name(theme_name, theme_setting is not None)
        self._query_terminal_colors()

    def wait_for_terminal_colors(self) -> Awaitable[None]:
        """Wait until the latest color query completed or timed out. Content
        that bakes theme colors into strings, such as the startup header,
        should be built after this. Terminals answer the DA1 request right
        after the color replies, so this only takes the full timeout when a
        terminal answers nothing."""
        return self._terminal_color_query.wait(None)

    def get_theme_selection(self) -> str | None:
        if self._current_theme_setting is not None:
            return self._current_theme_setting
        settings_theme = self._get_settings_manager().get_theme_setting()
        return settings_theme if settings_theme is not None else self._active_theme_name

    async def set_theme_name(self, theme_name: str, show_error: bool = False) -> dict:
        async with self._apply_lock:
            await self._set_auto_sync(theme_name == SYSTEM_THEME_NAME)
            result = await self._apply_theme_name(theme_name, show_error)
            if result["success"]:
                self._current_theme_setting = theme_name
            return result

    async def set_theme_setting(self, theme_setting: str) -> None:
        async with self._apply_lock:
            self._current_theme_setting = theme_setting
            await self._apply_from_settings()

    async def set_theme_instance(self, theme_instance) -> dict:
        async with self._apply_lock:
            await self._set_auto_sync(False)
            apply_theme_instance(theme_instance)
            self._active_theme_name = "<in-memory>"
            self._notify_changed()
            return {"success": True}

    async def preview(self, theme_setting_or_name: str) -> None:
        async with self._apply_lock:
            theme_name = resolve_theme_setting(theme_setting_or_name, get_terminal_theme())
            if theme_name is None:
                theme_name = self._active_theme_name
            if not theme_name:
                return
            if (await set_theme(theme_name, True))["success"]:
                with self._ui.state_lock:
                    self._ui.invalidate()
                    self._ui.request_render()

    def disable_auto_sync(self) -> Awaitable[None]:
        return self._set_auto_sync(False)

    async def dispose(self) -> None:
        await self._set_auto_sync(False)
        self._unbind_terminal_listeners()

    def get_terminal_theme(self) -> str:
        return get_terminal_theme()

    def _get_theme_setting(self) -> str | None:
        if self._current_theme_setting is not None:
            return self._current_theme_setting
        return self._get_settings_manager().get_theme_setting()

    def _resolve_theme_name(self) -> str:
        """The theme for the current setting and terminal appearance. Without
        a setting, pidrei uses the system theme."""
        theme_name = resolve_theme_setting(self._get_theme_setting(), get_terminal_theme())
        return theme_name if theme_name is not None else SYSTEM_THEME_NAME

    async def _apply_theme_name(self, theme_name: str, show_error: bool = False) -> dict:
        result = await set_theme(theme_name, True)
        self._active_theme_name = theme_name if result["success"] else SYSTEM_THEME_NAME
        self._notify_changed()
        if not result["success"] and show_error:
            self._show_error(
                f'Failed to load theme "{theme_name}": {result.get("error")}\nFell back to the system theme.'
            )
        return result

    def _query_terminal_colors(self) -> None:
        """Query the terminal's colors without waiting for them;
        `wait_for_terminal_colors()` waits for this query."""
        self._terminal_color_query = request_terminal_colors(self._ui)

    async def _apply_terminal_colors(self, reported: dict) -> None:
        """Record reported colors: themes use the default colors for tokens
        set to "", the system theme is generated from all of them, and
        light/dark detection uses them. Re-renders only when they changed."""
        async with self._apply_lock:
            previous = self._terminal_colors or {}

            def merged(key: str):
                value = reported.get(key)
                return value if value is not None else previous.get(key)

            next_colors = {key: merged(key) for key in ("foreground", "background", "palette")}
            # Re-rendering rebuilds every component, so skip it when nothing
            # changed (including timeouts).
            if self._terminal_colors is not None and _same_terminal_colors(self._terminal_colors, next_colors):
                return
            self._terminal_colors = next_colors
            set_terminal_colors(next_colors)
            await self._reapply_for_terminal()
            with self._ui.state_lock:
                self._ui.invalidate()
                self._ui.request_render()

    async def _reapply_for_terminal(self) -> None:
        """Re-apply the setting after the terminal's colors or appearance
        changed: regenerate the system theme, or switch the theme of a pair.
        Themes set through extensions or previews are left alone."""
        if self._active_theme_name == "<in-memory>":
            return
        theme_name = self._resolve_theme_name()
        if theme_name == SYSTEM_THEME_NAME or theme_name != self._active_theme_name:
            await self._apply_theme_name(theme_name)

    async def _set_auto_sync(self, enabled: bool) -> None:
        # Claimed under the lock, before the await: two callers never both
        # act on the same transition.
        with self._ui.state_lock:
            if self._auto_sync_enabled == enabled:
                return
            self._auto_sync_enabled = enabled
        await self._ui.set_terminal_color_scheme_notifications(enabled)

    def _bind_terminal_listeners(self) -> None:
        self._terminal_color_scheme_unsubscribe = self._ui.on_terminal_color_scheme_change(
            self._apply_terminal_color_scheme_change
        )
        self._terminal_colors_unsubscribe = self._ui.on_terminal_colors(self._apply_terminal_colors)

    def _unbind_terminal_listeners(self) -> None:
        if self._terminal_color_scheme_unsubscribe is not None:
            self._terminal_color_scheme_unsubscribe()
        if self._terminal_colors_unsubscribe is not None:
            self._terminal_colors_unsubscribe()
        self._terminal_color_scheme_unsubscribe = self._terminal_colors_unsubscribe = None

    async def _apply_terminal_color_scheme_change(self, terminal_theme: str) -> None:
        """The terminal reported a light/dark switch. Its colors changed too,
        so query them again: they decide the appearance. The reported scheme
        only matters for terminals that do not report their background."""
        async with self._apply_lock:
            if not self._auto_sync_enabled:
                return
            previous = get_terminal_theme()
            set_terminal_color_scheme(terminal_theme)
            if get_terminal_theme() != previous:
                await self._reapply_for_terminal()
            self._query_terminal_colors()

    def _notify_changed(self) -> None:
        with self._ui.state_lock:
            self._ui.invalidate()
            self._on_changed()
