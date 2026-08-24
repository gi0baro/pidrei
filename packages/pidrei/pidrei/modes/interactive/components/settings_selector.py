"""Mirror of pi coding-agent src/modes/interactive/components/settings-selector.ts.

``config``/``callbacks`` are camelCase dict records mirroring pi's
SettingsConfig/SettingsCallbacks interfaces.
"""

from pidrei_tui import Container, SelectList, SettingsList, Spacer, Text, get_capabilities

from ....core.http_config import HTTP_IDLE_TIMEOUT_CHOICES, format_http_idle_timeout_ms
from ..theme import get_select_list_theme, get_settings_list_theme, parse_auto_theme_setting, theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_display_text


SETTINGS_SUBMENU_SELECT_LIST_LAYOUT = {"minPrimaryColumnWidth": 12, "maxPrimaryColumnWidth": 32}

THINKING_DESCRIPTIONS = {
    "off": "No reasoning",
    "minimal": "Very brief reasoning (~1k tokens)",
    "low": "Light reasoning (~2k tokens)",
    "medium": "Moderate reasoning (~8k tokens)",
    "high": "Deep reasoning (~16k tokens)",
    "xhigh": "Extra-high reasoning (~32k tokens)",
    "max": "Maximum reasoning",
}

DEFAULT_PROJECT_TRUST_LABELS = {
    "ask": "Ask",
    "always": "Always trust",
    "never": "Never trust",
}

DEFAULT_PROJECT_TRUST_BY_LABEL = {label: value for value, label in DEFAULT_PROJECT_TRUST_LABELS.items()}


class WarningSettingsSubmenu(Container):
    """A submenu component for the individual warning toggles."""

    def __init__(self, warnings: dict, on_change, on_cancel) -> None:
        super().__init__()

        self._state = dict(warnings)

        anthropic_extra_usage = self._state.get("anthropicExtraUsage")
        items = [
            {
                "id": "anthropic-extra-usage",
                "label": "Anthropic extra usage",
                "description": "Warn when Anthropic subscription auth may use paid extra usage",
                "currentValue": "true"
                if (anthropic_extra_usage if anthropic_extra_usage is not None else True)
                else "false",
                "values": ["true", "false"],
            },
        ]

        async def handle_change(item_id: str, new_value: str) -> None:
            if item_id == "anthropic-extra-usage":
                self._state = {**self._state, "anthropicExtraUsage": new_value == "true"}
                on_change(dict(self._state))

        self._settings_list = SettingsList(
            items,
            min(len(items), 10),
            get_settings_list_theme(),
            handle_change,
            on_cancel,
        )

        self.add_child(self._settings_list)

    async def handle_input(self, data: str) -> None:
        await self._settings_list.handle_input(data)


class SelectSubmenu(Container):
    def __init__(
        self,
        title: str,
        description: str,
        options: list,
        current_value: str,
        on_select,
        on_cancel,
        on_selection_change=None,
    ) -> None:
        super().__init__()

        # Title
        self.add_child(Text(theme.bold(theme.fg("accent", title)), 0, 0))

        # Description
        if description:
            self.add_child(Spacer(1))
            self.add_child(Text(theme.fg("muted", description), 0, 0))

        # Spacer
        self.add_child(Spacer(1))

        # Select list
        self._select_list = SelectList(
            options,
            min(len(options), 10),
            get_select_list_theme(),
            SETTINGS_SUBMENU_SELECT_LIST_LAYOUT,
        )

        # Pre-select current value
        current_index = next((i for i, o in enumerate(options) if o["value"] == current_value), -1)
        if current_index != -1:
            self._select_list.set_selected_index(current_index)

        self._select_list.on_select = lambda item: on_select(item["value"])
        self._select_list.on_cancel = on_cancel

        if on_selection_change is not None:
            self._select_list.on_selection_change = lambda item: on_selection_change(item["value"])

        self.add_child(self._select_list)

        # Hint
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("dim", "  Enter to select · Esc to go back"), 0, 0))

    async def handle_input(self, data: str) -> None:
        await self._select_list.handle_input(data)


def _theme_items(available_themes: list) -> list:
    return [{"value": name, "label": name} for name in available_themes]


AUTOMATIC_THEME_VALUE = "/"


def _single_mode_theme_items(available_themes: list) -> list:
    return [
        {
            "value": AUTOMATIC_THEME_VALUE,
            "label": "Automatic",
            "description": "Use separate themes for light and dark terminal appearance",
        },
        *_theme_items(available_themes),
    ]


def _preferred_theme(available_themes: list, preferred: str | None, fallback: str) -> str:
    if preferred and preferred in available_themes:
        return preferred
    if fallback in available_themes:
        return fallback
    return available_themes[0] if available_themes else fallback


def _default_automatic_themes(current_theme_setting: str, available_themes: list) -> dict:
    auto_theme = parse_auto_theme_setting(current_theme_setting)
    if auto_theme:
        return auto_theme

    current_fixed_theme = None if "/" in current_theme_setting else current_theme_setting
    theme_name = _preferred_theme(available_themes, current_fixed_theme, "dark")
    return {"lightTheme": theme_name, "darkTheme": theme_name}


class ThemeSubmenu(Container):
    def __init__(
        self,
        current_theme_setting: str,
        terminal_theme: str,
        available_themes: list,
        callbacks: dict,
        on_done,
    ) -> None:
        super().__init__()
        self._callbacks = callbacks
        self._available_themes = available_themes
        self._terminal_theme = terminal_theme
        self._on_done = on_done
        self._original_theme_setting = current_theme_setting
        self._input_component = None
        auto_theme = parse_auto_theme_setting(current_theme_setting)
        automatic_themes = _default_automatic_themes(current_theme_setting, available_themes)
        fixed_theme = None if (auto_theme or "/" in current_theme_setting) else current_theme_setting
        self._mode = "automatic" if auto_theme else "single"
        self._light_theme = automatic_themes["lightTheme"]
        self._dark_theme = automatic_themes["darkTheme"]
        self._single_theme = _preferred_theme(
            available_themes,
            fixed_theme if fixed_theme is not None else (self._get_active_automatic_theme() if auto_theme else None),
            "dark",
        )

        if self._mode == "automatic":
            self._show_automatic_menu()
        else:
            self._show_single_menu()

    async def handle_input(self, data: str) -> None:
        if self._input_component is not None:
            handle = getattr(self._input_component, "handle_input", None)
            if handle is not None:
                await handle(data)

    async def _preview(self, value: str) -> None:
        on_theme_preview = self._callbacks.get("onThemePreview")
        if on_theme_preview is not None:
            # Applying a preview loads the theme from disk; the callback is
            # coroutine-returning by contract (never-block rule).
            await on_theme_preview(value)

    def _set_content(self, render_component, input_component=None) -> None:
        self.clear()
        self.add_child(render_component)
        self._input_component = input_component if input_component is not None else render_component

    def _show_single_menu(self) -> None:
        self._mode = "single"

        async def on_select(value: str) -> None:
            if value == AUTOMATIC_THEME_VALUE:
                self._mode = "automatic"
                await self._preview(self._get_theme_setting())
                self._show_automatic_menu()
                return

            self._single_theme = value
            await self._apply(value)

        async def on_selection_change(value: str) -> None:
            await self._preview(self._get_automatic_theme_setting() if value == AUTOMATIC_THEME_VALUE else value)

        menu = SelectSubmenu(
            "Theme",
            "Select a theme, or choose Automatic to follow terminal appearance.",
            _single_mode_theme_items(self._available_themes),
            self._single_theme,
            on_select,
            lambda: self._cancel(),
            on_selection_change,
        )
        self._set_content(menu)

    def _show_automatic_menu(self) -> None:
        self._mode = "automatic"
        content = Container()
        content.add_child(Text(theme.bold(theme.fg("accent", "Automatic Theme")), 0, 0))
        content.add_child(Spacer(1))
        content.add_child(Text(theme.fg("muted", "Choose themes for terminal light and dark appearance."), 0, 0))
        content.add_child(Text(theme.fg("muted", "Light/dark detection requires terminal support."), 0, 0))
        content.add_child(Spacer(1))

        def light_submenu(current_value, done):
            async def on_select(value: str) -> None:
                self._light_theme = value
                await self._preview(self._get_theme_setting())
                await done(value)

            return self._create_theme_select(
                "Light Theme",
                "Select the theme to use for light terminal appearance",
                current_value,
                done,
                on_select,
            )

        def dark_submenu(current_value, done):
            async def on_select(value: str) -> None:
                self._dark_theme = value
                await self._preview(self._get_theme_setting())
                await done(value)

            return self._create_theme_select(
                "Dark Theme",
                "Select the theme to use for dark terminal appearance",
                current_value,
                done,
                on_select,
            )

        items = [
            {
                "id": "light-theme",
                "label": "Light theme",
                "description": "Theme to use in automatic mode when the terminal is light",
                "currentValue": self._light_theme,
                "submenu": light_submenu,
            },
            {
                "id": "dark-theme",
                "label": "Dark theme",
                "description": "Theme to use in automatic mode when the terminal is dark",
                "currentValue": self._dark_theme,
                "submenu": dark_submenu,
            },
            {
                "id": "apply",
                "label": "Apply",
                "description": "Save and go back",
                "currentValue": "save and go back",
                "values": ["save and go back"],
            },
            {
                "id": "single-mode",
                "label": "Change mode",
                "description": "Switch to one theme for light and dark",
                "currentValue": "switch to single theme",
                "values": ["switch to single theme"],
            },
        ]

        async def handle_change(item_id: str, _new_value: str) -> None:
            if item_id == "single-mode":
                self._mode = "single"
                self._single_theme = self._get_active_automatic_theme()
                await self._preview(self._single_theme)
                self._show_single_menu()
            elif item_id == "apply":
                await self._apply(self._get_automatic_theme_setting())

        settings_list = SettingsList(
            items,
            min(len(items), 10),
            get_settings_list_theme(),
            handle_change,
            lambda: self._cancel(),
        )
        content.add_child(settings_list)
        self._set_content(content, settings_list)

    def _create_theme_select(self, title: str, description: str, current_value: str, done, on_select) -> SelectSubmenu:
        async def on_cancel() -> None:
            await self._preview(self._get_theme_setting())
            await done()

        return SelectSubmenu(
            title,
            description,
            _theme_items(self._available_themes),
            current_value,
            on_select,
            on_cancel,
            lambda value: self._preview(value),
        )

    def _get_theme_setting(self) -> str:
        return self._get_automatic_theme_setting() if self._mode == "automatic" else self._single_theme

    def _get_active_automatic_theme(self) -> str:
        return self._light_theme if self._terminal_theme == "light" else self._dark_theme

    def _get_automatic_theme_setting(self) -> str:
        return f"{self._light_theme}/{self._dark_theme}"

    async def _apply(self, theme_setting: str) -> None:
        await self._on_done(theme_setting)

    async def _cancel(self) -> None:
        await self._preview(self._original_theme_setting)
        await self._on_done()


class SettingsSelectorComponent(Container):
    """Main settings selector component."""

    def __init__(self, config: dict, callbacks: dict) -> None:
        super().__init__()

        supports_images = get_capabilities()["images"]
        follow_up_key = key_display_text("app.message.followUp")
        current_warnings = dict(config["warnings"])

        def warnings_submenu(_current_value, done):
            def on_change(warnings: dict) -> None:
                nonlocal current_warnings
                current_warnings = warnings
                callbacks["onWarningsChange"](warnings)

            return WarningSettingsSubmenu(current_warnings, on_change, lambda: done())

        def thinking_submenu(current_value, done):
            async def on_select(value: str) -> None:
                await callbacks["onThinkingLevelChange"](value)
                await done(value)

            return SelectSubmenu(
                "Thinking Level",
                "Select reasoning depth for thinking-capable models",
                [
                    {"value": level, "label": level, "description": THINKING_DESCRIPTIONS[level]}
                    for level in config["availableThinkingLevels"]
                ],
                current_value,
                on_select,
                lambda: done(),
            )

        def theme_submenu(current_value, done):
            return ThemeSubmenu(current_value, config["terminalTheme"], config["availableThemes"], callbacks, done)

        items = [
            {
                "id": "autocompact",
                "label": "Auto-compact",
                "description": "Automatically compact context when it gets too large",
                "currentValue": "true" if config["autoCompact"] else "false",
                "values": ["true", "false"],
            },
            {
                "id": "steering-mode",
                "label": "Steering mode",
                "description": (
                    "Enter while streaming queues steering messages. 'one-at-a-time': deliver one, "
                    "wait for response. 'all': deliver all at once."
                ),
                "currentValue": config["steeringMode"],
                "values": ["one-at-a-time", "all"],
            },
            {
                "id": "follow-up-mode",
                "label": "Follow-up mode",
                "description": (
                    f"{follow_up_key} queues follow-up messages until agent stops. 'one-at-a-time': "
                    "deliver one, wait for response. 'all': deliver all at once."
                ),
                "currentValue": config["followUpMode"],
                "values": ["one-at-a-time", "all"],
            },
            {
                "id": "transport",
                "label": "Transport",
                "description": "Preferred transport for providers that support multiple transports",
                "currentValue": config["transport"],
                "values": ["sse", "websocket", "websocket-cached", "auto"],
            },
            {
                "id": "http-idle-timeout",
                "label": "HTTP idle timeout",
                "description": (
                    "Maximum idle gap while waiting for HTTP headers or body chunks. Disable for "
                    "local models that pause longer than five minutes."
                ),
                "currentValue": format_http_idle_timeout_ms(config["httpIdleTimeoutMs"]),
                "values": [choice["label"] for choice in HTTP_IDLE_TIMEOUT_CHOICES],
            },
            {
                "id": "hide-thinking",
                "label": "Hide thinking",
                "description": "Hide thinking blocks in assistant responses",
                "currentValue": "true" if config["hideThinkingBlock"] else "false",
                "values": ["true", "false"],
            },
            {
                "id": "cache-miss-notices",
                "label": "Cache miss notices",
                "description": "Show transcript notices for significant prompt-cache misses and compaction costs",
                "currentValue": "true" if config["showCacheMissNotices"] else "false",
                "values": ["true", "false"],
            },
            {
                "id": "collapse-changelog",
                "label": "Collapse changelog",
                "description": "Show condensed changelog after updates",
                "currentValue": "true" if config["collapseChangelog"] else "false",
                "values": ["true", "false"],
            },
            {
                "id": "quiet-startup",
                "label": "Quiet startup",
                "description": "Disable verbose printing at startup",
                "currentValue": "true" if config["quietStartup"] else "false",
                "values": ["true", "false"],
            },
            {
                "id": "provider-attribution",
                "label": "Provider attribution",
                "description": "Identify pidrei to providers that credit the calling app (OpenRouter, NVIDIA, opencode)",
                "currentValue": "true" if config["enableProviderAttribution"] else "false",
                "values": ["true", "false"],
            },
            {
                "id": "default-project-trust",
                "label": "Default project trust",
                "description": "Fallback behavior when no extension or saved trust decision decides project trust",
                "currentValue": DEFAULT_PROJECT_TRUST_LABELS[config["defaultProjectTrust"]],
                "values": list(DEFAULT_PROJECT_TRUST_LABELS.values()),
            },
            {
                "id": "double-escape-action",
                "label": "Double-escape action",
                "description": "Action when pressing Escape twice with empty editor",
                "currentValue": config["doubleEscapeAction"],
                "values": ["tree", "fork", "none"],
            },
            {
                "id": "tree-filter-mode",
                "label": "Tree filter mode",
                "description": "Default filter when opening /tree",
                "currentValue": config["treeFilterMode"],
                "values": ["default", "no-tools", "user-only", "labeled-only", "all"],
            },
            {
                "id": "warnings",
                "label": "Warnings",
                "description": "Enable or disable individual warnings",
                "currentValue": "configure",
                "submenu": warnings_submenu,
            },
            {
                "id": "thinking",
                "label": "Thinking level",
                "description": "Reasoning depth for thinking-capable models",
                "currentValue": config["thinkingLevel"],
                "submenu": thinking_submenu,
            },
            {
                "id": "tui-mode",
                "label": "TUI mode",
                "description": "Interface layout; fullscreen mode is experimental",
                "currentValue": config["tuiMode"],
                "values": ["regular", "fullscreen"],
            },
            {
                "id": "fullscreen-exit-output",
                "label": "Fullscreen exit output",
                "description": "Print the transcript or only a session resume hint when exiting fullscreen mode",
                "currentValue": config["fullscreenExitOutput"],
                "values": ["transcript", "resume-hint"],
            },
            {
                "id": "fullscreen-scrollbar",
                "label": "Fullscreen scrollbar",
                "description": "Scrollbar behavior in fullscreen mode; has no effect in regular mode",
                "currentValue": config["fullscreenScrollbar"],
                "values": ["auto", "always", "hidden"],
            },
            {
                "id": "theme",
                "label": "Theme",
                "description": "Color theme for the interface",
                "currentValue": config["currentTheme"],
                "submenu": theme_submenu,
            },
        ]

        # Only show image toggle if terminal supports it
        if supports_images:
            # Insert after autocompact
            items.insert(
                1,
                {
                    "id": "show-images",
                    "label": "Show images",
                    "description": "Render images inline in terminal",
                    "currentValue": "true" if config["showImages"] else "false",
                    "values": ["true", "false"],
                },
            )
            items.insert(
                2,
                {
                    "id": "image-width-cells",
                    "label": "Image width",
                    "description": "Preferred inline image width in terminal cells",
                    "currentValue": str(config["imageWidthCells"]),
                    "values": ["60", "80", "120"],
                },
            )

        # Image auto-resize toggle (always available, affects both attached
        # and read images)
        items.insert(
            3 if supports_images else 1,
            {
                "id": "auto-resize-images",
                "label": "Auto-resize images",
                "description": "Resize large images to 2000x2000 max for better model compatibility",
                "currentValue": "true" if config["autoResizeImages"] else "false",
                "values": ["true", "false"],
            },
        )

        def insert_after(anchor_id: str, item: dict) -> None:
            anchor_index = next(i for i, entry in enumerate(items) if entry["id"] == anchor_id)
            items.insert(anchor_index + 1, item)

        # Block images toggle (always available, insert after auto-resize-images)
        insert_after(
            "auto-resize-images",
            {
                "id": "block-images",
                "label": "Block images",
                "description": "Prevent images from being sent to LLM providers",
                "currentValue": "true" if config["blockImages"] else "false",
                "values": ["true", "false"],
            },
        )

        # Skill commands toggle (insert after block-images)
        insert_after(
            "block-images",
            {
                "id": "skill-commands",
                "label": "Skill commands",
                "description": "Register skills as /skill:name commands",
                "currentValue": "true" if config["enableSkillCommands"] else "false",
                "values": ["true", "false"],
            },
        )

        # Hardware cursor toggle (insert after skill-commands)
        insert_after(
            "skill-commands",
            {
                "id": "show-hardware-cursor",
                "label": "Show hardware cursor",
                "description": "Show the terminal cursor while still positioning it for IME support",
                "currentValue": "true" if config["showHardwareCursor"] else "false",
                "values": ["true", "false"],
            },
        )

        # Editor padding toggle (insert after show-hardware-cursor)
        insert_after(
            "show-hardware-cursor",
            {
                "id": "editor-padding",
                "label": "Editor padding",
                "description": "Horizontal padding for input editor (0-3)",
                "currentValue": str(config["editorPaddingX"]),
                "values": ["0", "1", "2", "3"],
            },
        )

        # Output padding toggle (insert after editor-padding)
        insert_after(
            "editor-padding",
            {
                "id": "output-padding",
                "label": "Output padding",
                "description": "Horizontal padding for user messages, assistant messages, and thinking",
                "currentValue": str(config["outputPad"]),
                "values": ["0", "1"],
            },
        )

        # Autocomplete max visible toggle (insert after output-padding)
        insert_after(
            "output-padding",
            {
                "id": "autocomplete-max-visible",
                "label": "Autocomplete max items",
                "description": "Max visible items in autocomplete dropdown (3-20)",
                "currentValue": str(config["autocompleteMaxVisible"]),
                "values": ["3", "5", "7", "10", "15", "20"],
            },
        )

        # Clear on shrink toggle (insert after autocomplete-max-visible)
        insert_after(
            "autocomplete-max-visible",
            {
                "id": "clear-on-shrink",
                "label": "Clear on shrink",
                "description": "Clear empty rows when content shrinks (may cause flicker)",
                "currentValue": "true" if config["clearOnShrink"] else "false",
                "values": ["true", "false"],
            },
        )

        # Terminal progress toggle (insert after clear-on-shrink)
        insert_after(
            "clear-on-shrink",
            {
                "id": "terminal-progress",
                "label": "Terminal progress",
                "description": "Show OSC 9;4 progress indicators in the terminal tab bar",
                "currentValue": "true" if config["showTerminalProgress"] else "false",
                "values": ["true", "false"],
            },
        )

        # Add borders
        self.add_child(DynamicBorder())

        async def handle_change(item_id: str, new_value: str) -> None:
            if item_id == "autocompact":
                callbacks["onAutoCompactChange"](new_value == "true")
            elif item_id == "show-images":
                callbacks["onShowImagesChange"](new_value == "true")
            elif item_id == "image-width-cells":
                callbacks["onImageWidthCellsChange"](int(new_value))
            elif item_id == "auto-resize-images":
                callbacks["onAutoResizeImagesChange"](new_value == "true")
            elif item_id == "block-images":
                callbacks["onBlockImagesChange"](new_value == "true")
            elif item_id == "skill-commands":
                callbacks["onEnableSkillCommandsChange"](new_value == "true")
            elif item_id == "steering-mode":
                callbacks["onSteeringModeChange"](new_value)
            elif item_id == "follow-up-mode":
                callbacks["onFollowUpModeChange"](new_value)
            elif item_id == "transport":
                callbacks["onTransportChange"](new_value)
            elif item_id == "http-idle-timeout":
                choice = next((item for item in HTTP_IDLE_TIMEOUT_CHOICES if item["label"] == new_value), None)
                if choice is not None:
                    callbacks["onHttpIdleTimeoutMsChange"](choice["timeout_ms"])
            elif item_id == "hide-thinking":
                callbacks["onHideThinkingBlockChange"](new_value == "true")
            elif item_id == "cache-miss-notices":
                callbacks["onShowCacheMissNoticesChange"](new_value == "true")
            elif item_id == "collapse-changelog":
                callbacks["onCollapseChangelogChange"](new_value == "true")
            elif item_id == "quiet-startup":
                callbacks["onQuietStartupChange"](new_value == "true")
            elif item_id == "provider-attribution":
                callbacks["onEnableProviderAttributionChange"](new_value == "true")
            elif item_id == "default-project-trust":
                default_project_trust = DEFAULT_PROJECT_TRUST_BY_LABEL.get(new_value)
                if default_project_trust:
                    callbacks["onDefaultProjectTrustChange"](default_project_trust)
            elif item_id == "double-escape-action":
                callbacks["onDoubleEscapeActionChange"](new_value)
            elif item_id == "tree-filter-mode":
                callbacks["onTreeFilterModeChange"](new_value)
            elif item_id == "show-hardware-cursor":
                callbacks["onShowHardwareCursorChange"](new_value == "true")
            elif item_id == "editor-padding":
                callbacks["onEditorPaddingXChange"](int(new_value))
            elif item_id == "output-padding":
                callbacks["onOutputPadChange"](0 if new_value == "0" else 1)
            elif item_id == "autocomplete-max-visible":
                callbacks["onAutocompleteMaxVisibleChange"](int(new_value))
            elif item_id == "clear-on-shrink":
                callbacks["onClearOnShrinkChange"](new_value == "true")
            elif item_id == "terminal-progress":
                callbacks["onShowTerminalProgressChange"](new_value == "true")
            elif item_id == "tui-mode":
                # Awaited, unlike its siblings: switching renderers is async here
                # (pi's stop/start are sync) and the status line it writes must
                # land after the swap, not race it.
                await callbacks["onTuiModeChange"](new_value)
            elif item_id == "fullscreen-exit-output":
                callbacks["onFullscreenExitOutputChange"](new_value)
            elif item_id == "fullscreen-scrollbar":
                callbacks["onFullscreenScrollbarChange"](new_value)
            elif item_id == "theme":
                callbacks["onThemeChange"](new_value)

        self._settings_list = SettingsList(
            items,
            10,
            get_settings_list_theme(),
            handle_change,
            callbacks["onCancel"],
            {"enableSearch": True},
        )

        self.add_child(self._settings_list)
        self.add_child(DynamicBorder())

    def get_settings_list(self) -> SettingsList:
        return self._settings_list
