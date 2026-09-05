"""Mirror of pi coding-agent src/modes/interactive/components/custom-editor.ts."""

from pidrei_tui import Editor, visible_width


def sync_action(fn):
    """Adapt a sync callable to the awaitable-returning action-handler contract.

    Handlers registered via `on_action`/`on_escape`/`on_ctrl_d` must return an
    awaitable (async-only callback policy); pi types them `() => void`, but
    input handling is async here and a handler that persists something (the
    thinking-level cycle, for one) must be awaited, not dropped.
    """

    async def handler():
        fn()

    return handler


class CustomEditor(Editor):
    """Editor that handles app-level keybindings for the coding agent.

    ``options`` is pi's ``CustomEditorOptions``: the ``Editor`` options plus
    ``embedWorkingStatus`` — render the streaming working status in the
    editor's top border instead of the standalone status row.
    """

    def __init__(self, tui, theme: dict, keybindings, options: dict | None = None) -> None:
        super().__init__(tui, theme, options)
        self._keybindings = keybindings
        self._working_status_indicator = None
        self.embed_working_status: bool = bool((options or {}).get("embedWorkingStatus", False))
        self.action_handlers: dict = {}

        # Special handlers that can be dynamically replaced
        self.on_escape = None
        self.on_ctrl_d = None
        self.on_paste_image = None
        # Handler for extension-registered shortcuts. Returns True if handled.
        self.on_extension_shortcut = None

    def set_working_status_indicator(self, indicator) -> None:
        self._working_status_indicator = indicator

    def render_top_border(self, width: int, hidden_line_count: int) -> str:
        indicator = self._working_status_indicator
        if not self.embed_working_status or indicator is None or width <= 0:
            return super().render_top_border(width, hidden_line_count)

        status = indicator.render_in_border(max(1, width - 5))
        status_width = visible_width(status)
        if status_width == 0:
            return super().render_top_border(width, hidden_line_count)

        overflow_label = f" ↑ {hidden_line_count} more " if hidden_line_count > 0 else None
        overflow_label_width = visible_width(overflow_label) if overflow_label else 0
        overflow_start = (width - overflow_label_width) // 2

        def can_fit_overflow() -> bool:
            return (
                overflow_label is not None
                and overflow_label_width + 2 <= width
                and overflow_start - (3 + status_width + 1) >= 1
            )

        if overflow_label and not can_fit_overflow():
            status = indicator.render_spinner_in_border(width)
            status_width = visible_width(status)

        if can_fit_overflow():
            left_block_width = 3 + status_width + 1
            return (
                self.border_color("── ")
                + status
                + self.border_color(
                    f" {'─' * (overflow_start - left_block_width)}{overflow_label}"
                    f"{'─' * (width - overflow_start - overflow_label_width)}"
                )
            )

        if width >= status_width + 5:
            return self.border_color("── ") + status + self.border_color(f" {'─' * (width - status_width - 4)}")

        status = indicator.render_spinner_in_border(width)
        status_width = visible_width(status)
        prefix_width = min(3, max(0, width - status_width))
        return (
            self.border_color("─" * prefix_width)
            + status
            + self.border_color("─" * max(0, width - prefix_width - status_width))
        )

    def on_action(self, action: str, handler) -> None:
        """Register a handler for an app action."""
        self.action_handlers[action] = handler

    async def handle_input(self, data: str) -> None:
        # Check extension-registered shortcuts first
        if self.on_extension_shortcut is not None and self.on_extension_shortcut(data):
            return

        # Check for clipboard paste keybinding
        if self._keybindings.matches(data, "app.clipboard.pasteImage"):
            if self.on_paste_image is not None:
                self.on_paste_image()
            return

        # Check app keybindings first

        # Escape/interrupt - only if autocomplete is NOT active
        if self._keybindings.matches(data, "app.interrupt"):
            if not self.is_showing_autocomplete():
                # Use dynamic on_escape if set, otherwise registered handler
                handler = self.on_escape or self.action_handlers.get("app.interrupt")
                if handler is not None:
                    await handler()
                    return
            # Let parent handle escape for autocomplete cancellation
            await super().handle_input(data)
            return

        # Exit (Ctrl+D) - only when editor is empty
        if self._keybindings.matches(data, "app.exit") and len(self.get_text()) == 0:
            handler = self.on_ctrl_d or self.action_handlers.get("app.exit")
            if handler is not None:
                await handler()
            return
            # (non-empty editors fall through to delete-char-forward below)

        # Explicit history bindings take precedence over app actions while the
        # editor is focused. This lets users bind Ctrl+P even though it cycles
        # models by default.
        if self._keybindings.matches(data, "tui.editor.historyPrevious") or self._keybindings.matches(
            data, "tui.editor.historyNext"
        ):
            await super().handle_input(data)
            return

        # Check all other app actions
        for action, handler in list(self.action_handlers.items()):
            if action not in ("app.interrupt", "app.exit") and self._keybindings.matches(data, action):
                await handler()
                return

        # Pass to parent for editor handling
        await super().handle_input(data)
