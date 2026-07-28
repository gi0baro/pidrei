"""Mirror of pi coding-agent src/modes/interactive/components/custom-editor.ts."""

from pidrei_tui import Editor


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
    """Editor that handles app-level keybindings for the coding agent."""

    def __init__(self, tui, theme: dict, keybindings, options: dict | None = None) -> None:
        super().__init__(tui, theme, options)
        self._keybindings = keybindings
        self.action_handlers: dict = {}

        # Special handlers that can be dynamically replaced
        self.on_escape = None
        self.on_ctrl_d = None
        self.on_paste_image = None
        # Handler for extension-registered shortcuts. Returns True if handled.
        self.on_extension_shortcut = None

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

        # Check all other app actions
        for action, handler in list(self.action_handlers.items()):
            if action not in ("app.interrupt", "app.exit") and self._keybindings.matches(data, action):
                await handler()
                return

        # Pass to parent for editor handling
        await super().handle_input(data)
