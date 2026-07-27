"""Mirror of pi coding-agent src/modes/interactive/components/extension-selector.ts.

Generic selector component for extensions. Displays a list of string options
with keyboard navigation.
"""

from pidrei_tui import Container, Spacer, Text, get_keybindings

from ..theme import theme
from .countdown_timer import CountdownTimer
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint


class ExtensionSelectorComponent(Container):
    """Opts: ``{"tui"?, "timeout"?, "onToggleToolsExpanded"?}``."""

    def __init__(self, title: str, options: list, on_select, on_cancel, opts: dict | None = None) -> None:
        super().__init__()
        opts = opts or {}

        self._options = options
        self._selected_index = 0
        self._on_select_callback = on_select
        self._on_cancel_callback = on_cancel
        self._on_toggle_tools_expanded = opts.get("onToggleToolsExpanded")
        self._base_title = title
        self._countdown: CountdownTimer | None = None

        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))

        self._title_text = Text(theme.fg("accent", theme.bold(title)), 1, 0)
        self.add_child(self._title_text)
        self.add_child(Spacer(1))

        timeout = opts.get("timeout")
        if timeout and timeout > 0 and opts.get("tui") is not None:
            self._countdown = CountdownTimer(
                timeout,
                opts["tui"],
                lambda s: self._title_text.set_text(theme.fg("accent", theme.bold(f"{self._base_title} ({s}s)"))),
                lambda: self._on_cancel_callback(),
            )

        self._list_container = Container()
        self.add_child(self._list_container)
        self.add_child(Spacer(1))
        self.add_child(
            Text(
                raw_key_hint("↑↓", "navigate")
                + "  "
                + key_hint("tui.select.confirm", "select")
                + "  "
                + key_hint("tui.select.cancel", "cancel"),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

        self._update_list()

    def _update_list(self) -> None:
        self._list_container.clear()
        for i, option in enumerate(self._options):
            is_selected = i == self._selected_index
            if is_selected:
                text = theme.fg("accent", "→ ") + theme.fg("accent", option)
            else:
                text = f"  {theme.fg('text', option)}"
            self._list_container.add_child(Text(text, 1, 0))

    async def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "app.tools.expand"):
            if self._on_toggle_tools_expanded is not None:
                self._on_toggle_tools_expanded()
        elif kb.matches(key_data, "tui.select.up") or key_data == "k":
            self._selected_index = max(0, self._selected_index - 1)
            self._update_list()
        elif kb.matches(key_data, "tui.select.down") or key_data == "j":
            self._selected_index = min(len(self._options) - 1, self._selected_index + 1)
            self._update_list()
        elif kb.matches(key_data, "tui.select.confirm") or key_data == "\n":
            if 0 <= self._selected_index < len(self._options):
                self._on_select_callback(self._options[self._selected_index])
        elif kb.matches(key_data, "tui.select.cancel"):
            self._on_cancel_callback()

    def dispose(self) -> None:
        if self._countdown is not None:
            self._countdown.dispose()
