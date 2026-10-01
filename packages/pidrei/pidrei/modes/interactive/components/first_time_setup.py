"""Mirror of pi coding-agent src/modes/interactive/components/first-time-setup.ts.

DIVERGED: pi's second step asks for an anonymous-usage-analytics opt-in.
pidrei sends no telemetry, so the dialog is theme-only and confirming the
theme finishes setup.
"""

from pidrei_tui import Container, Spacer, Text, get_keybindings

from ....config import APP_NAME
from ..theme import SYSTEM_THEME_NAME, theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint


THEME_OPTIONS = [
    {"value": SYSTEM_THEME_NAME, "label": "System (matches your terminal colors)"},
    {"value": "dark", "label": "Dark"},
    {"value": "light", "label": "Light"},
]

# pi's logo followed by a 3 (pidrei-only), one pixel apart.
SETUP_LOGO_LINES = ["██████    ██████", "██  ██      ████", "████  ██      ██", "██    ██  ██████"]


class FirstTimeSetupComponent(Container):
    """First-time setup dialog: theme choice.

    Options: ``{"onThemePreview", "onSubmit", "onCancel"}``; submit receives
    a ``{"theme"}`` record.
    """

    def __init__(self, options: dict) -> None:
        super().__init__()
        self._options = options
        self._theme_index = 0
        self._update()

    def invalidate(self) -> None:
        """Rebuild on theme changes, e.g. when the system theme receives the
        terminal's colors."""
        self._update()
        super().invalidate()

    # Rebuild the whole dialog on every change so theme previews recolor all
    # text.
    def _update(self) -> None:
        self.clear()
        self.add_child(DynamicBorder())
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", "\n".join(SETUP_LOGO_LINES)), 1, 0))
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("accent", theme.bold(f"Welcome to {APP_NAME}, the minimal coding agent.")), 1, 0))
        self.add_child(Spacer(1))

        self.add_child(Text(theme.fg("text", "Pick a theme."), 1, 0))
        self.add_child(Spacer(1))
        self._add_option_list([option["label"] for option in THEME_OPTIONS], self._theme_index)

        self.add_child(Spacer(1))
        self.add_child(
            Text(
                raw_key_hint("↑↓", "navigate")
                + "  "
                + key_hint("tui.select.confirm", "finish")
                + "  "
                + key_hint("tui.select.cancel", "skip setup"),
                1,
                0,
            )
        )
        self.add_child(Spacer(1))
        self.add_child(DynamicBorder())

    def _add_option_list(self, labels: list, selected_index: int) -> None:
        for i, raw_label in enumerate(labels):
            is_selected = i == selected_index
            prefix = theme.fg("accent", "→ ") if is_selected else "  "
            label = theme.fg("accent", raw_label) if is_selected else theme.fg("text", raw_label)
            self.add_child(Text(f"{prefix}{label}", 1, 0))

    def _move_selection(self, delta: int) -> None:
        next_index = max(0, min(len(THEME_OPTIONS) - 1, self._theme_index + delta))
        if next_index != self._theme_index:
            self._theme_index = next_index
            self._options["onThemePreview"](THEME_OPTIONS[self._theme_index]["value"])
        self._update()

    def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.up") or key_data == "k":
            self._move_selection(-1)
        elif kb.matches(key_data, "tui.select.down") or key_data == "j":
            self._move_selection(1)
        elif kb.matches(key_data, "tui.select.confirm") or key_data == "\n":
            self._options["onSubmit"]({"theme": THEME_OPTIONS[self._theme_index]["value"]})
        elif kb.matches(key_data, "tui.select.cancel"):
            self._options["onCancel"]()
