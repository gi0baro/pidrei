"""Mirror of pi coding-agent src/modes/interactive/components/first-time-setup.ts."""

from pidrei_tui import Container, Spacer, Text, get_keybindings

from ....config import APP_NAME
from ..theme import theme
from .dynamic_border import DynamicBorder
from .keybinding_hints import key_hint, raw_key_hint


THEME_OPTIONS = [
    {"value": "dark", "label": "Dark"},
    {"value": "light", "label": "Light"},
]

ANALYTICS_OPTIONS = [
    {"value": True, "label": "Share anonymous usage data"},
    {"value": False, "label": "Don't share"},
]

SETUP_LOGO_LINES = ["██████", "██  ██", "████  ██", "██    ██"]


class FirstTimeSetupComponent(Container):
    """First-time setup dialog: theme choice and analytics opt-in.

    Options: ``{"detectedTheme", "onThemePreview", "onSubmit", "onCancel"}``;
    submit receives a ``{"theme", "shareAnalytics"}`` record.
    """

    def __init__(self, options: dict) -> None:
        super().__init__()
        self._options = options
        self._step = "theme"
        self._analytics_index = 0
        self._theme_index = max(
            0,
            next((i for i, option in enumerate(THEME_OPTIONS) if option["value"] == options["detectedTheme"]), -1),
        )
        self._update()

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

        if self._step == "theme":
            self.add_child(Text(theme.fg("text", "Pick a theme."), 1, 0))
            self.add_child(
                Text(theme.fg("muted", f"Detected system appearance: {self._options['detectedTheme']}"), 1, 0)
            )
            self.add_child(Spacer(1))
            self._add_option_list([option["label"] for option in THEME_OPTIONS], self._theme_index)
        else:
            self.add_child(Text(theme.fg("text", "Opt-in to anonymous usage data sharing?"), 1, 0))
            self.add_child(
                Text(
                    theme.fg(
                        "muted",
                        "Opting in stores a tracking identifier in settings.json and enables anonymous\n"
                        "usage analytics. This helps us to better debug, reproduce, and resolve issues\n"
                        "and bugs within pidrei. You can observe what is shared using /privacy and make\n"
                        "changes anytime in settings.json.",
                    ),
                    1,
                    0,
                )
            )
            self.add_child(Spacer(1))
            self._add_option_list([option["label"] for option in ANALYTICS_OPTIONS], self._analytics_index)

        self.add_child(Spacer(1))
        self.add_child(
            Text(
                raw_key_hint("↑↓", "navigate")
                + "  "
                + key_hint("tui.select.confirm", "continue" if self._step == "theme" else "finish")
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

    async def _move_selection(self, delta: int) -> None:
        if self._step == "theme":
            next_index = max(0, min(len(THEME_OPTIONS) - 1, self._theme_index + delta))
            if next_index != self._theme_index:
                self._theme_index = next_index
                # Coroutine-returning by contract: previewing loads the theme
                # from disk (never-block rule).
                await self._options["onThemePreview"](THEME_OPTIONS[self._theme_index]["value"])
        else:
            self._analytics_index = max(0, min(len(ANALYTICS_OPTIONS) - 1, self._analytics_index + delta))
        self._update()

    async def handle_input(self, key_data: str) -> None:
        kb = get_keybindings()
        if kb.matches(key_data, "tui.select.up") or key_data == "k":
            await self._move_selection(-1)
        elif kb.matches(key_data, "tui.select.down") or key_data == "j":
            await self._move_selection(1)
        elif kb.matches(key_data, "tui.select.confirm") or key_data == "\n":
            if self._step == "theme":
                self._step = "analytics"
                self._update()
            else:
                self._options["onSubmit"](
                    {
                        "theme": THEME_OPTIONS[self._theme_index]["value"],
                        "shareAnalytics": ANALYTICS_OPTIONS[self._analytics_index]["value"],
                    }
                )
        elif kb.matches(key_data, "tui.select.cancel"):
            self._options["onCancel"]()
