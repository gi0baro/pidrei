"""Mirror of pi coding-agent src/modes/interactive/components/themed-text.ts."""

from pidrei_tui import Text


class ThemedText(Text):
    """Text whose content applies theme colors. Plain `Text` keeps the colors
    its string was built with, so a theme change, or the system theme
    receiving the terminal's colors, would leave it stale. This rebuilds the
    string from ``build`` after every invalidation, which the UI performs on
    theme changes.

    ``build`` must return the same content each time, apart from colors.
    Snapshot changing data before creating the component, or call
    `invalidate()` after changing state that ``build`` reads.
    """

    def __init__(self, build, padding_x: int = 1, padding_y: int = 1) -> None:
        super().__init__("", padding_x, padding_y)
        self._build = build
        self._stale = True

    def invalidate(self) -> None:
        super().invalidate()
        self._stale = True

    def render(self, width: int) -> list[str]:
        if self._stale:
            # After `set_text`: pidrei's goes through `invalidate()` (pi's
            # clears its cache directly), which marks the text stale again.
            self.set_text(self._build())
            self._stale = False
        return super().render(width)
