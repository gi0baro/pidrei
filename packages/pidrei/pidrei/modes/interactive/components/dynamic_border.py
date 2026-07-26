"""Mirror of pi coding-agent src/modes/interactive/components/dynamic-border.ts."""

from ..theme import theme


class DynamicBorder:
    """Dynamic border component that adjusts to viewport width."""

    def __init__(self, color=None) -> None:
        self._color = color if color is not None else (lambda text: theme.fg("border", text))

    def invalidate(self) -> None:
        # No cached state to invalidate currently
        pass

    def render(self, width: int) -> list:
        return [self._color("─" * max(1, width))]
