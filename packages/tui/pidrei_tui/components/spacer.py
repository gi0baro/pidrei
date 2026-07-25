"""Spacer component that renders empty lines (port of pi tui ``components/spacer.ts``)."""

__all__ = ["Spacer"]


class Spacer:
    __slots__ = ("_lines",)

    def __init__(self, lines: int = 1) -> None:
        self._lines = lines

    def set_lines(self, lines: int) -> None:
        self._lines = lines

    def invalidate(self) -> None:
        # No cached state to invalidate currently
        pass

    def render(self, _width: int) -> list[str]:
        return ["" for _ in range(self._lines)]
