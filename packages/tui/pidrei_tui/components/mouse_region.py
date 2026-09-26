"""MouseRegion component (port of pi tui ``components/mouse-region.ts``).

``on_mouse`` is a sync ``(event) -> TuiMouseEventResult | None`` callback,
like pi's ``MouseRegionHandler``.
"""

from ..tui import TuiMouseDispatchResult, TuiMouseEvent, TuiMouseEventResult, dispatch_mouse_event


__all__ = ["MouseRegion"]


class MouseRegion:
    """Adds mouse handling to an existing component without changing its rendering."""

    def __init__(self, child, on_mouse) -> None:
        self._child = child
        self._on_mouse = on_mouse

    def render(self, width: int) -> list[str]:
        return self._child.render(width)

    def handle_mouse(self, event: TuiMouseEvent) -> TuiMouseDispatchResult | TuiMouseEventResult | None:
        child_result = dispatch_mouse_event(self._child, event)
        return child_result if child_result is not None else self._on_mouse(event)

    def invalidate(self) -> None:
        self._child.invalidate()
