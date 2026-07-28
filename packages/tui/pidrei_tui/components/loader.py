"""Loader component that updates with an optional spinning animation.

Port of pi tui ``components/loader.ts``. pi drives the animation with
``setInterval``; here it is an ``Interval`` (a cooperative tonio task), so a
``Loader`` must be constructed on a tonio runtime thread.
"""

from .._timers import Interval
from .text import Text


__all__ = ["DEFAULT_FRAMES", "DEFAULT_INTERVAL_MS", "Loader"]

DEFAULT_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
DEFAULT_INTERVAL_MS = 80


class Loader(Text):
    def __init__(
        self,
        ui,
        spinner_color_fn,
        message_color_fn,
        message: str = "Loading...",
        indicator: dict | None = None,
    ) -> None:
        """``indicator`` mirrors pi's ``LoaderIndicatorOptions``: an optional
        ``{"frames": [...], "intervalMs": n}`` record. Passing a record (even
        an empty one) renders the frames verbatim instead of colored, exactly
        like pi's ``indicator !== undefined`` check; ``frames: []`` hides the
        indicator.
        """
        super().__init__("", 1, 0)
        self._frames = list(DEFAULT_FRAMES)
        self._interval_ms = DEFAULT_INTERVAL_MS
        self._current_frame = 0
        self._interval: Interval | None = None
        self._ui = ui
        self._render_indicator_verbatim = False
        self._spinner_color_fn = spinner_color_fn
        self._message_color_fn = message_color_fn
        self._message = message
        self.set_indicator(indicator)

    def render(self, width: int) -> list[str]:
        return ["", *super().render(width)]

    def start(self) -> None:
        self._update_display()
        self._restart_animation()

    def stop(self) -> None:
        if self._interval is not None:
            self._interval.cancel()
            self._interval = None

    def set_message(self, message: str) -> None:
        self._message = message
        self._update_display()

    def set_indicator(self, indicator: dict | None = None) -> None:
        self._render_indicator_verbatim = indicator is not None
        frames = indicator.get("frames") if indicator is not None else None
        self._frames = list(frames) if frames is not None else list(DEFAULT_FRAMES)
        interval_ms = indicator.get("intervalMs") if indicator is not None else None
        self._interval_ms = interval_ms if interval_ms is not None and interval_ms > 0 else DEFAULT_INTERVAL_MS
        self._current_frame = 0
        self.start()

    def _restart_animation(self) -> None:
        self.stop()
        if len(self._frames) <= 1:
            return

        async def advance() -> None:
            self._current_frame = (self._current_frame + 1) % len(self._frames)
            self._update_display()

        self._interval = Interval(self._interval_ms, advance)

    def _update_display(self) -> None:
        frame = self._frames[self._current_frame] if self._current_frame < len(self._frames) else ""
        rendered_frame = frame if self._render_indicator_verbatim else self._spinner_color_fn(frame)
        indicator = f"{rendered_frame} " if frame else ""
        self.set_text(f"{indicator}{self._message_color_fn(self._message)}")
        if self._ui is not None:
            self._ui.request_render()
