"""Mirror of pi coding-agent src/modes/interactive/components/status-indicator.ts."""

import math

from pidrei_tui import Loader, truncate_to_width

from ..theme import theme
from .countdown_timer import CountdownTimer
from .keybinding_hints import key_text


class StatusIndicator(Loader):
    """Kinds: "working" | "retry" | "compaction" | "branchSummary"."""

    def __init__(self, kind: str, ui, spinner_color_fn, message_color_fn, message: str, indicator=None) -> None:
        super().__init__(ui, spinner_color_fn, message_color_fn, message, indicator)
        self.kind = kind

    def dispose(self) -> None:
        self.stop()


class WorkingStatusIndicator(StatusIndicator):
    def __init__(self, ui, message: str, indicator=None, color_fn=None) -> None:
        super().__init__(
            "working",
            ui,
            color_fn if color_fn is not None else (lambda text: theme.fg("accent", text)),
            color_fn if color_fn is not None else (lambda text: theme.fg("muted", text)),
            message,
            indicator,
        )

    def render_in_border(self, width: int) -> str:
        rendered = super().render(width + 2)
        line = rendered[1] if len(rendered) > 1 else ""
        return truncate_to_width(line[1:].rstrip() if line.startswith(" ") else line.rstrip(), width, "")

    def render_spinner_in_border(self, width: int) -> str:
        return truncate_to_width(self._get_rendered_indicator(), width, "")


class RetryStatusIndicator(StatusIndicator):
    def __init__(self, ui, attempt: int, max_attempts: int, delay_ms: float) -> None:
        def retry_message(seconds: int) -> str:
            return f"Retrying ({attempt}/{max_attempts}) in {seconds}s... ({key_text('app.interrupt')} to cancel)"

        super().__init__(
            "retry",
            ui,
            lambda spinner: theme.fg("warning", spinner),
            lambda text: theme.fg("muted", text),
            retry_message(math.ceil(delay_ms / 1000)),
        )

        def on_expire() -> None:
            self._countdown = None

        self._countdown: CountdownTimer | None = CountdownTimer(
            delay_ms,
            ui,
            lambda seconds: self.set_message(retry_message(seconds)),
            on_expire,
        )

    def dispose(self) -> None:
        if self._countdown is not None:
            self._countdown.dispose()
            self._countdown = None
        super().dispose()


class CompactionStatusIndicator(StatusIndicator):
    """Reasons: "manual" | "threshold" | "overflow"."""

    def __init__(self, ui, reason: str) -> None:
        cancel_hint = f"({key_text('app.interrupt')} to cancel)"
        if reason == "manual":
            label = f"Compacting context... {cancel_hint}"
        else:
            prefix = "Context overflow detected, " if reason == "overflow" else ""
            label = f"{prefix}Auto-compacting... {cancel_hint}"
        super().__init__(
            "compaction",
            ui,
            lambda spinner: theme.fg("accent", spinner),
            lambda text: theme.fg("muted", text),
            label,
        )


class BranchSummaryStatusIndicator(StatusIndicator):
    def __init__(self, ui) -> None:
        super().__init__(
            "branchSummary",
            ui,
            lambda spinner: theme.fg("accent", spinner),
            lambda text: theme.fg("muted", text),
            f"Summarizing branch... ({key_text('app.interrupt')} to cancel)",
        )


class IdleStatus:
    def invalidate(self) -> None:
        # No cached state to invalidate.
        pass

    def render(self, width: int) -> list:
        empty_line = " " * width
        return [empty_line, empty_line]
