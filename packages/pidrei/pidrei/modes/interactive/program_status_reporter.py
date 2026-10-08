"""Mirror of pi coding-agent src/modes/interactive/program-status-reporter.ts.

Divergence: pi calls the reporter on its one thread. Here it is called under
the UI state lock (session events, dialog mounts and hides) and without it
(logins), so its state is under its own lock. `report()` writes to the
terminal while still holding it, so the order of writes is the order of
state changes when two callers race. Lock order: UI state lock, then this
lock, then the terminal's protocol lock; the terminal's reader holds the
protocol lock and never calls the reporter.
"""

import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from pidrei_tui import ProgramStatus, Terminal

from ...config import APP_NAME


@dataclass(frozen=True, slots=True)
class BlockedStatus:
    kind: str  # "permission" | "question" | "auth"
    message: str


def _first_line(text: str | None) -> str:
    line = re.split(r"\r?\n", text, maxsplit=1)[0].strip() if text is not None else ""
    return line or "Error"


class ProgramStatusReporter:
    """Reports interactive-mode state to the terminal (OSC 7501): `working`
    during agent runs and compaction, `blocked` while a dialog waits for the
    user, then `done`, `error`, or `idle` once the run settles. Messages are
    limited to the session name, dialog titles, and the first line of errors;
    prompts and assistant output are never reported."""

    def __init__(self, get_terminal: Callable[[], Terminal], get_session_name: Callable[[], str | None]) -> None:
        self._get_terminal = get_terminal
        self._get_session_name = get_session_name
        self._lock = threading.Lock()
        self._run_active = False
        self._compacting = False
        # Outcome of the current run, reported once it settles.
        self._run_result = ProgramStatus(state="done")
        # Status while no run is active.
        self._resting_status = ProgramStatus(state="idle")
        # Open dialogs by source, in the order they opened. The most recent one is reported.
        self._blocked: dict[str, BlockedStatus] = {}
        self._last_report: ProgramStatus | None = None

    def handle_event(self, event: Any) -> None:
        with self._lock:
            match event.type:
                case "agent_start":
                    self._run_active = True
                    self._run_result = ProgramStatus(state="done")
                case "message_end":
                    # The latest response decides the outcome, so a retried error is replaced by its successful retry.
                    message = event.message
                    if message.role != "assistant":
                        return
                    self._run_result = (
                        ProgramStatus(state="error", message=_first_line(message.error_message))
                        if message.stop_reason == "error"
                        else ProgramStatus(state="done")
                    )
                case "compaction_start":
                    self._compacting = True
                case "compaction_end":
                    self._compacting = False
                    if self._run_active:
                        # A failed recovery compaction ends the run unless a later response succeeds.
                        if event.aborted:
                            self._run_result = ProgramStatus(state="idle")
                        elif event.error_message:
                            self._run_result = ProgramStatus(state="error", message=_first_line(event.error_message))
                    elif event.aborted:
                        self._resting_status = ProgramStatus(state="idle")
                    elif event.reason == "manual":
                        self._resting_status = (
                            ProgramStatus(state="error", message=_first_line(event.error_message))
                            if event.error_message
                            else ProgramStatus(state="done")
                        )
                case "agent_settled":
                    self._run_active = False
                    self._resting_status = ProgramStatus(state="idle") if event.aborted else self._run_result
                case "session_info_changed":
                    # The session name is part of working and done reports.
                    pass
                case _:
                    return
            self._report_locked()

    def set_blocked(self, source: str, status: BlockedStatus | None) -> None:
        """Report `blocked` for a dialog until it is cleared with None. Reopening a source replaces it."""
        with self._lock:
            self._blocked.pop(source, None)
            if status is not None:
                self._blocked[source] = status
            self._report_locked()

    def reset(self) -> None:
        """Forget the previous session's run, for example after switching sessions."""
        with self._lock:
            self._run_active = False
            self._compacting = False
            self._run_result = ProgramStatus(state="done")
            self._resting_status = ProgramStatus(state="idle")
            self._report_locked()

    def report(self) -> None:
        with self._lock:
            self._report_locked()

    def _report_locked(self) -> None:
        status = replace(self._current_status_locked(), app=APP_NAME)
        if status == self._last_report:
            return
        self._last_report = status
        self._get_terminal().set_program_status(status)

    def _current_status_locked(self) -> ProgramStatus:
        if self._blocked:
            blocked = list(self._blocked.values())[-1]
            return ProgramStatus(state="blocked", kind=blocked.kind, message=blocked.message)  # type: ignore[arg-type]
        if self._compacting:
            return ProgramStatus(state="working", message="Compacting context")
        status = ProgramStatus(state="working") if self._run_active else self._resting_status
        if status.state in ("working", "done"):
            return replace(status, message=self._get_session_name())
        return status
