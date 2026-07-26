"""Mirror of pi coding-agent src/cli/project-trust.ts.

This is the *startup* trust context, used before interactive mode owns a TUI;
the in-session one is `InteractiveMode._create_project_trust_context`. The
interactive branches route through the startup TUI helpers
(`startup_ui.show_startup_selector` / `show_startup_input`); non-interactive
modes and headless runs resolve to "no answer" as pi does.
"""

import sys
from dataclasses import dataclass
from typing import Any

from ..core.extensions.types import ProjectTrustContext
from ..utils.colors import cyan, red, yellow
from .startup_ui import show_startup_input, show_startup_selector


@dataclass(slots=True, kw_only=True)
class CreateProjectTrustContextOptions:
    cwd: str
    mode: str  # AppMode
    settings_manager: Any
    has_ui: bool


class _ProjectTrustUI:
    def __init__(self, options: CreateProjectTrustContextOptions):
        self._options = options

    async def select(self, title: str, select_options: list[str]) -> str | None:
        if not self._options.has_ui:
            return None
        if self._options.mode != "interactive":
            return None
        return await show_startup_selector(
            self._options.settings_manager,
            title,
            [{"label": option, "value": option} for option in select_options],
        )

    async def confirm(self, title: str, message: str) -> bool:
        if not self._options.has_ui:
            return False
        if self._options.mode != "interactive":
            return False
        selected = await show_startup_selector(
            self._options.settings_manager,
            f"{title}\n{message}",
            [{"label": "Yes", "value": True}, {"label": "No", "value": False}],
        )
        # pi: `?? false` — a selected `False` must survive, only cancel defaults.
        return selected if selected is not None else False

    async def input(self, title: str, placeholder: str | None = None) -> str | None:
        if not self._options.has_ui:
            return None
        if self._options.mode != "interactive":
            return None
        return await show_startup_input(self._options.settings_manager, title, placeholder)

    def notify(self, message: str, type: str = "info") -> None:
        if self._options.mode != "interactive":
            color = red if type == "error" else yellow if type == "warning" else cyan
            print(color(message), file=sys.stderr)


def create_project_trust_context(options: CreateProjectTrustContextOptions) -> ProjectTrustContext:
    return ProjectTrustContext(
        cwd=options.cwd,
        mode="tui" if options.mode == "interactive" else options.mode,
        has_ui=options.has_ui,
        ui=_ProjectTrustUI(options),
    )
