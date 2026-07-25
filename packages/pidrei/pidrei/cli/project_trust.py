"""Mirror of pi coding-agent src/cli/project-trust.ts.

pi's interactive branches route through the startup TUI selector
(src/cli/startup-ui.ts), which lands with the Phase 4 TUI slice. Until
then interactive select/confirm/input resolve to "no answer", exactly like
pi's non-interactive modes.
"""

import sys
from dataclasses import dataclass
from typing import Any

from ..core.extensions.types import ProjectTrustContext
from ..utils.colors import cyan, red, yellow


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
        # TODO(Phase 4): show_startup_selector(settings_manager, title, options)
        return None

    async def confirm(self, title: str, message: str) -> bool:
        if not self._options.has_ui:
            return False
        if self._options.mode != "interactive":
            return False
        # TODO(Phase 4): show_startup_selector(settings_manager, title+message, yes/no)
        return False

    async def input(self, title: str, placeholder: str | None = None) -> str | None:
        if not self._options.has_ui:
            return None
        if self._options.mode != "interactive":
            return None
        # TODO(Phase 4): show_startup_input(settings_manager, title, placeholder)
        return None

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
