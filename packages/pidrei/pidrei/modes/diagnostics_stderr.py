"""Mirror of pi coding-agent src/modes/diagnostics-stderr.ts.

Writes runtime diagnostics to stderr for the CLI and the non-interactive modes.
"""

from collections.abc import Iterable

from ..core.agent_session_services import AgentSessionRuntimeDiagnostic
from ..core.output_guard import write_stderr
from ..utils.colors import dim, red, yellow


def write_diagnostics_to_stderr(diagnostics: Iterable[AgentSessionRuntimeDiagnostic]) -> None:
    """Print each diagnostic to stderr as `Error: ...` (red), `Warning: ...` (yellow) or a dim info line."""
    for diagnostic in diagnostics:
        color = red if diagnostic.type == "error" else yellow if diagnostic.type == "warning" else dim
        prefix = "Error: " if diagnostic.type == "error" else "Warning: " if diagnostic.type == "warning" else ""
        write_stderr(color(f"{prefix}{diagnostic.message}") + "\n")
