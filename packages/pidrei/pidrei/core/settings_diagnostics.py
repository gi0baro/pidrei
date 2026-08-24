"""Mirror of pi coding-agent src/core/settings-diagnostics.ts."""

from collections.abc import Iterable

from .agent_session_services import AgentSessionRuntimeDiagnostic
from .settings_manager import SettingsManager


__all__ = ["collect_settings_diagnostics", "deduplicate_diagnostics"]


def collect_settings_diagnostics(settings_manager: SettingsManager) -> list[AgentSessionRuntimeDiagnostic]:
    return [
        AgentSessionRuntimeDiagnostic(
            type="warning",
            message=(
                f"Invalid settings file {entry.path}: {entry.error}"
                if entry.path
                else f"Invalid {entry.scope} settings: {entry.error}"
            ),
        )
        for entry in settings_manager.drain_errors()
    ]


def deduplicate_diagnostics(
    diagnostics: Iterable[AgentSessionRuntimeDiagnostic],
) -> list[AgentSessionRuntimeDiagnostic]:
    """Remove duplicate type/message diagnostics while preserving their first occurrence.

    Startup and runtime settings managers can report the same file error.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[AgentSessionRuntimeDiagnostic] = []
    for diagnostic in diagnostics:
        key = (diagnostic.type, diagnostic.message)
        if key in seen:
            continue
        seen.add(key)
        unique.append(diagnostic)
    return unique
