"""Mirror of pi coding-agent src/core/diagnostics.ts."""

from dataclasses import dataclass


@dataclass(slots=True)
class ResourceCollision:
    resource_type: str  # "extension" | "skill" | "prompt" | "theme"
    name: str  # skill name, command/tool/flag name, prompt name, theme name
    winner_path: str
    loser_path: str
    winner_source: str | None = None  # e.g., "npm:foo", "git:...", "local"
    loser_source: str | None = None


@dataclass(slots=True)
class ResourceDiagnostic:
    type: str  # "warning" | "error" | "collision"
    message: str
    path: str | None = None
    collision: ResourceCollision | None = None
