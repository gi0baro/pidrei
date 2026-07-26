"""Mirror of pi coding-agent src/core/extensions/index.ts (Phase 3 subset)."""

from .runner import (
    ExtensionRunner,
    InputEventResult,
    ResourcesDiscoverPaths,
    emit_project_trust_event,
    emit_session_shutdown_event,
)
from .types import (
    Extension,
    ExtensionContext,
    ExtensionError,
    ExtensionFlag,
    ExtensionLoadError,
    ExtensionRuntime,
    LoadExtensionsResult,
    RegisteredCommand,
    RegisteredTool,
    ResolvedCommand,
    ToolDefinition,
)
from .wrapper import wrap_registered_tool, wrap_registered_tools


__all__ = [
    "Extension",
    "ExtensionContext",
    "ExtensionError",
    "ExtensionFlag",
    "ExtensionLoadError",
    "ExtensionRunner",
    "ExtensionRuntime",
    "InputEventResult",
    "LoadExtensionsResult",
    "RegisteredCommand",
    "RegisteredTool",
    "ResolvedCommand",
    "ResourcesDiscoverPaths",
    "ToolDefinition",
    "emit_project_trust_event",
    "emit_session_shutdown_event",
    "wrap_registered_tool",
    "wrap_registered_tools",
]
