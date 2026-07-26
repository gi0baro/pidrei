"""Mirror of pi coding-agent src/core/extensions/index.ts."""

from .loader import (
    ExtensionAPI,
    clear_extension_cache,
    create_extension_runtime,
    discover_and_load_extensions,
    discover_extensions_in_dir,
    load_extension_from_factory,
    load_extensions,
    load_extensions_cached,
    resolve_extension_entries,
)
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
    ExtensionShortcut,
    LoadExtensionsResult,
    RegisteredCommand,
    RegisteredTool,
    ResolvedCommand,
    ToolDefinition,
)
from .wrapper import wrap_registered_tool, wrap_registered_tools


__all__ = [
    "Extension",
    "ExtensionAPI",
    "ExtensionContext",
    "ExtensionError",
    "ExtensionFlag",
    "ExtensionLoadError",
    "ExtensionRunner",
    "ExtensionRuntime",
    "ExtensionShortcut",
    "InputEventResult",
    "LoadExtensionsResult",
    "RegisteredCommand",
    "RegisteredTool",
    "ResolvedCommand",
    "ResourcesDiscoverPaths",
    "ToolDefinition",
    "clear_extension_cache",
    "create_extension_runtime",
    "discover_and_load_extensions",
    "discover_extensions_in_dir",
    "emit_project_trust_event",
    "emit_session_shutdown_event",
    "load_extension_from_factory",
    "load_extensions",
    "load_extensions_cached",
    "resolve_extension_entries",
    "wrap_registered_tool",
    "wrap_registered_tools",
]
