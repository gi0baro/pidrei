"""Subset of pi coding-agent src/core/extensions/types.ts.

The tool layer pieces (ToolDefinition) plus the structural records the
ExtensionRunner and AgentSession need (Extension, ExtensionRuntime,
RegisteredTool, commands, LoadExtensionsResult). Extension *loading* (the
ExtensionAPI surface handed to extension factories, discovery, the Python
extension ABI) is Phase 5 — in Phase 3 every LoadExtensionsResult carries an
empty extension list, so the runner's hook bus runs with zero handlers. The
TUI render hooks (renderCall/renderResult/renderShell) are Phase 4 and are
represented only by the optional `render_shell` marker.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class ExtensionContext:
    """Placeholder for pi's ExtensionContext (Phase 5).

    Tools only duck-read optional attributes from it (session_manager, model,
    thinking_level, ...); a plain attribute bag keeps the seam alive.
    """

    def __init__(self, **attributes: Any):
        for name, value in attributes.items():
            setattr(self, name, value)

    def __getattr__(self, _name: str) -> Any:
        return None


@dataclass(slots=True, kw_only=True)
class ToolDefinition:
    """Definition-first tool record (pi's ToolDefinition<TParams, TDetails>)."""

    # Tool name (used in LLM tool calls)
    name: str
    # Human-readable label for UI
    label: str
    # Description for LLM
    description: str
    # Parameter schema (JSON Schema; pi: TypeBox)
    parameters: dict[str, Any]
    # Execute the tool: async (tool_call_id, params, cancel, on_update, ctx) -> AgentToolResult
    execute: Any
    # Optional one-line snippet for the Available tools section in the default system prompt.
    prompt_snippet: str | None = None
    # Optional guideline bullets appended to the default system prompt Guidelines section.
    prompt_guidelines: list[str] | None = None
    # Optional provider-side constrained sampling request for this tool.
    constrained_sampling: Any = None
    # TUI shell rendering marker ("default" | "self"); renderers themselves are Phase 4.
    render_shell: str | None = None
    # Optional compatibility shim to prepare raw tool call arguments before schema validation.
    prepare_arguments: Any = None
    # Per-tool execution mode override ("sequential" | "parallel").
    execution_mode: str | None = None
    # Extra metadata slot mirroring pi's open object shape.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtensionError:
    """Error surfaced from an extension handler (pi's ExtensionError)."""

    extension_path: str
    event: str
    error: str
    stack: str | None = None


@dataclass(slots=True)
class ExtensionFlag:
    """CLI flag registered by an extension (pi's ExtensionFlag subset)."""

    type: str  # "boolean" | "string"
    description: str | None = None


@dataclass(slots=True)
class RegisteredTool:
    """Tool registered by an extension (pi's RegisteredTool)."""

    definition: ToolDefinition
    source_info: Any = None


@dataclass(slots=True)
class RegisteredCommand:
    """Slash command registered by an extension (pi's RegisteredCommand)."""

    name: str
    handler: Any  # async (args: str, ctx) -> None
    description: str | None = None
    source_info: Any = None


@dataclass(slots=True)
class ResolvedCommand(RegisteredCommand):
    """RegisteredCommand with its collision-resolved invocation name."""

    invocation_name: str = ""


@dataclass(slots=True)
class Extension:
    """Loaded extension record iterated by the ExtensionRunner.

    Phase 3 never constructs these (loading is Phase 5); the runner and its
    tests only need the structural shape.
    """

    path: str
    # event type -> handlers; a single extension may register several per event.
    handlers: dict[str, list[Any]] = field(default_factory=dict)
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)
    shortcuts: dict[str, Any] = field(default_factory=dict)
    message_renderers: dict[str, Any] = field(default_factory=dict)
    entry_renderers: dict[str, Any] = field(default_factory=dict)


class ExtensionRuntime:
    """Shared mutable runtime the loader creates and AgentSession binds.

    All extension API objects reference this; bind_core copies the session's
    action callables in, and provider registrations queued during extension
    loading are flushed through it.
    """

    def __init__(self) -> None:
        self.flag_values: dict[str, Any] = {}
        self.pending_provider_registrations: list[Any] = []
        self.pending_native_provider_registrations: list[Any] = []

        # Actions copied in by ExtensionRunner.bind_core().
        self.send_message: Callable[..., None] | None = None
        self.send_user_message: Callable[..., None] | None = None
        self.append_entry: Callable[..., None] | None = None
        self.set_session_name: Callable[..., None] | None = None
        self.get_session_name: Callable[..., Any] | None = None
        self.set_label: Callable[..., None] | None = None
        self.get_active_tools: Callable[[], list[str]] = list
        self.get_all_tools: Callable[[], list[Any]] = list
        self.set_active_tools: Callable[..., None] | None = None
        self.refresh_tools: Callable[[], None] | None = None
        self.get_commands: Callable[[], list[Any]] = list
        self.set_model: Callable[..., Any] | None = None
        self.get_thinking_level: Callable[[], Any] = lambda: "off"
        self.set_thinking_level: Callable[..., None] | None = None

        # Provider registration hooks (rebound by bind_core to take effect
        # immediately without a /reload).
        self.register_provider: Callable[..., None] | None = None
        self.register_native_provider: Callable[..., None] | None = None
        self.unregister_provider: Callable[..., None] | None = None

        self._stale_message: str | None = None

    def invalidate(self, message: str) -> None:
        if self._stale_message is None:
            self._stale_message = message


@dataclass(slots=True)
class ExtensionLoadError:
    path: str
    error: str


@dataclass(slots=True)
class LoadExtensionsResult:
    """pi's LoadExtensionsResult; extension loading itself is Phase 5, so the
    extension list is always empty in Phase 3 while the runtime is real."""

    extensions: list[Extension] = field(default_factory=list)
    errors: list[ExtensionLoadError] = field(default_factory=list)
    runtime: ExtensionRuntime = field(default_factory=ExtensionRuntime)
