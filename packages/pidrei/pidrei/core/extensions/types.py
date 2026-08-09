"""Mirror of pi coding-agent src/core/extensions/types.ts.

The tool layer pieces (ToolDefinition) plus the structural records the
ExtensionRunner, the loader and AgentSession need (Extension,
ExtensionRuntime, RegisteredTool, commands, LoadExtensionsResult). The
context objects extensions actually receive live in `runner.py`
(`_RunnerContext`/`_RunnerCommandContext`), because every field on them
resolves through the runner at access time.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


RUNTIME_NOT_INITIALIZED = "Extension runtime not initialized. Action methods cannot be called during extension loading."


class ExtensionContext:
    """Duck-typed stand-in for pi's ExtensionContext.

    The real context an extension sees is the runner's; this is the shape
    tools accept when there is no session behind them (they only duck-read
    optional attributes: session_manager, model, thinking_level, ...).
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
    # TUI shell rendering marker ("default" | "self").
    render_shell: str | None = None
    # TUI render hooks: render_call(args, theme, context) -> Component and
    # render_result(result, options, theme, context) -> Component.
    render_call: Any = None
    render_result: Any = None
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
    """CLI flag registered by an extension (pi's ExtensionFlag)."""

    type: str  # "boolean" | "string"
    description: str | None = None
    # Flag name without the leading "--" (pi carries it on the flag object;
    # pidrei registries also key flag maps by it).
    name: str = ""
    # Path of the extension that registered the flag.
    extension_path: str = ""
    default: Any = None


@dataclass(slots=True)
class ExtensionShortcut:
    """Keyboard shortcut registered by an extension (pi's ExtensionShortcut)."""

    shortcut: str
    handler: Any  # (ctx) -> None | awaitable
    description: str | None = None
    extension_path: str = ""


@dataclass(slots=True)
class ExtensionUIDialogOptions:
    """Options for extension UI dialog methods (pi's ExtensionUIDialogOptions).

    `cancel` stands in for pi's AbortSignal; `timeout` is in milliseconds."""

    cancel: Any = None
    timeout: float | None = None


@dataclass(slots=True)
class ProjectTrustContext:
    """Context handed to project_trust extension handlers and the trust
    prompt (pi's ProjectTrustContext). `mode` is "tui" | "print" | "json" |
    "rpc"; `ui` exposes select/confirm/input/notify."""

    cwd: str
    mode: str
    has_ui: bool
    ui: Any = None


@dataclass(slots=True)
class RegisteredTool:
    """Tool registered by an extension (pi's RegisteredTool)."""

    definition: ToolDefinition
    source_info: Any = None


@dataclass(slots=True)
class RegisteredCommand:
    """Slash command registered by an extension (pi's RegisteredCommand)."""

    name: str
    # (args: str, ctx) -> awaitable of None (async-only callback policy).
    handler: Any
    description: str | None = None
    source_info: Any = None
    # (argument_prefix) -> awaitable of list[AutocompleteItem] | None.
    get_argument_completions: Any = None


@dataclass(slots=True)
class ResolvedCommand(RegisteredCommand):
    """RegisteredCommand with its collision-resolved invocation name."""

    invocation_name: str = ""


@dataclass(slots=True)
class Extension:
    """Loaded extension record iterated by the ExtensionRunner."""

    path: str
    # Absolute path the module was imported from. For inline extensions
    # (`<inline:name>`) pi keeps the pseudo-path in both fields.
    resolved_path: str = ""
    source_info: Any = None
    # Omit this extension from the startup Extensions list.
    hidden: bool = False
    # event type -> handlers; a single extension may register several per event.
    handlers: dict[str, list[Any]] = field(default_factory=dict)
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)
    shortcuts: dict[str, Any] = field(default_factory=dict)
    message_renderers: dict[str, Any] = field(default_factory=dict)
    markdown_transformer: Any = None
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

        # register_tool() is valid during extension load; a refresh is only
        # needed once the session is bound.
        self.refresh_tools: Callable[[], None] = lambda: None

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
        self.get_commands: Callable[[], list[Any]] = list
        self.set_model: Callable[..., Any] | None = None
        self.get_thinking_level: Callable[[], Any] = lambda: "off"
        self.set_thinking_level: Callable[..., None] | None = None

        # Provider registration hooks. Pre-bind they queue, so a registration
        # made while extensions are still loading survives until the model
        # registry exists; bind_core() flushes the queues and replaces these
        # with direct calls, so later registrations need no /reload.
        self.register_provider: Callable[..., None] = self._queue_provider
        self.register_native_provider: Callable[..., None] = self._queue_native_provider
        self.unregister_provider: Callable[..., None] = self._unqueue_provider

        self._stale_message: str | None = None
        self._event_bus_unsubscribers: set = set()

    def _queue_provider(self, name: str, config: Any, extension_path: str = "<unknown>") -> None:
        self.pending_provider_registrations.append({"name": name, "config": config, "extension_path": extension_path})

    def _queue_native_provider(self, provider: Any, extension_path: str = "<unknown>") -> None:
        self.pending_native_provider_registrations.append({"provider": provider, "extension_path": extension_path})

    def _unqueue_provider(self, name: str, _extension_path: str = "<unknown>") -> None:
        self.pending_provider_registrations = [
            entry for entry in self.pending_provider_registrations if entry["name"] != name
        ]
        self.pending_native_provider_registrations = [
            entry for entry in self.pending_native_provider_registrations if entry["provider"].id != name
        ]

    def invalidate(self, message: str) -> None:
        """Mark this extension instance stale after runtime replacement or reload."""
        if self._stale_message is not None:
            return
        self._stale_message = message
        for unsubscribe in list(self._event_bus_unsubscribers):
            unsubscribe()
        self._event_bus_unsubscribers.clear()

    def track_event_bus_subscription(self, unsubscribe: Callable[[], None]) -> Callable[[], None]:
        """Retain an event-bus subscription until this runtime is invalidated."""
        active = True

        def tracked_unsubscribe() -> None:
            nonlocal active
            if not active:
                return
            active = False
            self._event_bus_unsubscribers.discard(tracked_unsubscribe)
            unsubscribe()

        self._event_bus_unsubscribers.add(tracked_unsubscribe)
        return tracked_unsubscribe

    def assert_active(self) -> None:
        if self._stale_message is not None:
            raise RuntimeError(self._stale_message)


@dataclass(slots=True)
class ExtensionLoadError:
    path: str
    error: str


@dataclass(slots=True)
class LoadExtensionsResult:
    """pi's LoadExtensionsResult."""

    extensions: list[Extension] = field(default_factory=list)
    errors: list[ExtensionLoadError] = field(default_factory=list)
    runtime: ExtensionRuntime = field(default_factory=ExtensionRuntime)
