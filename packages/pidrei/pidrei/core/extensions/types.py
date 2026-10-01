"""Mirror of pi coding-agent src/core/extensions/types.ts.

The tool layer pieces (ToolDefinition) plus the structural records the
ExtensionRunner, the loader and AgentSession need (Extension,
ExtensionRuntime, RegisteredTool, commands, LoadExtensionsResult). The
context objects extensions actually receive live in `runner.py`
(`_RunnerContext`/`_RunnerCommandContext`), because every field on them
resolves through the runner at access time.
"""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ..mcp_servers import McpServerRegistry


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


# How the model reaches a tool. "Callable" means callable from other tools
# through `ctx.execute_tool()`, as the codemode tool does.
# - "direct": declared to the model while active, and callable while active.
# - "model-only": declared to the model while active, never callable. Use it for
#   orchestrating or interactive tools.
# - "codemode": callable whenever registered. Not declared to the model unless
#   explicitly activated. Codemode tools list it in their description.
# - "deferred": like "codemode", but codemode tools do not list it; tool search can find it.
# - "hidden": registered but unreachable. Activating it has no effect.
# "direct" and "model-only" tools are activated when they are registered; the
# others are not. The active tool set (`get_active_tools`/`set_active_tools`)
# is the set declared to the model.
type ToolExposure = Literal["direct", "model-only", "codemode", "deferred", "hidden"]


@dataclass(slots=True, frozen=True, kw_only=True)
class ToolAnnotations:
    """Hints about what a tool does, with the meaning of MCP tool annotations.
    They come from the tool's author and are not verified; permission
    extensions can use them to decide which calls to confirm."""

    # The tool does not modify its environment.
    read_only_hint: bool | None = None
    # The tool may delete or overwrite data, rather than only add to it. Meaningful when not read-only.
    destructive_hint: bool | None = None
    # Repeating a call with the same arguments has no further effect. Meaningful when not read-only.
    idempotent_hint: bool | None = None
    # The tool reaches an open world of external entities, such as the web, rather than a closed domain.
    open_world_hint: bool | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class ToolNamespace:
    """A group of related tools, such as the tools of one MCP server. Codemode tools list them together."""

    # For example `mcp__docs`.
    name: str
    # Shown once above the group's tools.
    description: str | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class ToolLoadout:
    """The tools of a session as `ToolDefinition.prepare_loadout` sees them."""

    # Tools declared to the model (the active tools), in order, with their original descriptions.
    declared: tuple[Any, ...]
    # Tools callable through `ctx.execute_tool()`.
    callable: tuple[Any, ...]
    # Every registered tool.
    registered: tuple[Any, ...]
    get_exposure: Callable[[str], ToolExposure]
    get_namespace: Callable[[str], ToolNamespace | None]


@dataclass(slots=True, frozen=True, kw_only=True)
class ToolLoadoutChanges:
    """Changes `ToolDefinition.prepare_loadout` makes to what the model sees."""

    # Model-facing descriptions of declared tools, by tool name.
    descriptions: dict[str, str] | None = None
    # Declared tools whose declarations requests leave out. They stay active
    # and callable, and the transcript still declares them, so the active set
    # survives `/tree` and resume.
    hidden_declarations: tuple[str, ...] | None = None


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
    # JSON Schema of `structured_content` in successful results. Tools that
    # declare it should always set `structured_content`; codemode scripts then
    # receive it instead of the text content.
    output_schema: dict[str, Any] | None = None
    # How the model reaches the tool. None means "direct". See `ToolExposure`.
    exposure: ToolExposure | None = None
    # Group the tool belongs to, for example its MCP server.
    namespace: ToolNamespace | None = None
    # Hints about what the tool does, for example from an MCP server.
    annotations: ToolAnnotations | None = None
    # Whether registering the tool activates it. None means True for "direct"
    # and "model-only" tools; other exposures are never activated on
    # registration. A tool with `default_active=False` is activated by naming
    # it in `--tools` or the `defaultTools` setting, or with `set_active_tools()`.
    default_active: bool | None = None
    # Adjust how the loadout is presented to the model while this tool is
    # active: `(loadout: ToolLoadout) -> ToolLoadoutChanges | None`. Called
    # whenever the active tools change. Tools that orchestrate other tools use
    # it, for example to list the callable tools in their own description.
    prepare_loadout: Callable[[ToolLoadout], ToolLoadoutChanges | None] | None = None
    # Per-tool execution mode override ("sequential" | "parallel").
    execution_mode: str | None = None
    # Extra metadata slot mirroring pi's open object shape.
    extra: dict[str, Any] = field(default_factory=dict)


# Outcome of the activity that reached a boundary (`turn_end` / `agent_before_settle`).
type AgentActivityOutcome = Literal["completed", "aborted", "error"]


@dataclass(slots=True, frozen=True)
class BoundaryContextPreview:
    """The model context a boundary would leave behind (pi: `BoundaryContextPreview`).

    Boundary events (`turn_end`, `agent_before_settle`) carry this as
    `event["context"]`, rebuilt after each handler so later handlers see the
    effect of earlier drafts. Session boundary drafts themselves are plain
    camelCase dicts, like pi's objects: `{"type": "custom", "customType", "data"?}`,
    `{"type": "custom_message", "customType", "content", "display", "details"?}`,
    `{"type": "context_edit", "targetId", "replacement"}` and
    `{"type": "compaction", "summary", "firstKeptEntryId", "details"?, "usage"?}`
    (`firstKeptEntryId: None` keeps no preceding entries). Handlers return
    `{"entries"?, "continue"?}`.
    """

    # Projected session entries (`ProjectedSessionEntry`) after the drafts are applied.
    context_entries: list[Any]
    context_messages: list[Any]
    # `context_messages` after `convert_to_llm`.
    llm_messages: list[Any]
    # Queued steering/follow-up and pending custom messages not yet in context.
    pending_messages: list[Any]
    # Whether a continuation would have runnable model context.
    can_continue: bool


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


@dataclass(slots=True, kw_only=True)
class InlineExtension:
    """A named inline extension factory (pi's `InlineExtension` object arm; a
    bare factory is the other arm). Duck-typed objects with these attributes
    work too."""

    # Display name shown as `<inline:name>` in the startup Extensions list and
    # errors. With `builtin`, the extension is named `builtin:name` in errors
    # and diagnostics.
    name: str
    factory: Callable[[Any], Any]
    # Omit this extension from the startup Extensions list.
    hidden: bool = False
    # Leave this extension out when another extension registers a tool,
    # command, or flag with a name it registers during loading, instead of
    # reporting a conflict. The factory still runs, so it should only register
    # tools, commands, flags, and event handlers.
    replaceable: bool = False
    # Supply the code of the `builtin:<name>` extension instead of loading as an
    # inline extension. `builtin:<name>` is an extension resource like a file:
    # it loads by default, `pidrei config` lists it, `-builtin:<name>` in the
    # `extensions` setting and `--no-extensions` disable it, and
    # `-e builtin:<name>` loads it explicitly. It is hidden from the startup
    # Extensions list and loads after project trust is resolved, so it cannot
    # handle `project_trust`.
    builtin: bool = False


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
    # See `InlineExtension.replaceable`.
    replaceable: bool = False
    # event type -> handlers; a single extension may register several per event.
    handlers: dict[str, list[Any]] = field(default_factory=dict)
    tools: dict[str, RegisteredTool] = field(default_factory=dict)
    commands: dict[str, RegisteredCommand] = field(default_factory=dict)
    flags: dict[str, ExtensionFlag] = field(default_factory=dict)
    shortcuts: dict[str, Any] = field(default_factory=dict)
    message_renderers: dict[str, Any] = field(default_factory=dict)
    markdown_transformer: Any = None
    entry_renderers: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, kw_only=True)
class ExtensionVirtualModel:
    """Virtual model registered via `pi.register_virtual_model()`: the fields of
    `VirtualModelDefinition`, with a `route(request, ctx)` that also receives an
    extension context. Async-only."""

    provider: str
    id: str
    name: str
    route: Callable[[Any, Any], Awaitable[Any]]
    thinking_levels: Sequence[str] | None = None
    context_window: int | None = None
    max_tokens: int | None = None
    input: list[Literal["text", "image"]] | None = None


def _not_initialized(*_args: Any) -> Any:
    raise RuntimeError(RUNTIME_NOT_INITIALIZED)


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
        # Virtual model registrations queued during extension loading, processed when the runner binds.
        self.pending_virtual_model_registrations: list[Any] = []
        # Servers registered with `pi.register_mcp_server()`.
        self.mcp_servers = McpServerRegistry()

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
        self.get_settings: Callable[[], Any] = _not_initialized
        self.set_active_tools: Callable[..., None] | None = None
        self.get_commands: Callable[[], list[Any]] = list
        self.set_model: Callable[..., Any] | None = None
        self.get_thinking_level: Callable[[], Any] = lambda: "off"
        self.set_thinking_level: Callable[..., None] | None = None
        # Create an extension context. Raises before the runner binds.
        self.create_context: Callable[[], Any] = _not_initialized

        # Provider registration hooks. Pre-bind they queue, so a registration
        # made while extensions are still loading survives until the model
        # registry exists; bind_core() flushes the queues and replaces these
        # with direct calls, so later registrations need no /reload.
        self.register_provider: Callable[..., None] = self._queue_provider
        self.register_native_provider: Callable[..., None] = self._queue_native_provider
        self.unregister_provider: Callable[..., None] = self._unqueue_provider
        self.register_virtual_model: Callable[..., None] = self._queue_virtual_model
        self.unregister_virtual_model: Callable[..., None] = self._unqueue_virtual_model

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

    def _queue_virtual_model(self, definition: Any, extension_path: str = "<unknown>") -> None:
        self.pending_virtual_model_registrations.append({"definition": definition, "extension_path": extension_path})

    def _unqueue_virtual_model(self, provider: str, model_id: str) -> None:
        self.pending_virtual_model_registrations = [
            entry
            for entry in self.pending_virtual_model_registrations
            if entry["definition"].provider != provider or entry["definition"].id != model_id
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
class ExtensionLoadWarning:
    path: str
    warning: str


@dataclass(slots=True)
class LoadExtensionsResult:
    """pi's LoadExtensionsResult."""

    extensions: list[Extension] = field(default_factory=list)
    errors: list[ExtensionLoadError] = field(default_factory=list)
    warnings: list[ExtensionLoadWarning] = field(default_factory=list)
    runtime: ExtensionRuntime = field(default_factory=ExtensionRuntime)
