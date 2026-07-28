"""Mirror of pi coding-agent src/core/extensions/runner.ts.

Executes extension handlers and owns the hook bus AgentSession emits into.
"""

import copy
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pidrei.core.diagnostics import ResourceDiagnostic

from .types import (
    Extension,
    ExtensionError,
    ExtensionFlag,
    ExtensionRuntime,
    ExtensionShortcut,
    LoadExtensionsResult,
    RegisteredTool,
    ResolvedCommand,
)


# Extension shortcuts compete with canonical keybinding ids from keybindings.json.
# Only editor-global shortcuts are reserved here; picker-specific bindings are not.
RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS = (
    "app.interrupt",
    "app.clear",
    "app.exit",
    "app.suspend",
    "app.thinking.cycle",
    "app.model.cycleForward",
    "app.model.cycleBackward",
    "app.model.select",
    "app.tools.expand",
    "app.thinking.toggle",
    "app.editor.external",
    "app.message.copy",
    "app.message.followUp",
    "tui.input.submit",
    "tui.select.confirm",
    "tui.select.cancel",
    "tui.input.copy",
    "tui.editor.deleteToLineEnd",
)


def _build_builtin_keybindings(resolved_keybindings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    builtin: dict[str, dict[str, Any]] = {}
    for keybinding, keys in resolved_keybindings.items():
        if keys is None:
            continue
        key_list = keys if isinstance(keys, list) else [keys]
        restrict_override = keybinding in RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS
        for key in key_list:
            normalized_key = key.lower()
            # When several actions bind the same key the reserved one wins, so
            # extensions stay blocked regardless of iteration order.
            existing = builtin.get(normalized_key)
            if existing is not None and existing["restrict_override"] and not restrict_override:
                continue
            builtin[normalized_key] = {"keybinding": keybinding, "restrict_override": restrict_override}
    return builtin


_STALE_MESSAGE_DEFAULT = (
    "This extension ctx is stale after session replacement or reload. Do not use a captured pi or command "
    "ctx after ctx.newSession(), ctx.fork(), ctx.switchSession(), or ctx.reload(). For newSession, fork, and "
    "switchSession, move post-replacement work into withSession and use the ctx passed to withSession. For "
    "reload, do not use the old ctx after await ctx.reload()."
)

_SESSION_BEFORE_EVENT_TYPES = (
    "session_before_switch",
    "session_before_fork",
    "session_before_compact",
    "session_before_tree",
)


class _NoOpUIContext:
    """pi's noOpUIContext: UI surface used when no interactive UI is bound.

    The theme system is Phase 4; `theme` is None here (pi returns the default
    theme object).
    """

    async def select(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def confirm(self, *args: Any, **kwargs: Any) -> bool:
        return False

    async def input(self, *args: Any, **kwargs: Any) -> None:
        return None

    def notify(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_terminal_input(self, *args: Any, **kwargs: Any) -> Callable[[], None]:
        return lambda: None

    def set_status(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_working_message(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_working_visible(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_working_indicator(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_hidden_thinking_label(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_widget(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_footer(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_header(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_title(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def custom(self, *args: Any, **kwargs: Any) -> None:
        return None

    def paste_to_editor(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_editor_text(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_editor_text(self) -> str:
        return ""

    async def editor(self, *args: Any, **kwargs: Any) -> None:
        return None

    def add_autocomplete_provider(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_editor_component(self, *args: Any, **kwargs: Any) -> None:
        pass

    def get_editor_component(self) -> None:
        return None

    @property
    def theme(self) -> None:
        return None

    def get_all_themes(self) -> list[Any]:
        return []

    def get_theme(self) -> None:
        return None

    def set_theme(self, _theme: Any) -> dict[str, Any]:
        return {"success": False, "error": "UI not available"}

    def get_tools_expanded(self) -> bool:
        return False

    def set_tools_expanded(self, *args: Any, **kwargs: Any) -> None:
        pass


_NO_OP_UI_CONTEXT = _NoOpUIContext()


@dataclass(slots=True)
class InputEventResult:
    action: str  # "continue" | "handled" | "transform"
    text: str | None = None
    images: list[Any] | None = None


@dataclass(slots=True)
class ResourcesDiscoverPaths:
    skill_paths: list[dict[str, str]]
    prompt_paths: list[dict[str, str]]
    theme_paths: list[dict[str, str]]


class _RunnerContext:
    """pi's createContext() result: values resolve at call time through the
    runner, and every access asserts the runner is not stale."""

    def __init__(self, runner: ExtensionRunner):
        self._runner = runner

    @property
    def ui(self) -> Any:
        self._runner._assert_active()
        return self._runner._ui_context

    @property
    def mode(self) -> str:
        self._runner._assert_active()
        return self._runner._mode

    @property
    def has_ui(self) -> bool:
        self._runner._assert_active()
        return self._runner.has_ui()

    @property
    def cwd(self) -> str:
        self._runner._assert_active()
        return self._runner._cwd

    @property
    def session_manager(self) -> Any:
        self._runner._assert_active()
        return self._runner._session_manager

    @property
    def model_registry(self) -> Any:
        self._runner._assert_active()
        return self._runner._model_registry

    @property
    def model(self) -> Any:
        self._runner._assert_active()
        return self._runner._get_model()

    @property
    def thinking_level(self) -> Any:
        self._runner._assert_active()
        return self._runner._runtime.get_thinking_level()

    def is_idle(self) -> bool:
        self._runner._assert_active()
        return self._runner._is_idle_fn()

    def is_project_trusted(self) -> bool:
        self._runner._assert_active()
        return self._runner._is_project_trusted_fn()

    @property
    def signal(self) -> Any:
        self._runner._assert_active()
        return self._runner._get_signal_fn()

    def abort(self) -> None:
        self._runner._assert_active()
        self._runner._abort_fn()

    def has_pending_messages(self) -> bool:
        self._runner._assert_active()
        return self._runner._has_pending_messages_fn()

    def shutdown(self) -> None:
        self._runner._assert_active()
        self._runner._shutdown_handler()

    def get_context_usage(self) -> Any:
        self._runner._assert_active()
        return self._runner._get_context_usage_fn()

    def compact(self, options: Any = None) -> None:
        self._runner._assert_active()
        self._runner._compact_fn(options)

    def get_system_prompt(self) -> str:
        self._runner._assert_active()
        return self._runner._get_system_prompt_fn()


class _RunnerCommandContext(_RunnerContext):
    def get_system_prompt_options(self) -> Any:
        self._runner._assert_active()
        return self._runner._get_system_prompt_options_fn()

    async def wait_for_idle(self) -> None:
        self._runner._assert_active()
        await self._runner._wait_for_idle_fn()

    async def new_session(self, options: Any = None) -> Any:
        self._runner._assert_active()
        return await self._runner._new_session_handler(options)

    async def fork(self, entry_id: str, options: Any = None) -> Any:
        self._runner._assert_active()
        return await self._runner._fork_handler(entry_id, options)

    async def navigate_tree(self, target_id: str, options: Any = None) -> Any:
        self._runner._assert_active()
        return await self._runner._navigate_tree_handler(target_id, options)

    async def switch_session(self, session_path: str, options: Any = None) -> Any:
        self._runner._assert_active()
        return await self._runner._switch_session_handler(session_path, options)

    async def reload(self) -> None:
        self._runner._assert_active()
        await self._runner._reload_handler()


async def _default_cancelled_result(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {"cancelled": False}


async def _default_async_noop(*_args: Any, **_kwargs: Any) -> None:
    return None


async def emit_session_shutdown_event(extension_runner: ExtensionRunner, event: dict[str, Any]) -> bool:
    """Emit session_shutdown to extensions. Returns True if emitted."""
    if extension_runner.has_handlers("session_shutdown"):
        await extension_runner.emit(event)
        return True
    return False


async def emit_project_trust_event(
    extensions_result: LoadExtensionsResult, event: dict[str, Any], ctx: Any
) -> tuple[Any, list[ExtensionError]]:
    """First project_trust handler that returns yes/no wins; undecided falls through."""
    errors: list[ExtensionError] = []
    for ext in extensions_result.extensions:
        handlers = ext.handlers.get("project_trust")
        if not handlers:
            continue
        for handler in handlers:
            try:
                handler_result = await handler(event, ctx)
                if isinstance(handler_result, dict) and handler_result.get("trusted") == "undecided":
                    continue
                return handler_result, errors
            except Exception as error:
                errors.append(
                    ExtensionError(
                        extension_path=ext.path,
                        event=event.get("type", "project_trust"),
                        error=str(error),
                        stack=traceback.format_exc(),
                    )
                )
    return None, errors


class ExtensionRunner:
    """Executes extension handlers and manages their lifecycle."""

    def __init__(
        self,
        extensions: list[Extension],
        runtime: ExtensionRuntime,
        cwd: str,
        session_manager: Any,
        model_registry: Any,
    ):
        self._extensions = extensions
        self._runtime = runtime
        self._ui_context: Any = _NO_OP_UI_CONTEXT
        self._mode = "print"
        self._cwd = cwd
        self._session_manager = session_manager
        self._model_registry = model_registry
        self._error_listeners: list[Callable[[ExtensionError], None]] = []
        self._shortcut_diagnostics: list[ResourceDiagnostic] = []
        self._stale_message: str | None = None

        self._get_model: Callable[[], Any] = lambda: None
        self._is_idle_fn: Callable[[], bool] = lambda: True
        self._is_project_trusted_fn: Callable[[], bool] = lambda: True
        self._get_signal_fn: Callable[[], Any] = lambda: None
        self._abort_fn: Callable[[], None] = lambda: None
        self._has_pending_messages_fn: Callable[[], bool] = lambda: False
        self._get_context_usage_fn: Callable[[], Any] = lambda: None
        self._compact_fn: Callable[..., None] = lambda options=None: None
        self._get_system_prompt_fn: Callable[[], str] = lambda: ""
        self._get_system_prompt_options_fn: Callable[[], Any] = lambda: {"cwd": self._cwd}
        self._shutdown_handler: Callable[[], None] = lambda: None

        self._wait_for_idle_fn = _default_async_noop
        self._new_session_handler = _default_cancelled_result
        self._fork_handler = _default_cancelled_result
        self._navigate_tree_handler = _default_cancelled_result
        self._switch_session_handler = _default_cancelled_result
        self._reload_handler = _default_async_noop

    # -- binding ---------------------------------------------------------------

    def bind_core(
        self, actions: dict[str, Any], context_actions: dict[str, Any], provider_actions: dict | None = None
    ) -> None:
        # Copy actions into the shared runtime (all extension APIs reference this)
        runtime = self._runtime
        runtime.send_message = actions["send_message"]
        runtime.send_user_message = actions["send_user_message"]
        runtime.append_entry = actions["append_entry"]
        runtime.set_session_name = actions["set_session_name"]
        runtime.get_session_name = actions["get_session_name"]
        runtime.set_label = actions["set_label"]
        runtime.get_active_tools = actions["get_active_tools"]
        runtime.get_all_tools = actions["get_all_tools"]
        runtime.set_active_tools = actions["set_active_tools"]
        runtime.refresh_tools = actions["refresh_tools"]
        runtime.get_commands = actions["get_commands"]
        runtime.set_model = actions["set_model"]
        runtime.get_thinking_level = actions["get_thinking_level"]
        runtime.set_thinking_level = actions["set_thinking_level"]

        # Context actions (required)
        self._get_model = context_actions["get_model"]
        self._is_idle_fn = context_actions["is_idle"]
        self._is_project_trusted_fn = context_actions["is_project_trusted"]
        self._get_signal_fn = context_actions["get_signal"]
        self._abort_fn = context_actions["abort"]
        self._has_pending_messages_fn = context_actions["has_pending_messages"]
        self._shutdown_handler = context_actions["shutdown"]
        self._get_context_usage_fn = context_actions["get_context_usage"]
        self._compact_fn = context_actions["compact"]
        self._get_system_prompt_fn = context_actions["get_system_prompt"]
        get_options = context_actions.get("get_system_prompt_options")
        self._get_system_prompt_options_fn = get_options if get_options is not None else lambda: {"cwd": self._cwd}

        provider_actions = provider_actions or {}
        register_provider = provider_actions.get("register_provider")
        register_native_provider = provider_actions.get("register_native_provider")
        unregister_provider = provider_actions.get("unregister_provider")

        # Flush provider registrations queued during extension loading
        for registration in runtime.pending_provider_registrations:
            try:
                if register_provider is not None:
                    register_provider(registration["name"], registration["config"])
                else:
                    self._model_registry.register_provider(registration["name"], registration["config"])
            except Exception as error:
                self.emit_error(
                    ExtensionError(
                        extension_path=registration.get("extension_path", "<unknown>"),
                        event="register_provider",
                        error=str(error),
                        stack=traceback.format_exc(),
                    )
                )
        runtime.pending_provider_registrations = []
        for registration in runtime.pending_native_provider_registrations:
            try:
                if register_native_provider is not None:
                    register_native_provider(registration["provider"])
                else:
                    self._model_registry.register_provider(registration["provider"])
            except Exception as error:
                self.emit_error(
                    ExtensionError(
                        extension_path=registration.get("extension_path", "<unknown>"),
                        event="register_provider",
                        error=str(error),
                        stack=traceback.format_exc(),
                    )
                )
        runtime.pending_native_provider_registrations = []

        # From this point on, provider registration/unregistration takes effect
        # immediately without requiring a /reload.
        def runtime_register_provider(name: str, config: Any, _extension_path: str = "<unknown>") -> None:
            if register_provider is not None:
                register_provider(name, config)
                return
            self._model_registry.register_provider(name, config)

        def runtime_register_native_provider(provider: Any, _extension_path: str = "<unknown>") -> None:
            if register_native_provider is not None:
                register_native_provider(provider)
                return
            self._model_registry.register_provider(provider)

        def runtime_unregister_provider(name: str, _extension_path: str = "<unknown>") -> None:
            if unregister_provider is not None:
                unregister_provider(name)
                return
            self._model_registry.unregister_provider(name)

        runtime.register_provider = runtime_register_provider
        runtime.register_native_provider = runtime_register_native_provider
        runtime.unregister_provider = runtime_unregister_provider

    def bind_command_context(self, actions: dict[str, Any] | None = None) -> None:
        if actions:
            self._wait_for_idle_fn = actions["wait_for_idle"]
            self._new_session_handler = actions["new_session"]
            self._fork_handler = actions["fork"]
            self._navigate_tree_handler = actions["navigate_tree"]
            self._switch_session_handler = actions["switch_session"]
            self._reload_handler = actions["reload"]
            return

        self._wait_for_idle_fn = _default_async_noop
        self._new_session_handler = _default_cancelled_result
        self._fork_handler = _default_cancelled_result
        self._navigate_tree_handler = _default_cancelled_result
        self._switch_session_handler = _default_cancelled_result
        self._reload_handler = _default_async_noop

    def set_ui_context(self, ui_context: Any = None, mode: str = "print") -> None:
        self._ui_context = ui_context if ui_context is not None else _NO_OP_UI_CONTEXT
        self._mode = mode

    def get_ui_context(self) -> Any:
        return self._ui_context

    def has_ui(self) -> bool:
        return self._ui_context is not _NO_OP_UI_CONTEXT

    # -- registry access ---------------------------------------------------------

    def get_extension_paths(self) -> list[str]:
        return [ext.path for ext in self._extensions]

    def get_all_registered_tools(self) -> list[RegisteredTool]:
        """All registered tools from all extensions (first registration per name wins)."""
        tools_by_name: dict[str, RegisteredTool] = {}
        for ext in self._extensions:
            for tool in ext.tools.values():
                if tool.definition.name not in tools_by_name:
                    tools_by_name[tool.definition.name] = tool
        return list(tools_by_name.values())

    def get_tool_definition(self, tool_name: str) -> Any:
        for ext in self._extensions:
            tool = ext.tools.get(tool_name)
            if tool is not None:
                return tool.definition
        return None

    def get_flags(self) -> dict[str, ExtensionFlag]:
        all_flags: dict[str, ExtensionFlag] = {}
        for ext in self._extensions:
            for name, flag in ext.flags.items():
                if name not in all_flags:
                    all_flags[name] = flag
        return all_flags

    def set_flag_value(self, name: str, value: Any) -> None:
        self._runtime.flag_values[name] = value

    def get_flag_values(self) -> dict[str, Any]:
        return dict(self._runtime.flag_values)

    def get_shortcuts(self, resolved_keybindings: dict[str, Any]) -> dict[str, ExtensionShortcut]:
        """Extension shortcuts, minus the ones a reserved built-in owns.

        Reserved keys are refused outright; every other collision is allowed
        with a warning and last-registered wins.
        """
        self._shortcut_diagnostics = []
        builtin_keybindings = _build_builtin_keybindings(resolved_keybindings)
        extension_shortcuts: dict[str, ExtensionShortcut] = {}

        def add_diagnostic(message: str, extension_path: str) -> None:
            self._shortcut_diagnostics.append(ResourceDiagnostic(type="warning", message=message, path=extension_path))
            if not self.has_ui():
                print(message, file=sys.stderr)

        for ext in self._extensions:
            for key, shortcut in ext.shortcuts.items():
                normalized_key = key.lower()

                builtin = builtin_keybindings.get(normalized_key)
                if builtin is not None and builtin["restrict_override"]:
                    add_diagnostic(
                        f"Extension shortcut '{key}' from {shortcut.extension_path} conflicts with "
                        "built-in shortcut. Skipping.",
                        shortcut.extension_path,
                    )
                    continue

                if builtin is not None and not builtin["restrict_override"]:
                    add_diagnostic(
                        f"Extension shortcut conflict: '{key}' is built-in shortcut for "
                        f"{builtin['keybinding']} and {shortcut.extension_path}. Using {shortcut.extension_path}.",
                        shortcut.extension_path,
                    )

                existing = extension_shortcuts.get(normalized_key)
                if existing is not None:
                    add_diagnostic(
                        f"Extension shortcut conflict: '{key}' registered by both {existing.extension_path} "
                        f"and {shortcut.extension_path}. Using {shortcut.extension_path}.",
                        shortcut.extension_path,
                    )
                extension_shortcuts[normalized_key] = shortcut

        return extension_shortcuts

    def get_shortcut_diagnostics(self) -> list[ResourceDiagnostic]:
        return self._shortcut_diagnostics

    def get_model_registry(self) -> Any:
        return self._model_registry

    def _resolve_registered_commands(self) -> list[ResolvedCommand]:
        commands: list[Any] = []
        counts: dict[str, int] = {}

        for ext in self._extensions:
            for command in ext.commands.values():
                commands.append(command)
                counts[command.name] = counts.get(command.name, 0) + 1

        seen: dict[str, int] = {}
        taken_invocation_names: set[str] = set()
        resolved: list[ResolvedCommand] = []

        for command in commands:
            occurrence = seen.get(command.name, 0) + 1
            seen[command.name] = occurrence

            invocation_name = f"{command.name}:{occurrence}" if counts.get(command.name, 0) > 1 else command.name

            if invocation_name in taken_invocation_names:
                suffix = occurrence
                while True:
                    suffix += 1
                    invocation_name = f"{command.name}:{suffix}"
                    if invocation_name not in taken_invocation_names:
                        break

            taken_invocation_names.add(invocation_name)
            resolved.append(
                ResolvedCommand(
                    name=command.name,
                    handler=command.handler,
                    description=command.description,
                    source_info=command.source_info,
                    invocation_name=invocation_name,
                )
            )
        return resolved

    def get_registered_commands(self) -> list[ResolvedCommand]:
        return self._resolve_registered_commands()

    def get_command(self, name: str) -> ResolvedCommand | None:
        return next(
            (command for command in self._resolve_registered_commands() if command.invocation_name == name), None
        )

    def shutdown(self) -> None:
        """Request a graceful shutdown. Called by extension tools and handlers."""
        self._shutdown_handler()

    def get_active_tools(self) -> list[str]:
        self._assert_active()
        return self._runtime.get_active_tools()

    # -- staleness / errors --------------------------------------------------------

    def invalidate(self, message: str = _STALE_MESSAGE_DEFAULT) -> None:
        if not self._stale_message:
            self._stale_message = message
            self._runtime.invalidate(message)

    def _assert_active(self) -> None:
        if self._stale_message:
            raise Exception(self._stale_message)

    def on_error(self, listener: Callable[[ExtensionError], None]) -> Callable[[], None]:
        self._error_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._error_listeners:
                self._error_listeners.remove(listener)

        return unsubscribe

    def emit_error(self, error: ExtensionError) -> None:
        for listener in list(self._error_listeners):
            listener(error)

    def has_handlers(self, event_type: str) -> bool:
        return any(ext.handlers.get(event_type) for ext in self._extensions)

    def get_message_renderer(self, custom_type: str) -> Any:
        for ext in self._extensions:
            renderer = ext.message_renderers.get(custom_type)
            if renderer is not None:
                return renderer
        return None

    def get_entry_renderer(self, custom_type: str) -> Any:
        for ext in self._extensions:
            renderer = ext.entry_renderers.get(custom_type)
            if renderer is not None:
                return renderer
        return None

    # -- contexts --------------------------------------------------------------

    def create_context(self) -> _RunnerContext:
        return _RunnerContext(self)

    def create_command_context(self) -> _RunnerCommandContext:
        return _RunnerCommandContext(self)

    # -- emits -----------------------------------------------------------------

    async def emit(self, event: dict[str, Any]) -> Any:
        ctx = self.create_context()
        result: Any = None
        is_session_before = event.get("type") in _SESSION_BEFORE_EVENT_TYPES

        for ext in self._extensions:
            handlers = ext.handlers.get(event.get("type"))
            if not handlers:
                continue

            for handler in handlers:
                try:
                    handler_result = await handler(event, ctx)

                    if is_session_before and handler_result:
                        result = handler_result
                        if isinstance(result, dict) and result.get("cancel"):
                            return result
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event=event.get("type"),
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        return result

    async def emit_message_end(self, event: dict[str, Any]) -> Any:
        ctx = self.create_context()
        current_message = event.get("message")
        modified = False

        for ext in self._extensions:
            handlers = ext.handlers.get("message_end")
            if not handlers:
                continue

            for handler in handlers:
                try:
                    current_event = {**event, "message": current_message}
                    handler_result = await handler(current_event, ctx)
                    replacement = handler_result.get("message") if isinstance(handler_result, dict) else None
                    if replacement is None:
                        continue

                    if getattr(replacement, "role", None) != getattr(current_message, "role", None):
                        self.emit_error(
                            ExtensionError(
                                extension_path=ext.path,
                                event="message_end",
                                error="message_end handlers must return a message with the same role",
                            )
                        )
                        continue

                    current_message = replacement
                    modified = True
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="message_end",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        return current_message if modified else None

    async def emit_tool_result(self, event: dict[str, Any]) -> dict[str, Any] | None:
        ctx = self.create_context()
        current_event = dict(event)
        modified = False

        for ext in self._extensions:
            handlers = ext.handlers.get("tool_result")
            if not handlers:
                continue

            for handler in handlers:
                try:
                    handler_result = await handler(current_event, ctx)
                    if not isinstance(handler_result, dict):
                        continue

                    for key in ("content", "details", "isError", "usage"):
                        if key in handler_result:
                            current_event[key] = handler_result[key]
                            modified = True
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="tool_result",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        if not modified:
            return None

        return {
            "content": current_event.get("content"),
            "details": current_event.get("details"),
            "isError": current_event.get("isError"),
            "usage": current_event.get("usage"),
        }

    async def emit_tool_call(self, event: dict[str, Any]) -> dict[str, Any] | None:
        ctx = self.create_context()
        result: dict[str, Any] | None = None

        for ext in self._extensions:
            handlers = ext.handlers.get("tool_call")
            if not handlers:
                continue

            for handler in handlers:
                # Intentionally no try/except: tool_call handler errors block execution
                # (AgentSession wraps them; mirrors pi).
                handler_result = await handler(event, ctx)

                if handler_result:
                    result = handler_result
                    if result.get("block"):
                        return result

        return result

    async def emit_user_bash(self, event: dict[str, Any]) -> Any:
        ctx = self.create_context()

        for ext in self._extensions:
            handlers = ext.handlers.get("user_bash")
            if not handlers:
                continue

            for handler in handlers:
                try:
                    handler_result = await handler(event, ctx)
                    if handler_result:
                        return handler_result
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="user_bash",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        return None

    async def emit_context(self, messages: list[Any]) -> list[Any]:

        ctx = self.create_context()
        # pi structuredClones here; only pay for the deep copy when a handler exists.
        if not self.has_handlers("context"):
            return messages
        current_messages = copy.deepcopy(messages)

        for ext in self._extensions:
            handlers = ext.handlers.get("context")
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = {"type": "context", "messages": current_messages}
                    handler_result = await handler(event, ctx)
                    if isinstance(handler_result, dict) and handler_result.get("messages"):
                        current_messages = handler_result["messages"]
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="context",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        return current_messages

    async def emit_before_provider_request(self, payload: Any) -> Any:
        ctx = self.create_context()
        current_payload = payload

        for ext in self._extensions:
            handlers = ext.handlers.get("before_provider_request")
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = {"type": "before_provider_request", "payload": current_payload}
                    handler_result = await handler(event, ctx)
                    if handler_result is not None:
                        current_payload = handler_result
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="before_provider_request",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        return current_payload

    async def emit_before_provider_headers(self, headers: dict[str, Any]) -> dict[str, Any]:
        ctx = self.create_context()

        for ext in self._extensions:
            handlers = ext.handlers.get("before_provider_headers")
            if not handlers:
                continue

            for handler in handlers:
                try:
                    # Handlers mutate `headers` in place; the return value is ignored.
                    event = {"type": "before_provider_headers", "headers": headers}
                    await handler(event, ctx)
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="before_provider_headers",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        return headers

    async def emit_before_agent_start(
        self,
        prompt: str,
        images: list[Any] | None,
        system_prompt: str,
        system_prompt_options: Any,
    ) -> dict[str, Any] | None:
        current_system_prompt = system_prompt
        ctx = self.create_context()
        ctx.get_system_prompt = lambda: self._assert_active() or current_system_prompt
        messages: list[Any] = []
        system_prompt_modified = False

        for ext in self._extensions:
            handlers = ext.handlers.get("before_agent_start")
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = {
                        "type": "before_agent_start",
                        "prompt": prompt,
                        "images": images,
                        "systemPrompt": current_system_prompt,
                        "systemPromptOptions": system_prompt_options,
                    }
                    handler_result = await handler(event, ctx)

                    if isinstance(handler_result, dict):
                        if handler_result.get("message") is not None:
                            messages.append(handler_result["message"])
                        if "systemPrompt" in handler_result and handler_result["systemPrompt"] is not None:
                            current_system_prompt = handler_result["systemPrompt"]
                            system_prompt_modified = True
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="before_agent_start",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        if messages or system_prompt_modified:
            return {
                "messages": messages if messages else None,
                "systemPrompt": current_system_prompt if system_prompt_modified else None,
            }

        return None

    async def emit_resources_discover(self, cwd: str, reason: str) -> ResourcesDiscoverPaths:
        ctx = self.create_context()
        skill_paths: list[dict[str, str]] = []
        prompt_paths: list[dict[str, str]] = []
        theme_paths: list[dict[str, str]] = []

        for ext in self._extensions:
            handlers = ext.handlers.get("resources_discover")
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = {"type": "resources_discover", "cwd": cwd, "reason": reason}
                    handler_result = await handler(event, ctx)
                    if not isinstance(handler_result, dict):
                        continue
                    for key, bucket in (
                        ("skillPaths", skill_paths),
                        ("promptPaths", prompt_paths),
                        ("themePaths", theme_paths),
                    ):
                        paths = handler_result.get(key)
                        if paths:
                            bucket.extend({"path": path, "extension_path": ext.path} for path in paths)
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="resources_discover",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        return ResourcesDiscoverPaths(skill_paths=skill_paths, prompt_paths=prompt_paths, theme_paths=theme_paths)

    async def emit_input(
        self,
        text: str,
        images: list[Any] | None,
        source: str,
        streaming_behavior: str | None = None,
    ) -> InputEventResult:
        """Emit input event. Transforms chain, "handled" short-circuits."""
        ctx = self.create_context()
        current_text = text
        current_images = images

        for ext in self._extensions:
            for handler in ext.handlers.get("input", ()):
                try:
                    event = {
                        "type": "input",
                        "text": current_text,
                        "images": current_images,
                        "source": source,
                        "streamingBehavior": streaming_behavior,
                    }
                    result = await handler(event, ctx)
                    if isinstance(result, dict):
                        if result.get("action") == "handled":
                            return InputEventResult(action="handled")
                        if result.get("action") == "transform":
                            current_text = result.get("text")
                            current_images = (
                                result.get("images") if result.get("images") is not None else current_images
                            )
                except Exception as error:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="input",
                            error=str(error),
                            stack=traceback.format_exc(),
                        )
                    )

        if current_text != text or current_images is not images:
            return InputEventResult(action="transform", text=current_text, images=current_images)
        return InputEventResult(action="continue")
