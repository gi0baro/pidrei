"""Mirror of pi coding-agent src/modes/rpc/rpc-mode.ts.

RPC mode: Headless operation with JSON stdin/stdout protocol.

Used for embedding the agent in other applications. Receives commands as
JSON on stdin, outputs events and responses as JSON on stdout.

Protocol:
- Commands: JSON objects with `type` field, optional `id` for correlation
- Responses: JSON objects with `type: "response"`, `command`, `success`,
  and optional `data`/`error`
- Events: AgentSessionEvent objects streamed as they occur
- Extension UI: Extension UI requests are emitted, client responds with
  extension_ui_response
"""

import json
import signal
import sys
import uuid
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import ImageContent

from ...core.agent_session import ExtensionBindings, PromptOptions
from ...core.json_wire import to_wire
from ...core.output_guard import (
    flush_raw_stdout,
    take_over_stdout,
    wait_for_raw_stdout_backpressure,
    write_raw_stdout,
)
from ...core.session_manager import SessionManager
from ...utils.fd_io import FdReader, hard_exit
from ...utils.shell import kill_tracked_detached_children
from .jsonl import JsonlLineDecoder, serialize_json_line
from .rpc_types import RpcSessionState, RpcSlashCommand


_STDIN_READ_SIZE = 65536


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop None values from a protocol dict (pi's undefined fields, which
    JSON.stringify omits)."""
    return {key: value for key, value in payload.items() if value is not None}


class _RpcExtensionUIContext:
    """Extension UI context that uses the RPC protocol.

    Methods that need TUI access (working indicator, widgets with component
    factories, footers, themes) are no-ops, exactly like pi's RPC mode.
    """

    def __init__(self, output, create_dialog_promise, pending_extension_requests):
        self._output = output
        self._create_dialog_promise = create_dialog_promise
        self._pending_extension_requests = pending_extension_requests

    async def select(self, title: str, options: list[str], opts: Any = None) -> str | None:
        def parse(response: dict[str, Any]) -> str | None:
            if response.get("cancelled"):
                return None
            return response.get("value")

        timeout = getattr(opts, "timeout", None) if opts is not None else None
        return await self._create_dialog_promise(
            opts, None, {"method": "select", "title": title, "options": options, "timeout": timeout}, parse
        )

    async def confirm(self, title: str, message: str, opts: Any = None) -> bool:
        def parse(response: dict[str, Any]) -> bool:
            if response.get("cancelled"):
                return False
            return bool(response.get("confirmed", False))

        timeout = getattr(opts, "timeout", None) if opts is not None else None
        return await self._create_dialog_promise(
            opts, False, {"method": "confirm", "title": title, "message": message, "timeout": timeout}, parse
        )

    async def input(self, title: str, placeholder: str | None = None, opts: Any = None) -> str | None:
        def parse(response: dict[str, Any]) -> str | None:
            if response.get("cancelled"):
                return None
            return response.get("value")

        timeout = getattr(opts, "timeout", None) if opts is not None else None
        return await self._create_dialog_promise(
            opts, None, {"method": "input", "title": title, "placeholder": placeholder, "timeout": timeout}, parse
        )

    def notify(self, message: str, type: str | None = None) -> None:
        # Fire and forget - no response needed
        self._output(
            _compact(
                {
                    "type": "extension_ui_request",
                    "id": str(uuid.uuid4()),
                    "method": "notify",
                    "message": message,
                    "notifyType": type,
                }
            )
        )

    def on_terminal_input(self, *_args, **_kwargs):
        # Raw terminal input not supported in RPC mode
        return lambda: None

    def set_status(self, key: str, text: str | None) -> None:
        # Fire and forget - no response needed
        self._output(
            _compact(
                {
                    "type": "extension_ui_request",
                    "id": str(uuid.uuid4()),
                    "method": "setStatus",
                    "statusKey": key,
                    "statusText": text,
                }
            )
        )

    def set_working_message(self, message: str | None = None) -> None:
        """Working message not supported in RPC mode - requires TUI loader access."""

    def set_working_visible(self, visible: bool) -> None:
        """Working visibility not supported in RPC mode - requires TUI loader access."""

    def set_working_indicator(self, options: Any = None) -> None:
        """Working indicator customization not supported in RPC mode - requires TUI loader access."""

    def set_hidden_thinking_label(self, label: str | None = None) -> None:
        """Hidden thinking label not supported in RPC mode - requires TUI message rendering access."""

    def set_widget(self, key: str, content: Any, options: Any = None) -> None:
        # Only support string lists in RPC mode - component factories are ignored
        if content is None or isinstance(content, list):
            self._output(
                _compact(
                    {
                        "type": "extension_ui_request",
                        "id": str(uuid.uuid4()),
                        "method": "setWidget",
                        "widgetKey": key,
                        "widgetLines": content,
                        "widgetPlacement": getattr(options, "placement", None) if options is not None else None,
                    }
                )
            )

    def set_footer(self, factory: Any) -> None:
        """Custom footer not supported in RPC mode - requires TUI access."""

    def set_header(self, factory: Any) -> None:
        """Custom header not supported in RPC mode - requires TUI access."""

    def set_title(self, title: str) -> None:
        # Fire and forget - host can implement terminal title control
        self._output({"type": "extension_ui_request", "id": str(uuid.uuid4()), "method": "setTitle", "title": title})

    async def custom(self, *_args, **_kwargs):
        # Custom UI not supported in RPC mode
        return None

    async def paste_to_editor(self, text: str) -> None:
        # Paste handling not supported in RPC mode - falls back to set_editor_text
        self.set_editor_text(text)

    def set_editor_text(self, text: str) -> None:
        # Fire and forget - host can implement editor control
        self._output(
            {"type": "extension_ui_request", "id": str(uuid.uuid4()), "method": "set_editor_text", "text": text}
        )

    def get_editor_text(self) -> str:
        # Synchronous method can't wait for RPC response
        # Host should track editor state locally if needed
        return ""

    async def editor(self, title: str, prefill: str | None = None) -> str | None:
        request_id = str(uuid.uuid4())
        event = tonio.Event()
        slot: dict[str, Any] = {}

        def resolve(response: dict[str, Any]) -> None:
            slot["response"] = response
            event.set()

        self._pending_extension_requests[request_id] = resolve
        self._output(
            _compact(
                {
                    "type": "extension_ui_request",
                    "id": request_id,
                    "method": "editor",
                    "title": title,
                    "prefill": prefill,
                }
            )
        )
        await event.wait()
        response = slot.get("response") or {}
        if response.get("cancelled"):
            return None
        return response.get("value")

    def add_autocomplete_provider(self, *_args, **_kwargs) -> None:
        """Autocomplete provider composition is not supported in RPC mode."""

    def set_editor_component(self, *_args, **_kwargs) -> None:
        """Custom editor components not supported in RPC mode."""

    def get_editor_component(self) -> Any:
        # Custom editor components not supported in RPC mode
        return None

    @property
    def theme(self) -> Any:
        # The theme system lands with the Phase 4 TUI slice.
        return None

    async def get_all_themes(self) -> list[Any]:
        return []

    async def get_theme(self, name: str) -> Any:
        return None

    async def set_theme(self, theme: Any) -> dict[str, Any]:
        # Theme switching not supported in RPC mode
        return {"success": False, "error": "Theme switching not supported in RPC mode"}

    def get_tools_expanded(self) -> bool:
        # Tool expansion not supported in RPC mode - no TUI
        return False

    def set_tools_expanded(self, expanded: bool) -> None:
        """Tool expansion not supported in RPC mode - no TUI."""


async def run_rpc_mode(runtime_host) -> None:  # noqa: C901
    """Run in RPC mode.

    Listens for JSON commands on stdin, outputs events and responses on
    stdout. Never returns (the process exits on shutdown).
    """
    take_over_stdout()
    session = runtime_host.session
    unsubscribe: Any = None
    unsubscribe_backpressure: Any = None

    def output(obj: Any) -> None:
        write_raw_stdout(serialize_json_line(to_wire(obj)))

    def success(id: str | None, command: str, data: Any = "__omit__") -> dict[str, Any]:
        response: dict[str, Any] = {}
        if id is not None:
            response["id"] = id
        response.update({"type": "response", "command": command, "success": True})
        if data != "__omit__":
            response["data"] = data
        return response

    def error(id: str | None, command: str, message: str) -> dict[str, Any]:
        response: dict[str, Any] = {}
        if id is not None:
            response["id"] = id
        response.update({"type": "response", "command": command, "success": False, "error": message})
        return response

    # Pending extension UI requests waiting for response
    pending_extension_requests: dict[str, Any] = {}

    # Shutdown request flag
    shutdown_requested = False
    shutting_down = False
    signal_cleanup_handlers: list[Any] = []

    async def create_dialog_promise(opts: Any, default_value: Any, request: dict[str, Any], parse_response) -> Any:
        """Helper for dialog methods with cancel/timeout support."""
        cancel = getattr(opts, "cancel", None) if opts is not None else None
        timeout = getattr(opts, "timeout", None) if opts is not None else None
        if cancel is not None and cancel.cancelled:
            return default_value

        request_id = str(uuid.uuid4())
        event = tonio.Event()
        slot: dict[str, Any] = {}

        def resolve(response: dict[str, Any]) -> None:
            slot["response"] = response
            event.set()

        pending_extension_requests[request_id] = resolve
        unsubscribe_cancel = None
        if cancel is not None:
            unsubscribe_cancel = cancel.on_cancel(lambda _reason: event.set())

        output(_compact({"type": "extension_ui_request", "id": request_id, **request}))
        try:
            if timeout:
                await event.wait(timeout / 1000)
            else:
                await event.wait()
        finally:
            pending_extension_requests.pop(request_id, None)
            if unsubscribe_cancel is not None:
                unsubscribe_cancel()

        response = slot.get("response")
        if response is None:
            return default_value
        return parse_response(response)

    def create_extension_ui_context() -> _RpcExtensionUIContext:
        return _RpcExtensionUIContext(output, create_dialog_promise, pending_extension_requests)

    async def rebind_from_runtime(_session=None) -> None:
        await rebind_session()

    runtime_host.set_rebind_session(rebind_from_runtime)

    async def rebind_session() -> None:
        nonlocal session, unsubscribe, unsubscribe_backpressure
        session = runtime_host.session

        async def fork(entry_id, fork_options=None):
            result = await runtime_host.fork(entry_id, **(fork_options or {}))
            return {"cancelled": result["cancelled"]}

        async def navigate_tree(target_id, navigate_options=None):
            navigate_options = navigate_options or {}
            result = await session.navigate_tree(
                target_id,
                {
                    "summarize": navigate_options.get("summarize"),
                    "custom_instructions": navigate_options.get("custom_instructions"),
                    "replace_instructions": navigate_options.get("replace_instructions"),
                    "label": navigate_options.get("label"),
                },
            )
            return {"cancelled": result.cancelled}

        async def switch_session(session_path, switch_options=None):
            return await runtime_host.switch_session(session_path, **(switch_options or {}))

        async def new_session(new_session_options=None):
            return await runtime_host.new_session(**(new_session_options or {}))

        async def reload():
            await session.reload()

        def shutdown_handler() -> None:
            nonlocal shutdown_requested
            shutdown_requested = True

        def on_error(err) -> None:
            output(
                {
                    "type": "extension_error",
                    "extensionPath": err.extension_path,
                    "event": err.event,
                    "error": err.error,
                }
            )

        await session.bind_extensions(
            ExtensionBindings(
                ui_context=create_extension_ui_context(),
                mode="rpc",
                command_context_actions={
                    "wait_for_idle": session.wait_for_idle,
                    "new_session": new_session,
                    "fork": fork,
                    "navigate_tree": navigate_tree,
                    "switch_session": switch_session,
                    "reload": reload,
                },
                shutdown_handler=shutdown_handler,
                on_error=on_error,
            )
        )

        if unsubscribe is not None:
            unsubscribe()
        if unsubscribe_backpressure is not None:
            unsubscribe_backpressure()

        def on_event(event) -> None:
            output(event)
            if getattr(event, "type", None) == "agent_settled":
                tonio.spawn.without_tracking(check_shutdown_requested())

        async def on_agent_event(*_args) -> None:
            await wait_for_raw_stdout_backpressure()

        unsubscribe = session.subscribe(on_event)
        unsubscribe_backpressure = session.agent.subscribe(on_agent_event)

    def register_signal_handlers() -> None:
        # Same signal-handler note as print mode: the sync handler cannot
        # await runtime disposal, so it kills tracked children and exits.
        def make_handler(exit_status: int):
            def handler(_signum, _frame) -> None:
                kill_tracked_detached_children()
                hard_exit(exit_status)

            return handler

        try:
            previous_term = signal.signal(signal.SIGTERM, make_handler(143))
            signal_cleanup_handlers.append(lambda: signal.signal(signal.SIGTERM, previous_term))
            previous_hup = signal.signal(signal.SIGHUP, make_handler(129))
            signal_cleanup_handlers.append(lambda: signal.signal(signal.SIGHUP, previous_hup))
        except ValueError:
            pass

    await rebind_session()
    register_signal_handlers()

    # Handle a single command
    async def handle_command(command: dict[str, Any]) -> dict[str, Any] | None:  # noqa: C901
        id = command.get("id")
        command_type = command.get("type")

        match command_type:
            # =================================================================
            # Prompting
            # =================================================================

            case "prompt":
                # Start prompt handling immediately, but emit the authoritative
                # response only after prompt preflight succeeds. Queued and
                # immediately handled prompts also count as success.
                preflight_succeeded = False

                def preflight_result(did_succeed: bool) -> None:
                    nonlocal preflight_succeeded
                    if did_succeed:
                        preflight_succeeded = True
                        output(success(id, "prompt"))

                async def run_prompt() -> None:
                    try:
                        await session.prompt(
                            command.get("message"),
                            PromptOptions(
                                images=_parse_images(command.get("images")),
                                streaming_behavior=command.get("streamingBehavior"),
                                source="rpc",
                                preflight_result=preflight_result,
                            ),
                        )
                    except Exception as prompt_error:
                        if not preflight_succeeded:
                            output(error(id, "prompt", str(prompt_error)))

                tonio.spawn.without_tracking(run_prompt())
                return None

            case "steer":
                await session.steer(command.get("message"), _parse_images(command.get("images")))
                return success(id, "steer")

            case "follow_up":
                await session.follow_up(command.get("message"), _parse_images(command.get("images")))
                return success(id, "follow_up")

            case "abort":
                await session.abort()
                return success(id, "abort")

            case "new_session":
                kwargs = {}
                if command.get("parentSession"):
                    kwargs["parent_session"] = command["parentSession"]
                result = await runtime_host.new_session(**kwargs)
                if not result["cancelled"]:
                    await rebind_session()
                return success(id, "new_session", result)

            # =================================================================
            # State
            # =================================================================

            case "get_state":
                state = RpcSessionState(
                    model=session.model,
                    thinking_level=session.thinking_level,
                    is_streaming=session.is_streaming,
                    is_compacting=session.is_compacting,
                    steering_mode=session.steering_mode,
                    follow_up_mode=session.follow_up_mode,
                    session_file=session.session_file,
                    session_id=session.session_id,
                    session_name=session.session_name,
                    auto_compaction_enabled=session.auto_compaction_enabled,
                    message_count=len(session.messages),
                    pending_message_count=session.pending_message_count,
                )
                return success(id, "get_state", state)

            # =================================================================
            # Model
            # =================================================================

            case "set_model":
                models = await session.model_runtime.get_available()
                model = next(
                    (m for m in models if m.provider == command.get("provider") and m.id == command.get("modelId")),
                    None,
                )
                if model is None:
                    return error(
                        id, "set_model", f"Model not found: {command.get('provider')}/{command.get('modelId')}"
                    )
                await session.set_model(model)
                return success(id, "set_model", model)

            case "cycle_model":
                result = await session.cycle_model()
                if result is None:
                    return success(id, "cycle_model", None)
                return success(id, "cycle_model", result)

            case "get_available_models":
                models = await session.model_runtime.get_available()
                return success(id, "get_available_models", {"models": models})

            # =================================================================
            # Thinking
            # =================================================================

            case "set_thinking_level":
                await session.set_thinking_level(command.get("level"))
                return success(id, "set_thinking_level")

            case "cycle_thinking_level":
                level = await session.cycle_thinking_level()
                if level is None:
                    return success(id, "cycle_thinking_level", None)
                return success(id, "cycle_thinking_level", {"level": level})

            case "get_available_thinking_levels":
                levels = session.get_available_thinking_levels()
                return success(id, "get_available_thinking_levels", {"levels": levels})

            # =================================================================
            # Queue Modes
            # =================================================================

            case "set_steering_mode":
                session.set_steering_mode(command.get("mode"))
                return success(id, "set_steering_mode")

            case "set_follow_up_mode":
                session.set_follow_up_mode(command.get("mode"))
                return success(id, "set_follow_up_mode")

            # =================================================================
            # Compaction
            # =================================================================

            case "compact":
                result = await session.compact(command.get("customInstructions"))
                return success(id, "compact", result)

            case "set_auto_compaction":
                session.set_auto_compaction_enabled(command.get("enabled"))
                return success(id, "set_auto_compaction")

            # =================================================================
            # Retry
            # =================================================================

            case "set_auto_retry":
                session.set_auto_retry_enabled(command.get("enabled"))
                return success(id, "set_auto_retry")

            case "abort_retry":
                session.abort_retry()
                return success(id, "abort_retry")

            # =================================================================
            # Bash
            # =================================================================

            case "bash":
                result = await session.execute_bash(
                    command.get("command"),
                    None,
                    {"exclude_from_context": command.get("excludeFromContext"), "id": id},
                )
                return success(id, "bash", result)

            case "abort_bash":
                session.abort_bash()
                return success(id, "abort_bash")

            # =================================================================
            # Session
            # =================================================================

            case "get_session_stats":
                stats = session.get_session_stats()
                return success(id, "get_session_stats", stats)

            case "export_html":
                path = await session.export_to_html(command.get("outputPath"))
                return success(id, "export_html", {"path": path})

            case "switch_session":
                result = await runtime_host.switch_session(command.get("sessionPath"))
                if not result["cancelled"]:
                    await rebind_session()
                return success(id, "switch_session", result)

            case "fork":
                result = await runtime_host.fork(command.get("entryId"))
                if not result["cancelled"]:
                    await rebind_session()
                return success(id, "fork", {"text": result.get("selectedText"), "cancelled": result["cancelled"]})

            case "clone":
                leaf_id = session.session_manager.get_leaf_id()
                if not leaf_id:
                    return error(id, "clone", "Cannot clone session: no current entry selected")
                result = await runtime_host.fork(leaf_id, position="at")
                if not result["cancelled"]:
                    await rebind_session()
                return success(id, "clone", {"cancelled": result["cancelled"]})

            case "get_fork_messages":
                messages = session.get_user_messages_for_forking()
                return success(id, "get_fork_messages", {"messages": messages})

            case "get_entries":
                session_manager: SessionManager = session.session_manager
                entries = session_manager.get_entries()
                since = command.get("since")
                if since is not None:
                    since_index = next((i for i, e in enumerate(entries) if e.get("id") == since), -1)
                    if since_index == -1:
                        return error(id, "get_entries", f"Entry not found: {since}")
                    entries = entries[since_index + 1 :]
                return success(id, "get_entries", {"entries": entries, "leafId": session_manager.get_leaf_id()})

            case "get_tree":
                session_manager = session.session_manager
                return success(
                    id, "get_tree", {"tree": session_manager.get_tree(), "leafId": session_manager.get_leaf_id()}
                )

            case "get_last_assistant_text":
                text = session.get_last_assistant_text()
                return success(id, "get_last_assistant_text", {"text": text})

            case "set_session_name":
                name = (command.get("name") or "").strip()
                if not name:
                    return error(id, "set_session_name", "Session name cannot be empty")
                await session.set_session_name(name)
                return success(id, "set_session_name")

            # =================================================================
            # Messages
            # =================================================================

            case "get_messages":
                return success(id, "get_messages", {"messages": session.messages})

            # =================================================================
            # Commands (available for invocation via prompt)
            # =================================================================

            case "get_commands":
                commands: list[RpcSlashCommand] = []

                for registered in session.extension_runner.get_registered_commands():
                    commands.append(
                        RpcSlashCommand(
                            name=registered.invocation_name,
                            description=registered.description,
                            source="extension",
                            source_info=registered.source_info,
                        )
                    )

                for template in session.prompt_templates:
                    commands.append(
                        RpcSlashCommand(
                            name=template.name,
                            description=template.description,
                            source="prompt",
                            source_info=template.source_info,
                        )
                    )

                for skill in session.resource_loader.get_skills().skills:
                    commands.append(
                        RpcSlashCommand(
                            name=f"skill:{skill.name}",
                            description=skill.description,
                            source="skill",
                            source_info=skill.source_info,
                        )
                    )

                return success(id, "get_commands", {"commands": commands})

            case _:
                return error(id, str(command_type), f"Unknown command: {command_type}")

    # Shutdown handling: called after handling each command and when stdin ends.
    async def shutdown(exit_code: int = 0, signal_name: str | None = None) -> None:
        nonlocal shutting_down
        if shutting_down:
            hard_exit(exit_code)
        shutting_down = True
        for cleanup in signal_cleanup_handlers:
            cleanup()
        if unsubscribe is not None:
            unsubscribe()
        if unsubscribe_backpressure is not None:
            unsubscribe_backpressure()
        await runtime_host.dispose()
        if signal_name != "SIGTERM":
            await flush_raw_stdout()
        hard_exit(exit_code)

    async def check_shutdown_requested() -> None:
        if not shutdown_requested:
            return
        await shutdown()

    async def handle_input_line(line: str) -> None:
        try:
            parsed = json.loads(line)
        except Exception as parse_error:
            output(error(None, "parse", f"Failed to parse command: {parse_error}"))
            await wait_for_raw_stdout_backpressure()
            return

        # Handle extension UI responses
        if isinstance(parsed, dict) and parsed.get("type") == "extension_ui_response":
            pending = pending_extension_requests.pop(parsed.get("id"), None)
            if pending is not None:
                pending(parsed)
            return

        command = parsed if isinstance(parsed, dict) else {}
        try:
            response = await handle_command(command)
            if response is not None:
                output(response)
                await wait_for_raw_stdout_backpressure()
            await check_shutdown_requested()
        except Exception as command_error:
            output(error(command.get("id"), command.get("type"), str(command_error)))
            await wait_for_raw_stdout_backpressure()

    # Lines are handled fire-and-forget so a long prompt does not block
    # subsequent commands (pi: void handleInputLine(line)).
    def on_input_line(line: str) -> None:
        tonio.spawn.without_tracking(handle_input_line(line))

    await _pump_stdin_commands(on_input_line, shutdown)


async def _pump_stdin_commands(on_line, on_end) -> None:
    """Strict-JSONL read loop over stdin.

    Module-level so tests can substitute it and drive lines directly (pi's
    tests mock attachJsonlLineReader the same way)."""
    decoder = JsonlLineDecoder()
    # Readiness-driven when stdin is a pipe or socket — the usual case here,
    # where the peer is a supervising program rather than a shell — and
    # `fs.wrap_file` when it is a redirected file. Either way no pool thread
    # sits parked between records.
    reader = FdReader(sys.stdin.fileno(), size=_STDIN_READ_SIZE)
    try:
        while True:
            chunk = await reader.read()
            if not chunk:
                for line in decoder.end():
                    on_line(line)
                await on_end()
                return
            for line in decoder.feed(chunk):
                on_line(line)
    finally:
        reader.close()


def _parse_images(images: Any) -> list[Any] | None:
    """Convert wire image dicts to ImageContent values."""
    if not images:
        return None

    parsed: list[Any] = []
    for image in images:
        if isinstance(image, dict):
            parsed.append(ImageContent(data=image.get("data"), mime_type=image.get("mimeType")))
        else:
            parsed.append(image)
    return parsed
