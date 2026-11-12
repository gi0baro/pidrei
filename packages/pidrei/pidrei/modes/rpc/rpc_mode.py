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
import threading
import uuid
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import ImageContent
from pidrei_tui.tui import call_sync

from ...core.agent_session import ExtensionBindings, PromptDisposition, PromptOptions
from ...core.bash_executor import BashResult
from ...core.json_wire import to_wire
from ...core.output_guard import drain_output, take_over_stdout, write_raw_stdout
from ...core.session_manager import SessionManager
from ...utils.fd_io import StdinReader, hard_exit
from ...utils.shell import SignalDispatcher, kill_tracked_detached_children, signal_exit_status
from ..interactive.theme import theme
from ..json_event import to_json_event
from .jsonl import JsonlLineDecoder, serialize_json_line
from .rpc_types import RpcSessionState, RpcSlashCommand


async def _answer(value: Any) -> Any:
    return value


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

    def select(self, title: str, options: list[str], opts: dict[str, Any] | None = None) -> Any:
        def parse(response: dict[str, Any]) -> str | None:
            if response.get("cancelled"):
                return None
            return response.get("value")

        timeout = (opts or {}).get("timeout")
        return self._create_dialog_promise(
            opts, None, {"method": "select", "title": title, "options": options, "timeout": timeout}, parse
        )

    def confirm(self, title: str, message: str, opts: dict[str, Any] | None = None) -> Any:
        def parse(response: dict[str, Any]) -> bool:
            if response.get("cancelled"):
                return False
            return bool(response.get("confirmed", False))

        timeout = (opts or {}).get("timeout")
        return self._create_dialog_promise(
            opts, False, {"method": "confirm", "title": title, "message": message, "timeout": timeout}, parse
        )

    def input(self, title: str, placeholder: str | None = None, opts: dict[str, Any] | None = None) -> Any:
        def parse(response: dict[str, Any]) -> str | None:
            if response.get("cancelled"):
                return None
            return response.get("value")

        timeout = (opts or {}).get("timeout")
        return self._create_dialog_promise(
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

    def set_widget(self, key: str, content: Any, options: dict[str, Any] | None = None) -> None:
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
                        "widgetPlacement": (options or {}).get("placement"),
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

    def write_terminal(self, sequence: str) -> None:
        """Raw terminal output not supported in RPC mode - the host owns the terminal."""

    def custom(self, *_args, **_kwargs) -> Any:
        # Custom UI not supported in RPC mode
        return tonio.spawn(_answer(None))

    def apply(self, fn) -> Any:
        # No UI state to guard: run it (coroutines refused, as in the TUI).
        return call_sync(fn)

    def paste_to_editor(self, text: str) -> None:
        # Paste handling not supported in RPC mode - falls back to set_editor_text
        self.set_editor_text(text)

    def set_editor_text(self, text: str) -> None:
        # Fire and forget - host can implement editor control
        self._output(
            {"type": "extension_ui_request", "id": str(uuid.uuid4()), "method": "set_editor_text", "text": text}
        )

    def get_editor_text(self) -> str:
        # No request round-trip (pi's method is sync and returns ""): the
        # host should track editor state locally if needed.
        return ""

    def editor(self, title: str, prefill: str | None = None) -> Any:
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

        async def wait() -> str | None:
            await event.wait()
            response = slot.get("response") or {}
            if response.get("cancelled"):
                return None
            return response.get("value")

        return tonio.spawn(wait())

    def add_autocomplete_provider(self, *_args, **_kwargs) -> None:
        """Autocomplete provider composition is not supported in RPC mode."""

    def set_editor_component(self, *_args, **_kwargs) -> None:
        """Custom editor components not supported in RPC mode."""

    def get_editor_component(self) -> Any:
        # Custom editor components not supported in RPC mode
        return None

    @property
    def theme(self) -> Any:
        # The global theme, as pi's (`main` initialises it in every mode).
        return theme

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


async def run_rpc_mode(runtime_host, *, stdin: StdinReader, signal_dispatcher: SignalDispatcher | None = None) -> None:
    """Run in RPC mode.

    Listens for JSON commands on `stdin`, outputs events and responses on
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

    # Extension shutdown requests (ctx.shutdown()) wait until input is attached, no command is
    # being handled, and the session is idle. See check_shutdown_requested().
    shutdown_requested = False
    shutting_down = False
    input_ready = False
    active_commands = 0
    # SIGTERM/SIGHUP received so far. Only a repeated signal forces exit during shutdown.
    signal_count = 0
    # The fields above are decided and claimed from the signal handler, stdin's end, every
    # handled line, settling prompts and the session listener, in parallel: each decision
    # and its action are one section under this (sync sections only).
    shutdown_guard = threading.Lock()
    # Set when a started shutdown ends without exiting the process (its dispose raised): input
    # that ended while it ran stops waiting for it.
    shutdown_ended = tonio.Event()

    def create_dialog_promise(
        opts: dict[str, Any] | None, default_value: Any, request: dict[str, Any], parse_response
    ) -> Any:
        """Helper for dialog methods with cancel/timeout support. The request
        goes out at call time; returns the spawn handle of the wait for the
        answer, which the caller may await or drop (spec/ui-island.md, `ctx.ui`)."""
        cancel = (opts or {}).get("signal")
        timeout = (opts or {}).get("timeout")
        if cancel is not None and cancel.cancelled:
            return tonio.spawn(_answer(default_value))

        request_id = str(uuid.uuid4())
        event = tonio.Event()
        slot = tonio.Result()

        def resolve(response: dict[str, Any]) -> None:
            slot.store(response)
            event.set()

        pending_extension_requests[request_id] = resolve

        output(_compact({"type": "extension_ui_request", "id": request_id, **request}))

        async def wait() -> Any:
            try:
                if cancel is not None:
                    # `timeout` is in milliseconds; `Waiter.any` takes microseconds.
                    await tonio.Waiter.any(event, cancel.event, timeout=round(timeout * 1000) if timeout else None)
                elif timeout:
                    await event.wait(timeout / 1000)
                else:
                    await event.wait()
            finally:
                pending_extension_requests.pop(request_id, None)

            response = slot.fetch()
            if response is None:
                return default_value
            return parse_response(response)

        return tonio.spawn(wait())

    def create_extension_ui_context() -> _RpcExtensionUIContext:
        return _RpcExtensionUIContext(output, create_dialog_promise, pending_extension_requests)

    # Shutdown handling. Defined before the session is bound: a session_start handler can
    # request a shutdown synchronously.
    def claim_shutdown() -> bool:
        """pi's `shutdown()` runs once: the first caller claims it, later calls (stdin EOF
        after a signal) do nothing. On a claim, its synchronous prefix runs here."""
        nonlocal shutting_down
        with shutdown_guard:
            if shutting_down:
                return False
            shutting_down = True
        begin_shutdown()
        return True

    def begin_shutdown() -> None:
        """The part of pi's `shutdown()` before its first await, run by the claimer."""
        if unsubscribe is not None:
            unsubscribe()
        if unsubscribe_backpressure is not None:
            unsubscribe_backpressure()

    async def finish_shutdown(exit_code: int = 0, signal_name: str | None = None) -> None:
        """Dispose the runtime and exit: the rest of pi's `shutdown()`, after a claim."""
        try:
            await runtime_host.dispose()
            if signal_name != "SIGTERM":
                await drain_output()
            hard_exit(exit_code)
        finally:
            shutdown_ended.set()

    def check_shutdown_requested() -> None:
        """Shut down if an extension requested it and nothing still needs the process: input is
        attached, no command (including a prompt that has not settled) is being handled, and the
        session is idle. Each of those ending calls this again. The decision and the claim are
        one section under `shutdown_guard`."""
        nonlocal shutting_down
        with shutdown_guard:
            if not shutdown_requested or shutting_down or not input_ready or active_commands > 0 or not session.is_idle:
                return
            shutting_down = True
        # pi: `void shutdown()`.
        begin_shutdown()
        tonio.spawn.without_tracking(finish_shutdown())

    async def check_shutdown_requested_later() -> None:
        check_shutdown_requested()

    async def rebind_from_runtime(_session, swap) -> None:
        swap()
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
            with shutdown_guard:
                shutdown_requested = True
            # An idle session has no command response or agent_settled left to check the flag.
            # pi checks on the next tick so the requesting handler finishes first. ctx.shutdown()
            # is reachable from any extension handler, tool or timer, so there is no point here
            # where the requesting invocation is known to complete: the check is spawned, and
            # can begin the shutdown while that handler is still running the code after the
            # call. A request from a command is still held until the command finishes (the
            # command counts in `active_commands`).
            tonio.spawn.without_tracking(check_shutdown_requested_later())

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
            output(to_json_event(event))
            if getattr(event, "type", None) == "agent_settled":
                check_shutdown_requested()

        async def on_agent_event(*_args) -> None:
            await drain_output()

        unsubscribe = session.subscribe(on_event)
        unsubscribe_backpressure = session.agent.subscribe(on_agent_event)

    await rebind_session()

    # Handle a single command
    async def handle_command(command: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal active_commands
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

                def preflight_result(disposition: PromptDisposition) -> None:
                    nonlocal preflight_succeeded
                    preflight_succeeded = True
                    output(success(id, "prompt", {"disposition": disposition}))

                async def run_prompt() -> None:
                    nonlocal active_commands
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
                    finally:
                        # Synchronous, so it also runs on a cancelled chain.
                        with shutdown_guard:
                            active_commands -= 1
                        check_shutdown_requested()

                # Extension commands run inside prompt(); defer shutdown requests until it settles.
                with shutdown_guard:
                    active_commands += 1
                tonio.spawn.without_tracking(run_prompt())
                return None

            case "steer":
                disposition = await session.steer(
                    command.get("message"), _parse_images(command.get("images")), source="rpc"
                )
                return success(id, "steer", {"disposition": disposition})

            case "follow_up":
                disposition = await session.follow_up(
                    command.get("message"), _parse_images(command.get("images")), source="rpc"
                )
                return success(id, "follow_up", {"disposition": disposition})

            case "abort":
                await session.abort()
                return success(id, "abort")

            case "clear_queue":
                return success(id, "clear_queue", session.clear_queue())

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
                models = session.model_runtime.get_available_snapshot()
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
                models = session.model_runtime.get_available_snapshot()
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
                event_result = await session.extension_runner.emit_user_bash(
                    {
                        "type": "user_bash",
                        "command": command.get("command"),
                        "excludeFromContext": command.get("excludeFromContext") or False,
                        "cwd": session.session_manager.get_cwd(),
                    }
                )

                if event_result and event_result.get("result"):
                    extension_result = event_result["result"]
                    bash_result = BashResult(
                        output=extension_result.get("output") or "",
                        exit_code=extension_result.get("exitCode"),
                        cancelled=bool(extension_result.get("cancelled")),
                        truncated=bool(extension_result.get("truncated")),
                        full_output_path=extension_result.get("fullOutputPath"),
                    )
                    await session.record_bash_result(
                        command.get("command"),
                        bash_result,
                        {"excludeFromContext": command.get("excludeFromContext")},
                    )
                    return success(id, "bash", bash_result)

                result = await session.execute_bash(
                    command.get("command"),
                    None,
                    {
                        "exclude_from_context": command.get("excludeFromContext"),
                        "id": id,
                        "operations": event_result.get("operations") if event_result else None,
                    },
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

    async def handle_input_line(line: str) -> None:
        nonlocal active_commands
        try:
            parsed = json.loads(line)
        except Exception as parse_error:
            output(error(None, "parse", f"Failed to parse command: {parse_error}"))
            await drain_output()
            return

        # Handle extension UI responses
        if isinstance(parsed, dict) and parsed.get("type") == "extension_ui_response":
            pending = pending_extension_requests.pop(parsed.get("id"), None)
            if pending is not None:
                pending(parsed)
            return

        command = parsed if isinstance(parsed, dict) else {}
        with shutdown_guard:
            active_commands += 1
        try:
            try:
                response = await handle_command(command)
            except Exception as command_error:
                response = error(command.get("id"), command.get("type"), str(command_error))
                failed = True
            else:
                failed = False
            if response is not None:
                output(response)
        finally:
            # pi stops counting the command and writes its response in one synchronous step.
            # Here a check can run on another coroutine in between, so the response is written
            # first: a shutdown cannot exit ahead of it.
            with shutdown_guard:
                active_commands -= 1
        if response is not None:
            await drain_output()
        if not failed:
            check_shutdown_requested()

    # Lines are handled fire-and-forget so a long prompt does not block
    # subsequent commands (pi: void handleInputLine(line)).
    def on_input_line(line: str) -> None:
        tonio.spawn.without_tracking(handle_input_line(line))

    async def on_input_end() -> None:
        # pi: `void shutdown()`. The pump waits here, so the mode does not return while the
        # shutdown runs (pi keeps the process alive with a promise that never settles); the
        # shutdown exits the process.
        if claim_shutdown():
            await finish_shutdown()
        else:
            # A shutdown started by a signal or an extension request is running.
            await shutdown_ended.wait()

    async def on_shutdown_signal(signum: int) -> None:
        nonlocal signal_count, shutting_down
        exit_code = signal_exit_status(signum)
        kill_tracked_detached_children()
        # Count the signal, test, claim: one section, since signals are handled in parallel.
        with shutdown_guard:
            signal_count += 1
            repeated = signal_count > 1
            claimed = not shutting_down
            shutting_down = True
        if claimed:
            begin_shutdown()
            await finish_shutdown(exit_code, "SIGHUP" if signum == signal.SIGHUP else "SIGTERM")
        elif repeated:
            # Process managers often close stdin and send SIGTERM together, so one signal
            # during shutdown is not a request to skip cleanup; a repeated one is.
            hard_exit(exit_code)

    # Installed once the shutdown exists, and kept until exit: without one, a signal during
    # shutdown would kill the process before dispose() finishes. A signal before this ends
    # the process through the default handler.
    if signal_dispatcher is not None:
        signal_dispatcher.set_handler(signal.SIGTERM, on_shutdown_signal)
        signal_dispatcher.set_handler(signal.SIGHUP, on_shutdown_signal)
    with shutdown_guard:
        input_ready = True
    # Handle a shutdown requested during startup (e.g. from session_start).
    check_shutdown_requested()
    await _pump_stdin_commands(stdin, on_input_line, on_input_end)


async def _pump_stdin_commands(stdin: StdinReader, on_line, on_end) -> None:
    """Strict-JSONL read loop over stdin.

    Module-level so tests can substitute it and drive lines directly (pi's
    tests mock attachJsonlLineReader the same way)."""
    decoder = JsonlLineDecoder()
    # `stdin` reads by readiness when it is a pipe or socket — the usual case
    # here, where the peer is a supervising program rather than a shell — and
    # through `fs.wrap_file` when it is a redirected file. Either way no pool
    # thread sits parked between records.
    while True:
        for line in decoder.feed(stdin.take()):
            on_line(line)
        if not await stdin.fill():
            for line in decoder.end():
                on_line(line)
            await on_end()
            return


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
