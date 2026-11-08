"""Mirror of pi coding-agent src/modes/print-mode.ts.

Print mode (single-shot): Send prompts, output result, exit.

Used for:
- `pidrei -p "prompt"` - text output
- `pidrei --mode json "prompt"` - JSON event stream
"""

import json
import signal
import threading
from dataclasses import dataclass, field
from typing import Any

from ..core.agent_session import ExtensionBindings
from ..core.json_wire import to_wire
from ..core.output_guard import drain_output, write_raw_stdout, write_stderr
from ..utils.fd_io import hard_exit
from ..utils.shell import SignalDispatcher, kill_tracked_detached_children, signal_exit_status
from .json_event import to_json_event


@dataclass(slots=True, kw_only=True)
class PrintModeOptions:
    """Options for print mode."""

    # Output mode: "text" for final response only, "json" for all events
    mode: str
    # Array of additional prompts to send after initial_message
    messages: list[str] = field(default_factory=list)
    # First message to send (may contain @file content)
    initial_message: str | None = None
    # Images to attach to the initial message
    initial_images: list[Any] | None = None


async def run_print_mode(
    runtime_host, options: PrintModeOptions, *, signal_dispatcher: SignalDispatcher | None = None
) -> int:
    """Run in print (single-shot) mode.

    Sends prompts to the agent and outputs the result.
    """
    mode = options.mode
    messages = options.messages or []
    exit_code = 0
    session = runtime_host.session
    unsubscribe: Any = None
    unsubscribe_backpressure: Any = None
    disposed = False
    # The signal handler and the normal exit both dispose; the first one claims it.
    dispose_guard = threading.Lock()

    async def dispose_runtime() -> None:
        nonlocal disposed
        with dispose_guard:
            if disposed:
                return
            disposed = True
        if unsubscribe is not None:
            unsubscribe()
        if unsubscribe_backpressure is not None:
            unsubscribe_backpressure()
        await runtime_host.dispose()

    async def on_shutdown_signal(signum: int) -> None:
        # pi: kill the tracked children, then exit once the runtime is
        # disposed, so extensions get session_shutdown. Once the normal exit
        # has claimed the dispose, this exits at once (pi has removed its
        # handlers there, and the signal's default action ends the process).
        kill_tracked_detached_children()
        await dispose_runtime()
        hard_exit(signal_exit_status(signum))

    if signal_dispatcher is not None:
        signal_dispatcher.set_handler(signal.SIGTERM, on_shutdown_signal)
        signal_dispatcher.set_handler(signal.SIGHUP, on_shutdown_signal)

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

        def on_error(err) -> None:
            write_stderr(f"Extension error ({err.extension_path}): {err.error}\n")

        await session.bind_extensions(
            ExtensionBindings(
                mode="json" if mode == "json" else "print",
                command_context_actions={
                    "wait_for_idle": session.wait_for_idle,
                    "new_session": new_session,
                    "fork": fork,
                    "navigate_tree": navigate_tree,
                    "switch_session": switch_session,
                    "reload": reload,
                },
                on_error=on_error,
            )
        )

        if unsubscribe is not None:
            unsubscribe()
        if unsubscribe_backpressure is not None:
            unsubscribe_backpressure()

        def on_event(event) -> None:
            if mode == "json":
                write_raw_stdout(json.dumps(to_wire(to_json_event(event)), ensure_ascii=False) + "\n")

        async def on_agent_event(*_args) -> None:
            await drain_output()

        unsubscribe = session.subscribe(on_event)
        unsubscribe_backpressure = session.agent.subscribe(on_agent_event) if mode == "json" else None

    try:
        if mode == "json":
            header = session.session_manager.get_header()
            if header:
                write_raw_stdout(json.dumps(to_wire(header), ensure_ascii=False) + "\n")

        await rebind_session()

        if options.initial_message:
            # lazy: core <-> modes import cycle (see modes/__init__.py)
            from ..core.agent_session import PromptOptions

            await session.prompt(options.initial_message, PromptOptions(images=options.initial_images))

        for message in messages:
            await session.prompt(message)

        if mode == "text":
            state = session.state
            last_message = state.messages[-1] if state.messages else None

            if last_message is not None and getattr(last_message, "role", None) == "assistant":
                if last_message.stop_reason in ("error", "aborted"):
                    write_stderr(f"{last_message.error_message or f'Request {last_message.stop_reason}'}\n")
                    exit_code = 1
                else:
                    for content in last_message.content:
                        if getattr(content, "type", None) == "text":
                            write_raw_stdout(f"{content.text}\n")
                    if last_message.stop_reason == "length":
                        write_stderr("Response was truncated: the model reached its output token limit.\n")
                        exit_code = 1

        return exit_code
    except Exception as error:
        write_stderr(f"{error}\n")
        return 1
    finally:
        await dispose_runtime()
        await drain_output()
