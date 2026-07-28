"""Mirror of pi coding-agent src/modes/print-mode.ts.

Print mode (single-shot): Send prompts, output result, exit.

Used for:
- `pidrei -p "prompt"` - text output
- `pidrei --mode json "prompt"` - JSON event stream
"""

import json
import signal
import sys
from dataclasses import dataclass, field
from typing import Any

from ..core.agent_session import ExtensionBindings
from ..core.json_wire import to_wire
from ..core.output_guard import flush_raw_stdout, write_raw_stdout
from ..utils.fd_io import hard_exit
from ..utils.shell import kill_tracked_detached_children


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


async def run_print_mode(runtime_host, options: PrintModeOptions) -> int:
    """Run in print (single-shot) mode.

    Sends prompts to the agent and outputs the result.
    """
    mode = options.mode
    messages = options.messages or []
    exit_code = 0
    session = runtime_host.session
    unsubscribe: Any = None
    disposed = False
    signal_cleanup_handlers: list[Any] = []

    async def dispose_runtime() -> None:
        nonlocal disposed
        if disposed:
            return
        disposed = True
        if unsubscribe is not None:
            unsubscribe()
        await runtime_host.dispose()

    def register_signal_handlers() -> None:
        # pi awaits runtime disposal before exiting on a signal; a sync
        # Python signal handler cannot await, so the handler kills tracked
        # children and exits directly. Extension shutdown on signal is
        # pi's regression 5080, still to mirror.
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
            # Signals can only be installed from the main thread.
            pass

    register_signal_handlers()

    async def rebind_from_runtime(_session=None) -> None:
        await rebind_session()

    runtime_host.set_rebind_session(rebind_from_runtime)

    async def rebind_session() -> None:
        nonlocal session, unsubscribe
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
            print(f"Extension error ({err.extension_path}): {err.error}", file=sys.stderr)

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

        def on_event(event) -> None:
            if mode == "json":
                write_raw_stdout(json.dumps(to_wire(event), ensure_ascii=False) + "\n")

        unsubscribe = session.subscribe(on_event)

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
                    print(
                        last_message.error_message or f"Request {last_message.stop_reason}",
                        file=sys.stderr,
                    )
                    exit_code = 1
                else:
                    for content in last_message.content:
                        if getattr(content, "type", None) == "text":
                            write_raw_stdout(f"{content.text}\n")

        return exit_code
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        for cleanup in signal_cleanup_handlers:
            cleanup()
        await dispose_runtime()
        await flush_raw_stdout()
