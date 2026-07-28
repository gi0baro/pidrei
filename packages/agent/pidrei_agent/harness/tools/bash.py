"""Bash tool (port of pi `harness/tools/bash.ts`).

Runs a command through the execution env with streamed, 100 ms-throttled
partial updates and tail truncation; truncated runs point at a temp file with
the full output.

pi throttles updates with a `setTimeout` on the single JS thread; the port
spawns a tonio timer task instead, and guards the throttle state with a lock
because output chunks arrive from the env's reader tasks.
"""

import math
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_ai.types import TextContent
from pidrei_ai.utils.cancel import CancelToken

from ...types import AgentToolResult, AgentToolUpdateCallback
from ..types import AgentHarnessTool, get_or_throw
from ..utils.shell_output import ShellCaptureOptions, ShellCaptureProgress, execute_shell_with_capture
from ..utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size
from .tool_context import ExecutionToolContext


MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
BASH_UPDATE_THROTTLE_SECONDS = 0.1

_BASH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}


@dataclass(slots=True)
class BashToolDetails:
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


@dataclass(slots=True)
class BashExecution:
    command: str
    cwd: str
    env: dict[str, str]
    inherit_env: bool


# Prepare hook: receives (execution, context, cancel); awaitable-returning
# (async-only callback policy; pi types this `void | Promise<void>`).
type BashPrepare = Callable[[BashExecution, Any, CancelToken | None], Awaitable[None]]


@dataclass(slots=True)
class BashToolOptions:
    command_prefix: str | None = None
    prepare: BashPrepare | None = None


def _validate_timeout(timeout: float | None) -> None:
    if timeout is None:
        return
    if not math.isfinite(timeout) or timeout <= 0:
        raise Exception("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise Exception(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


class BashTool(AgentHarnessTool[ExecutionToolContext, BashToolDetails | None]):
    name = "bash"
    label = "bash"
    description = (
        "Execute a bash command in the current working directory. Returns stdout and stderr. "
        f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
        "(whichever is hit first). If truncated, full output is saved to a temp file. "
        "Optionally provide a timeout in seconds."
    )
    parameters = _BASH_SCHEMA

    def __init__(self, options: BashToolOptions | None = None):
        self._options = options

    async def execute(
        self,
        tool_call_id: str,
        params: dict[str, Any],
        cancel: CancelToken | None,
        on_update: AgentToolUpdateCallback[BashToolDetails | None] | None,
        context: ExecutionToolContext,
    ) -> AgentToolResult[BashToolDetails | None]:
        command: str = params["command"]
        timeout: float | None = params.get("timeout")
        _validate_timeout(timeout)
        env = context.env
        options = self._options
        prefixed = f"{options.command_prefix}\n{command}" if options is not None and options.command_prefix else command
        execution = BashExecution(command=prefixed, cwd=env.cwd, env={}, inherit_env=True)
        if options is not None and options.prepare is not None:
            await options.prepare(execution, context, cancel)

        throttle_lock = threading.RLock()
        get_latest_progress: Callable[[], ShellCaptureProgress] | None = None
        update_dirty = False
        last_update_at = 0.0
        pending_timer: tonio.Event | None = None

        def emit_output_update() -> None:
            nonlocal update_dirty, last_update_at
            with throttle_lock:
                if on_update is None or not update_dirty or get_latest_progress is None:
                    return
                update_dirty = False
                last_update_at = time.monotonic()
                getter = get_latest_progress
            # Call the getter outside the throttle lock: it takes the capture
            # state lock, which chunk callbacks hold while entering the
            # throttle lock (opposite order would deadlock).
            progress = getter()
            on_update(
                AgentToolResult(
                    content=[TextContent(text=progress.output)],
                    details=BashToolDetails(
                        truncation=progress.truncation if progress.truncation.truncated else None,
                        full_output_path=progress.full_output_path,
                    ),
                )
            )

        def clear_update_timer() -> None:
            nonlocal pending_timer
            with throttle_lock:
                if pending_timer is None:
                    return
                pending_timer.set()
                pending_timer = None

        async def update_timer(cancel_event: tonio.Event, delay: float) -> None:
            nonlocal pending_timer
            await cancel_event.wait(delay)
            with throttle_lock:
                if pending_timer is cancel_event:
                    pending_timer = None
                if cancel_event.is_set():
                    return
            emit_output_update()

        def schedule_output_update() -> None:
            nonlocal update_dirty, pending_timer
            if on_update is None:
                return
            with throttle_lock:
                update_dirty = True
                delay = BASH_UPDATE_THROTTLE_SECONDS - (time.monotonic() - last_update_at)
                if delay <= 0:
                    if pending_timer is not None:
                        pending_timer.set()
                        pending_timer = None
                    should_emit = True
                else:
                    should_emit = False
                    if pending_timer is None:
                        pending_timer = tonio.Event()
                        tonio.spawn.without_tracking(update_timer(pending_timer, delay))
            if should_emit:
                emit_output_update()

        def on_chunk(_chunk: str, get_progress: Callable[[], ShellCaptureProgress]) -> None:
            nonlocal get_latest_progress
            with throttle_lock:
                get_latest_progress = get_progress
            schedule_output_update()

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))
        try:
            capture = get_or_throw(
                await execute_shell_with_capture(
                    env,
                    execution.command,
                    ShellCaptureOptions(
                        cwd=execution.cwd,
                        env=execution.env,
                        inherit_env=execution.inherit_env,
                        timeout=timeout,
                        cancel=cancel,
                        return_execution_errors=True,
                        on_chunk=on_chunk,
                    ),
                )
            )
            clear_update_timer()
            with throttle_lock:
                get_latest_progress = lambda: ShellCaptureProgress(
                    output=capture.output,
                    truncation=capture.truncation,
                    full_output_path=capture.full_output_path,
                    last_line_bytes=capture.last_line_bytes,
                )
                update_dirty = True
            emit_output_update()

            output_text = capture.output
            details: BashToolDetails | None = None
            if capture.truncation.truncated:
                details = BashToolDetails(truncation=capture.truncation, full_output_path=capture.full_output_path)
                start_line = capture.truncation.total_lines - capture.truncation.output_lines + 1
                end_line = capture.truncation.total_lines
                if capture.truncation.last_line_partial:
                    last_line_size = format_size(capture.last_line_bytes)
                    output_text += (
                        f"\n\n[Showing last {format_size(capture.truncation.output_bytes)} of line {end_line} "
                        f"(line is {last_line_size}). Full output: {capture.full_output_path}]"
                    )
                elif capture.truncation.truncated_by == "lines":
                    output_text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {capture.truncation.total_lines}. "
                        f"Full output: {capture.full_output_path}]"
                    )
                else:
                    output_text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {capture.truncation.total_lines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {capture.full_output_path}]"
                    )

            def append_status(status: str) -> str:
                return f"{output_text}\n\n{status}" if output_text else status

            if capture.cancelled:
                raise Exception(append_status("Command aborted"))
            if capture.execution_error is not None and capture.execution_error.code == "timeout":
                raise Exception(append_status(f"Command timed out after {timeout} seconds")) from (
                    capture.execution_error
                )
            if capture.execution_error is not None:
                raise capture.execution_error
            if capture.exit_code != 0 and capture.exit_code is not None:
                raise Exception(append_status(f"Command exited with code {capture.exit_code}"))
            return AgentToolResult(content=[TextContent(text=output_text or "(no output)")], details=details)
        finally:
            clear_update_timer()


def create_bash_tool(options: BashToolOptions | None = None) -> BashTool:
    return BashTool(options)
