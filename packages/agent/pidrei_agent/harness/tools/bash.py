"""Bash tool (port of pi `harness/tools/bash.ts`).

Runs a command through the execution env with bounded, source-side captured
output: the env publishes rate-limited view updates, the tool folds them into
one view for partial updates and the final result; truncated runs point at the
env's spill file with the full output.

pi's durable-recovery checkpoint flag on `onUpdate` (every 2 s) is
harness-runtime facing and not ported (PORT_0.85.0.md, `eb1185d9`).
"""

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pidrei_ai.types import TextContent
from pidrei_ai.utils.cancel import CancelToken

from ...types import AgentToolResult, AgentToolUpdateCallback
from ..types import (
    AgentHarnessTool,
    ShellExecOptions,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    ShellOutputTruncation,
    ShellOutputView,
)
from ..utils.output_capture import apply_shell_output_update
from ..utils.truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size
from .tool_context import ExecutionToolContext


MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000

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
    truncation: ShellOutputTruncation | None = None
    full_output_path: str | None = None


@dataclass(slots=True)
class BashExecution:
    command: str
    cwd: str
    env: dict[str, str]
    inherit_env: bool


# Prepare hook: receives (execution, tool_context, cancel); awaitable-returning
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
        "Execute a bash command in the current working directory. Returns combined stdout and stderr. "
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
        on_update: AgentToolUpdateCallback[BashToolDetails | None] | None,
        tool_context: ExecutionToolContext,
        cancel: CancelToken | None = None,
    ) -> AgentToolResult[BashToolDetails | None]:
        command: str = params["command"]
        timeout: float | None = params.get("timeout")
        _validate_timeout(timeout)
        env = tool_context.env
        options = self._options
        prefixed = f"{options.command_prefix}\n{command}" if options is not None and options.command_prefix else command
        execution = BashExecution(command=prefixed, cwd=env.cwd, env={}, inherit_env=True)
        if options is not None and options.prepare is not None:
            await options.prepare(execution, tool_context, cancel)

        view: ShellOutputView | None = None
        accepting_updates = True

        def handle_update(update, _cancel) -> None:
            nonlocal view
            if not accepting_updates:
                return
            view = apply_shell_output_update(view, update)
            if on_update is None:
                return
            on_update(
                AgentToolResult(
                    content=[TextContent(text=view.text)],
                    details=BashToolDetails(
                        truncation=view.truncation if view.truncation.truncated else None,
                        full_output_path=view.spill_path,
                    ),
                )
            )

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))
        result = await env.exec(
            execution.command,
            ShellExecOptions(
                cwd=execution.cwd,
                env=execution.env,
                inherit_env=execution.inherit_env,
                timeout=timeout,
                capture=ShellOutputCaptureOptions(
                    limits=ShellOutputLimits(max_bytes=DEFAULT_MAX_BYTES, max_lines=DEFAULT_MAX_LINES, retain="tail"),
                    spill=True,
                ),
                on_update=handle_update,
            ),
            cancel,
        )
        accepting_updates = False

        output_text = view.text if view is not None else ""
        if result.ok:
            capture: ShellOutputView | None = ShellOutputView(
                text=output_text,
                truncation=result.value.truncation,
                spill_path=result.value.spill_path,
                last_line_bytes=result.value.last_line_bytes,
            )
        else:
            capture = view
        details: BashToolDetails | None = None
        if capture is not None and capture.truncation.truncated:
            details = BashToolDetails(truncation=capture.truncation, full_output_path=capture.spill_path)
            start_line = capture.truncation.total_lines - capture.truncation.output_lines + 1
            end_line = capture.truncation.total_lines
            if capture.truncation.last_line_partial:
                last_line_size = format_size(
                    capture.last_line_bytes if capture.last_line_bytes is not None else capture.truncation.output_bytes
                )
                output_text += (
                    f"\n\n[Showing last {format_size(capture.truncation.output_bytes)} of line {end_line} "
                    f"(line is {last_line_size}). Full output: {capture.spill_path}]"
                )
            elif capture.truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {capture.truncation.total_lines}. "
                    f"Full output: {capture.spill_path}]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line}-{end_line} of {capture.truncation.total_lines} "
                    f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {capture.spill_path}]"
                )

        if not result.ok:
            if result.error.code == "timeout":
                status = f"Command timed out after {timeout} seconds"
            elif result.error.code == "aborted":
                status = "Command aborted"
            else:
                status = result.error.message
            raise Exception(f"{output_text}\n\n{status}" if output_text else status) from result.error
        if result.value.exit_code != 0:
            raise Exception(
                f"{output_text}\n\nCommand exited with code {result.value.exit_code}"
                if output_text
                else f"Command exited with code {result.value.exit_code}"
            )
        return AgentToolResult(content=[TextContent(text=output_text or "(no output)")], details=details)


def create_bash_tool(options: BashToolOptions | None = None) -> BashTool:
    return BashTool(options)
