"""Mirror of pi coding-agent src/core/tools/bash.ts.

Local execution uses `tonio.open_process` with a detached process group, a
timeout watchdog, and a post-exit stdio grace window mirroring pi's
waitForChildProcess semantics (earendil-works/pi#5303).
"""

import math
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs, time as tonio_time

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Container, Text, truncate_to_width
from pidrei_tui._timers import Interval

from ...modes.interactive.components.keybinding_hints import key_hint
from ...modes.interactive.components.visual_truncate import truncate_to_visual_lines
from ...modes.interactive.theme import theme
from ...utils.shell import (
    ShellConfig,
    get_shell_config,
    get_shell_env,
    kill_process_tree,
    track_detached_child_pid,
    untrack_detached_child_pid,
)
from ..experimental import get_experimental_tool_sampling
from ..extensions.types import ToolDefinition
from .output_accumulator import OutputAccumulator
from .render_utils import get_text_output, invalid_arg_text, str_or_none
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size


MAX_TIMEOUT_MS = 2_147_483_647
MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS / 1000

_EXIT_STDIO_GRACE_S = 0.1

BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}

BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION: dict[str, Any] = {
    "snippet": "Execute bash commands (ls, grep, find, etc.)",
    "guidelines": ("You can inspect PIDREI_* environment variables for current model and session details.",),
}


def _resolve_timeout_ms(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise Exception("Invalid timeout: must be a finite number of seconds")

    timeout_ms = timeout * 1000
    if timeout_ms > MAX_TIMEOUT_MS:
        raise Exception(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")
    return timeout_ms


@dataclass(slots=True)
class BashToolDetails:
    truncation: TruncationResult | None = None
    full_output_path: str | None = None


BASH_PREVIEW_LINES = 5


class BashResultRenderComponent(Container):
    def __init__(self) -> None:
        super().__init__()
        self.state = {"cachedWidth": None, "cachedLines": None, "cachedSkipped": None}


class _BashPreviewOutput:
    """Width-aware cached preview (pi's inline render object)."""

    def __init__(self, state: dict, styled_output: str) -> None:
        self._state = state
        self._styled_output = styled_output

    def render(self, width: int) -> list:
        state = self._state
        if state["cachedLines"] is None or state["cachedWidth"] != width:
            preview = truncate_to_visual_lines(self._styled_output, BASH_PREVIEW_LINES, width)
            state["cachedLines"] = preview["visualLines"]
            state["cachedSkipped"] = preview["skippedCount"]
            state["cachedWidth"] = width
        if state["cachedSkipped"]:
            hint = (
                theme.fg("muted", f"... ({state['cachedSkipped']} earlier lines,")
                + f" {key_hint('app.tools.expand', 'to expand')}"
                + theme.fg("muted", ")")
            )
            return ["", truncate_to_width(hint, width, "..."), *(state["cachedLines"] or [])]
        return ["", *(state["cachedLines"] or [])]

    def invalidate(self) -> None:
        self._state["cachedWidth"] = None
        self._state["cachedLines"] = None
        self._state["cachedSkipped"] = None


def _format_duration(ms: float) -> str:
    return f"{ms / 1000:.1f}s"


def _format_shell_call(args: dict | None, prompt: str) -> str:
    args = args or {}
    command = str_or_none(args.get("command"))
    timeout = args.get("timeout")
    timeout_suffix = theme.fg("muted", f" (timeout {timeout}s)") if timeout else ""
    if command is None:
        command_display = invalid_arg_text(theme)
    elif command:
        command_display = command
    else:
        command_display = theme.fg("toolOutput", "...")
    return theme.fg("toolTitle", theme.bold(f"{prompt} {command_display}")) + timeout_suffix


def _result_details(result):
    details = result.get("details") if isinstance(result, dict) else getattr(result, "details", None)
    return details


def _rebuild_bash_result_render_component(
    component: BashResultRenderComponent,
    result,
    options: dict,
    show_images: bool,
    started_at: float | None,
    ended_at: float | None,
) -> None:
    state = component.state
    component.clear()

    output = get_text_output(result, show_images).strip()
    details = _result_details(result)
    truncation = getattr(details, "truncation", None) if details is not None else None
    full_output_path = getattr(details, "full_output_path", None) if details is not None else None
    if (
        not options.get("isPartial")
        and truncation is not None
        and truncation.truncated
        and full_output_path
        and output.endswith("]")
    ):
        footer_start = output.rfind("\n\n[")
        if footer_start != -1 and full_output_path in output[footer_start:]:
            output = output[:footer_start].rstrip()

    if output:
        styled_output = "\n".join(theme.fg("toolOutput", line) for line in output.split("\n"))

        if options.get("expanded"):
            component.add_child(Text(f"\n{styled_output}", 0, 0))
        else:
            component.add_child(_BashPreviewOutput(state, styled_output))

    if (truncation is not None and truncation.truncated) or full_output_path:
        warnings: list = []
        if full_output_path:
            warnings.append(f"Full output: {full_output_path}")
        if truncation is not None and truncation.truncated:
            if truncation.truncated_by == "lines":
                warnings.append(f"Truncated: showing {truncation.output_lines} of {truncation.total_lines} lines")
            else:
                max_bytes = truncation.max_bytes if truncation.max_bytes is not None else DEFAULT_MAX_BYTES
                warnings.append(f"Truncated: {truncation.output_lines} lines shown ({format_size(max_bytes)} limit)")
        component.add_child(Text("\n" + theme.fg("warning", f"[{'. '.join(warnings)}]"), 0, 0))

    if started_at is not None:
        label = "Elapsed" if options.get("isPartial") else "Took"
        end_time = ended_at if ended_at is not None else time.time() * 1000
        component.add_child(Text("\n" + theme.fg("muted", f"{label} {_format_duration(end_time - started_at)}"), 0, 0))


@dataclass(slots=True)
class BashExecResult:
    exit_code: int | None


@dataclass(slots=True, kw_only=True)
class BashSpawnContext:
    command: str
    cwd: str
    env: dict[str, str]


class LocalShellOperations:
    """Shared process execution used by the built-in shell tools."""

    def __init__(self, shell_name: str, resolve_shell_config: Callable[[], ShellConfig]):
        self._shell_name = shell_name
        self._resolve_shell_config = resolve_shell_config

    async def exec(
        self,
        command: str,
        cwd: str,
        *,
        on_data,
        cancel=None,
        timeout: float | None = None,
        env: dict[str, str] | None = None,
    ) -> BashExecResult:
        timeout_ms = _resolve_timeout_ms(timeout)
        if cancel is not None and cancel.cancelled:
            raise Exception("aborted")
        shell_config = await tonio.spawn_blocking(self._resolve_shell_config)
        if not await fs.Path(cwd).exists():
            raise Exception(f"Working directory does not exist: {cwd}\nCannot execute {self._shell_name} commands.")

        process = await tonio.open_process(
            [shell_config.shell, *shell_config.args, command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env if env is not None else get_shell_env(),
            start_new_session=True,
        )
        pid = process.pid
        if pid:
            track_detached_child_pid(pid)
        timed_out = False
        activity = {"count": 0}
        readers_done = tonio.Event()
        exited = tonio.Event()
        # stdout/stderr readers run in parallel; `on_data` (accumulator +
        # throttle state) assumed pi's single-thread ordering.
        chunk_lock = threading.Lock()

        def handle_chunk(chunk: bytes) -> None:
            with chunk_lock:
                activity["count"] += 1
                on_data(chunk)

        async def read_stream(stream) -> None:
            if stream is None:
                return
            try:
                while True:
                    chunk = await stream.receive_some()
                    if not chunk:
                        return
                    handle_chunk(chunk)
            except Exception:
                pass  # Stream force-closed by the post-exit grace logic or broken pipe.

        async def read_streams() -> None:
            try:
                await tonio.spawn(read_stream(process.stdout), read_stream(process.stderr))
            finally:
                readers_done.set()

        tonio.spawn.without_tracking(read_streams())

        async def watchdog() -> None:
            nonlocal timed_out
            await exited.wait(timeout_ms / 1000)
            if not exited.is_set():
                timed_out = True
                if pid:
                    kill_process_tree(pid)

        watchdog_join = tonio.spawn(watchdog()) if timeout_ms is not None else None

        unsubscribe = None
        if cancel is not None:
            unsubscribe = cancel.on_cancel(lambda _reason: kill_process_tree(pid))

        try:
            exit_code = await process.wait()
        finally:
            exited.set()
            if watchdog_join is not None:
                await watchdog_join
            if unsubscribe is not None:
                unsubscribe()
            if pid:
                untrack_detached_child_pid(pid)

        # Post-exit stdio grace: detached descendants can keep the inherited
        # pipes open. Give the readers a grace window per burst of data
        # (pi re-arms a 100 ms idle timer on each chunk), then force-close.
        while not readers_done.is_set():
            before = activity["count"]
            await readers_done.wait(_EXIT_STDIO_GRACE_S)
            if readers_done.is_set():
                break
            if activity["count"] == before:
                break

        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

        if cancel is not None and cancel.cancelled:
            raise Exception("aborted")
        if timed_out:
            raise Exception(f"timeout:{_format_timeout(timeout)}")
        return BashExecResult(exit_code=exit_code)


def _format_timeout(timeout: float | None) -> str:
    if timeout is None:
        return ""
    return f"{timeout:g}"


def create_local_bash_operations(*, shell_path: str | None = None) -> LocalShellOperations:
    """Create bash operations using the built-in local shell execution backend.

    This is useful for extensions that intercept user_bash and still want the
    standard local shell behavior while wrapping or rewriting commands.
    """
    return LocalShellOperations("bash", lambda: get_shell_config(shell_path))


_SESSION_ENV_VARS = (
    "PIDREI_SESSION_ID",
    "PIDREI_SESSION_FILE",
    "PIDREI_PROVIDER",
    "PIDREI_MODEL",
    "PIDREI_REASONING_LEVEL",
)


def _resolve_spawn_context(
    command: str,
    cwd: str,
    spawn_hook,
    expose_session_environment: bool,
    ctx,
) -> BashSpawnContext:
    env = dict(get_shell_env())
    for name in _SESSION_ENV_VARS:
        env.pop(name, None)
    if expose_session_environment and ctx is not None:
        model = getattr(ctx, "model", None)
        session_manager = getattr(ctx, "session_manager", None)
        if session_manager is not None:
            env["PIDREI_SESSION_ID"] = session_manager.get_session_id()
            session_file = session_manager.get_session_file()
            if session_file:
                env["PIDREI_SESSION_FILE"] = session_file
        if model is not None:
            env["PIDREI_PROVIDER"] = model.provider
            env["PIDREI_MODEL"] = model.id
        thinking_level = getattr(ctx, "thinking_level", None)
        if thinking_level:
            env["PIDREI_REASONING_LEVEL"] = thinking_level
    base_context = BashSpawnContext(command=command, cwd=cwd, env=env)
    return spawn_hook(base_context) if spawn_hook is not None else base_context


BASH_UPDATE_THROTTLE_S = 0.1


@dataclass(frozen=True, slots=True)
class ShellToolConfig:
    """What distinguishes one built-in shell tool from another.

    pi added this so the bash tool and a Windows-only powershell tool could
    share one implementation. powershell is dropped surface here (POSIX-only;
    PORT_0.84.3.md decision 2), so bash is the only config in tree — the seam
    is kept so the shared body stays 1:1 with upstream.
    """

    name: str
    label: str
    shell_name: str
    prompt: str
    prompt_snippet: str
    prompt_guidelines: tuple[str, ...] | None
    temp_file_prefix: str


def create_shell_tool_definition(
    cwd: str,
    config: ShellToolConfig,
    *,
    operations: Any = None,
    command_prefix: str | None = None,
    shell_path: str | None = None,
    expose_session_environment: bool = True,
    spawn_hook=None,
) -> ToolDefinition:
    ops = operations if operations is not None else create_local_bash_operations(shell_path=shell_path)

    async def execute(_tool_call_id, params, cancel=None, on_update=None, ctx=None):
        command = params["command"]
        timeout = params.get("timeout")
        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command
        spawn_context = _resolve_spawn_context(resolved_command, cwd, spawn_hook, expose_session_environment, ctx)
        output = OutputAccumulator(temp_file_prefix=config.temp_file_prefix)
        state = {
            "accepting_output": True,
            "dirty": False,
            "last_update_at": 0.0,
            "timer_generation": 0,
            "timer_running": False,
        }

        def emit_output_update() -> None:
            if on_update is None or not state["dirty"]:
                return
            state["dirty"] = False
            state["last_update_at"] = time.monotonic()
            snapshot = output.snapshot(persist_if_truncated=True)
            on_update(
                AgentToolResult(
                    content=[TextContent(text=snapshot.content or "")],
                    details=BashToolDetails(
                        truncation=snapshot.truncation if snapshot.truncation.truncated else None,
                        full_output_path=snapshot.full_output_path,
                    ),
                )
            )

        def clear_update_timer() -> None:
            state["timer_generation"] += 1
            state["timer_running"] = False

        def schedule_output_update() -> None:
            if on_update is None:
                return
            state["dirty"] = True
            delay = BASH_UPDATE_THROTTLE_S - (time.monotonic() - state["last_update_at"])
            if delay <= 0:
                clear_update_timer()
                emit_output_update()
                return
            if state["timer_running"]:
                return
            state["timer_running"] = True
            state["timer_generation"] += 1
            generation = state["timer_generation"]

            async def fire() -> None:
                await tonio_time.sleep(delay)
                if state["timer_generation"] == generation and state["timer_running"]:
                    state["timer_running"] = False
                    emit_output_update()

            tonio.spawn.without_tracking(fire())

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        def handle_data(data: bytes) -> None:
            if not state["accepting_output"]:
                return
            output.append(data)
            schedule_output_update()

        async def finish_output():
            state["accepting_output"] = False
            output.finish()
            clear_update_timer()
            emit_output_update()
            snapshot = output.snapshot(persist_if_truncated=True)
            await output.close_temp_file()
            return snapshot

        def format_output(snapshot, empty_text: str = "(no output)"):
            truncation = snapshot.truncation
            text = snapshot.content or empty_text
            details: BashToolDetails | None = None
            if truncation.truncated:
                details = BashToolDetails(truncation=truncation, full_output_path=snapshot.full_output_path)
                start_line = truncation.total_lines - truncation.output_lines + 1
                end_line = truncation.total_lines
                if truncation.last_line_partial:
                    last_line_size = format_size(output.get_last_line_bytes())
                    text += (
                        f"\n\n[Showing last {format_size(truncation.output_bytes)} of line {end_line} "
                        f"(line is {last_line_size}). Full output: {snapshot.full_output_path}]"
                    )
                elif truncation.truncated_by == "lines":
                    text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines}. "
                        f"Full output: {snapshot.full_output_path}]"
                    )
                else:
                    text += (
                        f"\n\n[Showing lines {start_line}-{end_line} of {truncation.total_lines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). Full output: {snapshot.full_output_path}]"
                    )
            return text, details

        def append_status(text: str, status: str) -> str:
            return f"{text}\n\n{status}" if text else status

        try:
            try:
                result = await ops.exec(
                    spawn_context.command,
                    spawn_context.cwd,
                    on_data=handle_data,
                    cancel=cancel,
                    timeout=timeout,
                    env=spawn_context.env,
                )
                exit_code = result.exit_code
            except Exception as error:
                snapshot = await finish_output()
                text, _details = format_output(snapshot, "")
                message = str(error)
                if message == "aborted":
                    raise Exception(append_status(text, "Command aborted"))
                if message.startswith("timeout:"):
                    timeout_secs = message.split(":", 1)[1]
                    raise Exception(append_status(text, f"Command timed out after {timeout_secs} seconds"))
                raise

            snapshot = await finish_output()
            output_text, details = format_output(snapshot)
            if exit_code is not None and exit_code != 0:
                raise Exception(append_status(output_text, f"Command exited with code {exit_code}"))
            return AgentToolResult(content=[TextContent(text=output_text)], details=details)
        finally:
            clear_update_timer()

    def render_call(args, _theme, context):
        state = context["state"]
        if context["executionStarted"] and state.get("startedAt") is None:
            state["startedAt"] = time.time() * 1000
            state["endedAt"] = None
        text = context["lastComponent"] if isinstance(context.get("lastComponent"), Text) else Text("", 0, 0)
        text.set_text(_format_shell_call(args, config.prompt))
        return text

    def render_result(result, options, _theme, context):
        state = context["state"]
        if state.get("startedAt") is not None and options.get("isPartial") and not state.get("interval"):
            invalidate = context["invalidate"]

            async def tick() -> None:
                invalidate()

            state["interval"] = Interval(1000, tick)
        if not options.get("isPartial") or context["isError"]:
            if state.get("endedAt") is None:
                state["endedAt"] = time.time() * 1000
            if state.get("interval"):
                state["interval"].cancel()
                state["interval"] = None
        component = (
            context["lastComponent"]
            if isinstance(context.get("lastComponent"), BashResultRenderComponent)
            else BashResultRenderComponent()
        )
        _rebuild_bash_result_render_component(
            component, result, options, context["showImages"], state.get("startedAt"), state.get("endedAt")
        )
        component.invalidate()
        return component

    return ToolDefinition(
        name=config.name,
        label=config.label,
        description=(
            f"Execute a {config.shell_name} command in the current working directory. "
            "Returns stdout and stderr. "
            f"Output is truncated to last {DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            "(whichever is hit first). If truncated, full output is saved to a temp file. "
            "Optionally provide a timeout in seconds."
        ),
        prompt_snippet=config.prompt_snippet,
        prompt_guidelines=(
            list(config.prompt_guidelines)
            if expose_session_environment and config.prompt_guidelines is not None
            else None
        ),
        parameters=BASH_SCHEMA,
        constrained_sampling=get_experimental_tool_sampling(),
        execute=execute,
        render_call=render_call,
        render_result=render_result,
    )


BASH_TOOL_CONFIG = ShellToolConfig(
    name="bash",
    label="bash",
    shell_name="bash",
    prompt="$",
    prompt_snippet=BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION["snippet"],
    prompt_guidelines=BASH_TOOL_SYSTEM_PROMPT_CONTRIBUTION["guidelines"],
    temp_file_prefix="pidrei-bash",
)


def create_bash_tool_definition(cwd: str, **options) -> ToolDefinition:
    return create_shell_tool_definition(cwd, BASH_TOOL_CONFIG, **options)


def create_bash_tool(cwd: str, **options) -> WrappedDefinitionTool:
    definition = create_bash_tool_definition(cwd, **options)
    return wrap_tool_definition(definition)
