"""Mirror of pi coding-agent src/core/tools/bash.ts.

Local execution uses `tonio.open_process` with a detached process group, a
timeout watchdog, and a post-exit stdio grace window mirroring pi's
waitForChildProcess semantics (earendil-works/pi#5303).
"""

import math
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs, time as tonio_time

from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import JsonSchemaConstrainedSampling, TextContent
from pidrei_utils import clock

from ...utils.process import EXIT_STDIO_GRACE_S, pipe_is_drained
from ...utils.shell import (
    ShellConfig,
    get_shell_config_blocking,
    get_shell_env,
    kill_process_tree,
    track_detached_child_pid,
    untrack_detached_child_pid,
)
from ..extensions.types import ToolDefinition
from .output_accumulator import OutputAccumulator
from .renderers.bash import BASH_UPDATE_THROTTLE_S, create_shell_renderers
from .tool_definition_wrapper import WrappedDefinitionTool, wrap_tool_definition
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, format_size


MAX_TIMEOUT_MS = 2_147_483_647
# Output limit of `structured_content["output"]`, which programmatic callers such as codemode scripts receive.
STRUCTURED_OUTPUT_MAX_BYTES = 1024 * 1024
MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS / 1000


BASH_SCHEMA = {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "number", "description": "Timeout in seconds (optional, no default timeout)"},
    },
    "required": ["command"],
}

# Result for programmatic callers such as codemode scripts. A non-zero exit code
# is an error result for the model, but scripts still resolve to this value.
# `output` is not limited like the model-facing output: callers decide how much
# of it reaches the model.
BASH_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "output": {"type": "string", "description": "Combined stdout and stderr, possibly truncated"},
        "truncated": {"type": "boolean"},
        "full_output_path": {"type": "string", "description": "Full output, when truncated"},
        "exit_code": {"type": "number"},
        "wall_time_seconds": {"type": "number"},
    },
    "required": ["output", "truncated", "exit_code", "wall_time_seconds"],
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


@dataclass(slots=True)
class BashExecResult:
    exit_code: int | None


@dataclass(frozen=True, slots=True)
class _ShellExit:
    exit_code: int
    after_seconds: float


class _ShellTimeoutError(Exception):
    """Timeout of the local shell backend. Its message keeps the `timeout:<seconds>` form custom operations raise.

    `shell_exit` is set when the shell itself exited before the deadline, so the tool was only still waiting
    because background processes kept writing to the inherited stdout/stderr pipes (#11613).
    """

    def __init__(self, timeout: float | None, shell_exit: _ShellExit | None):
        super().__init__(f"timeout:{_format_timeout(timeout)}")
        self.shell_exit = shell_exit


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
        started_at = clock.monotonic()
        # Set by this coroutine once `process.wait()` returned, read by the watchdog
        # for the timeout message (pi's `exit` listener).
        exit_lock = threading.Lock()
        shell_exited_after: float | None = None
        timed_out = False
        exited_before_deadline_after: float | None = None
        activity = {"count": 0}
        readers_done = tonio.Event()
        # The wait for the shell and its stdio is over (pi's `waitForChildProcess`
        # settled). The deadline and the abort stay armed until then: a background
        # process can keep the pipes open after the shell exited (#11613).
        settled = tonio.Event()
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

        def kill_shell_tree() -> None:
            # The shell leads its own group (`start_new_session`), and only the group is
            # signalled: this also runs after the shell exited, when its pid may be reused.
            if pid:
                kill_process_tree(pid)

        async def watchdog() -> None:
            nonlocal timed_out, exited_before_deadline_after
            await settled.wait(timeout_ms / 1000)
            if settled.is_set():
                return
            with exit_lock:
                timed_out = True
                exited_before_deadline_after = shell_exited_after
            # Also when the shell already exited: the group kill still reaches background
            # processes holding the pipes.
            kill_shell_tree()

        watchdog_join = tonio.spawn(watchdog()) if timeout_ms is not None else None

        unsubscribe = None
        if cancel is not None:
            unsubscribe = cancel.on_cancel(lambda _reason: kill_shell_tree())

        try:
            exit_code = await process.wait()
            with exit_lock:
                shell_exited_after = clock.monotonic() - started_at

            # Post-exit stdio grace: detached descendants can keep the inherited
            # pipes open. pi closes them after 100 ms with no data (its idle timer
            # re-arms on each chunk). Idle is a whole window with no chunk AND
            # nothing left unread in the pipes: a reader the scheduler has not run
            # yet (or woke but did not run yet) still has the output in its pipe.
            while not readers_done.is_set():
                before = activity["count"]
                await readers_done.wait(EXIT_STDIO_GRACE_S)
                if readers_done.is_set():
                    break
                if activity["count"] == before and all(map(pipe_is_drained, (process.stdout, process.stderr))):
                    break

            for stream in (process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            # A reader that already pulled a chunk off the pipe still delivers it;
            # the closed streams end its next receive.
            await readers_done.wait()
        finally:
            settled.set()
            if unsubscribe is not None:
                unsubscribe()
            # Tracked until the stdio settled, as pi: background processes holding the
            # pipes after the shell exited are still killed when pidrei is told to exit.
            if pid:
                untrack_detached_child_pid(pid)
        if watchdog_join is not None:
            # The watchdog returns at once now that `settled` is set; joining it
            # orders its writes before the reads below.
            await watchdog_join

        if cancel is not None and cancel.cancelled:
            raise Exception("aborted")
        # A signal-killed shell reports -signum. Use the standard shell convention so
        # callers do not mistake the termination for an ordinary exit status.
        if exit_code is not None and exit_code < 0:
            exit_code = 128 - exit_code
        if timed_out:
            raise _ShellTimeoutError(
                timeout,
                _ShellExit(exit_code=exit_code, after_seconds=_round_tenths(exited_before_deadline_after))
                if exited_before_deadline_after is not None
                else None,
            )
        return BashExecResult(exit_code=exit_code)


def _format_timeout(timeout: float | None) -> str:
    if timeout is None:
        return ""
    return f"{timeout:g}"


def _round_tenths(seconds: float) -> float:
    # pi: Math.round(ms / 100) / 10 — tenths of a second, halves rounded up.
    return math.floor(seconds * 10 + 0.5) / 10


def _js_number(value: float) -> str:
    # JS template interpolation of a number: integral floats print bare.
    return str(int(value)) if value.is_integer() else str(value)


def create_local_bash_operations(*, shell_path: str | None = None) -> LocalShellOperations:
    """Create bash operations using the built-in local shell execution backend.

    This is useful for extensions that intercept user_bash and still want the
    standard local shell behavior while wrapping or rewriting commands.
    """

    def resolve_shell_config_blocking() -> ShellConfig:
        return get_shell_config_blocking(shell_path)

    # Resolved through `spawn_blocking` by `LocalShellOperations.exec`.
    return LocalShellOperations("bash", resolve_shell_config_blocking)


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


@dataclass(frozen=True, slots=True)
class ShellToolConfig:
    """What distinguishes one built-in shell tool from another.

    pi added this so the bash tool and a Windows-only powershell tool could
    share one implementation. powershell is dropped surface here (POSIX-only;
    0.84.3 delta port), so bash is the only config in tree — the seam
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
        spawn_context = _resolve_spawn_context(
            resolved_command,
            (ctx.cwd if ctx is not None else None) or cwd,
            spawn_hook,
            expose_session_environment,
            ctx,
        )
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
            state["last_update_at"] = clock.monotonic()
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
            delay = BASH_UPDATE_THROTTLE_S - (clock.monotonic() - state["last_update_at"])
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

        started_at = clock.monotonic()

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
                    status = f"Command timed out after {timeout_secs} seconds"
                    if isinstance(error, _ShellTimeoutError) and error.shell_exit is not None:
                        # Both facts matter: the command's real result, and that its background processes
                        # were still writing output when the timeout hit (#11613).
                        status += (
                            f". The shell exited with code {error.shell_exit.exit_code} after "
                            f"{_js_number(error.shell_exit.after_seconds)} seconds, but background processes "
                            "it started kept writing output until the timeout."
                        )
                    raise Exception(append_status(text, status))
                raise

            snapshot = await finish_output()
            output_text, details = format_output(snapshot)
            if exit_code is None:
                raise Exception(append_status(output_text, "Command terminated without an exit code"))
            wall_time_seconds = _round_tenths(clock.monotonic() - started_at)
            full_output = await output.read_full_output(STRUCTURED_OUTPUT_MAX_BYTES)
            structured_content: dict[str, Any] = {"output": full_output.content, "truncated": full_output.truncated}
            if full_output.truncated and snapshot.full_output_path:
                structured_content["full_output_path"] = snapshot.full_output_path
            structured_content["exit_code"] = exit_code
            structured_content["wall_time_seconds"] = wall_time_seconds
            if exit_code != 0:
                return AgentToolResult(
                    content=[TextContent(text=append_status(output_text, f"Command exited with code {exit_code}"))],
                    details=details,
                    structured_content=structured_content,
                    is_error=True,
                )
            return AgentToolResult(
                content=[TextContent(text=output_text)], details=details, structured_content=structured_content
            )
        finally:
            clear_update_timer()

    renderers = create_shell_renderers(config.prompt)
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
        output_schema=BASH_OUTPUT_SCHEMA,
        constrained_sampling=JsonSchemaConstrainedSampling(strict="prefer"),
        execute=execute,
        render_call=renderers.render_call,
        render_result=renderers.render_result,
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
