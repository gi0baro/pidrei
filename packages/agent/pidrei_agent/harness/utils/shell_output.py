"""Streaming shell-output capture (port of pi `harness/utils/shell-output.ts`).

Captures combined stdout/stderr through the execution env with tail
truncation, live progress snapshots, and an overflow temp file holding the
full output.

pi serializes temp-file creation and appends on a promise chain that runs
concurrently with the command; the port uses the tonio-native equivalent — a
single writer task fed by an unbounded channel — with the same ordering
guarantees. Node's single thread also serializes the stdout/stderr `onChunk`
callbacks for free; here they arrive from two reader tasks, so the capture
state is guarded by a reentrant lock.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import tonio.colored as tonio
from tonio.colored.sync import channel

from pidrei_ai.utils.cancel import CancelToken

from ..types import ExecutionError, Result, ShellExecOptions, err, ok, to_error
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail, utf8_byte_length


@dataclass(slots=True)
class ShellCaptureProgress:
    output: str
    truncation: TruncationResult
    full_output_path: str | None = None
    last_line_bytes: int = 0


@dataclass(slots=True)
class ShellCaptureOptions:
    cwd: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    inherit_env: bool = True
    timeout: float | None = None
    cancel: CancelToken | None = None
    on_chunk: Callable[[str, Callable[[], ShellCaptureProgress]], None] | None = None
    # Return shell execution failures with captured output instead of as a failed Result.
    return_execution_errors: bool = False


@dataclass(slots=True)
class ShellCaptureResult:
    output: str
    truncation: TruncationResult
    exit_code: int | None
    cancelled: bool
    truncated: bool
    full_output_path: str | None = None
    last_line_bytes: int = 0
    execution_error: ExecutionError | None = None


def _to_execution_error(error: Exception) -> ExecutionError:
    if isinstance(error, ExecutionError):
        return error
    cause = to_error(error)
    return ExecutionError("unknown", str(cause), cause)


def sanitize_binary_output(text: str) -> str:
    return "".join(
        char
        for char in text
        if (code := ord(char)) in (0x09, 0x0A, 0x0D) or (code > 0x1F and not (0xFFF9 <= code <= 0xFFFB))
    )


def trim_to_last_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8", errors="replace")


_WRITE_SENTINEL = object()


async def execute_shell_with_capture(
    env,
    command: str,
    options: ShellCaptureOptions | None = None,
) -> Result[ShellCaptureResult, ExecutionError]:
    options = options if options is not None else ShellCaptureOptions()
    max_output_bytes = DEFAULT_MAX_BYTES * 2

    state_lock = threading.RLock()
    tail_output = ""
    total_bytes = 0
    completed_lines = 0
    has_open_line = False
    current_line_bytes = 0
    full_output_path: str | None = None
    full_output_requested = False
    accepting_output = True
    capture_error: ExecutionError | None = None
    write_error: ExecutionError | None = None

    write_sender, write_receiver = channel.unbounded()

    async def writer() -> None:
        # Single-writer task: the tonio equivalent of pi's ordered promise chain.
        nonlocal full_output_path, write_error
        error: ExecutionError | None = None
        while True:
            item = await write_receiver.receive()
            if item is _WRITE_SENTINEL:
                break
            if error is not None:
                continue
            kind, text = item
            if kind == "ensure":
                temp_file = await env.create_temp_file(prefix="bash-", suffix=".log")
                if not temp_file.ok:
                    error = _to_execution_error(temp_file.error)
                    continue
                with state_lock:
                    full_output_path = temp_file.value
                append_result = await env.append_file(temp_file.value, text)
                if not append_result.ok:
                    error = _to_execution_error(append_result.error)
            else:
                with state_lock:
                    path = full_output_path
                if path is None:
                    error = ExecutionError("unknown", "Full output path was not created")
                    continue
                append_result = await env.append_file(path, text)
                if not append_result.ok:
                    error = _to_execution_error(append_result.error)
        write_error = error

    writer_join = tonio.spawn(writer())

    def append_full_output(text: str) -> None:
        if not full_output_requested or capture_error is not None:
            return
        write_sender.send(("append", text))

    def ensure_full_output_file(initial_content: str) -> None:
        nonlocal full_output_requested
        if full_output_requested or capture_error is not None:
            return
        full_output_requested = True
        write_sender.send(("ensure", initial_content))

    def create_progress() -> ShellCaptureProgress:
        with state_lock:
            tail_truncation = truncate_tail(tail_output)
            total_lines = completed_lines + (1 if has_open_line else 0)
            truncated = total_lines > DEFAULT_MAX_LINES or total_bytes > DEFAULT_MAX_BYTES
            truncation = replace(
                tail_truncation,
                truncated=truncated,
                truncated_by=(
                    (tail_truncation.truncated_by or ("bytes" if total_bytes > DEFAULT_MAX_BYTES else "lines"))
                    if truncated
                    else None
                ),
                total_lines=total_lines,
                total_bytes=total_bytes,
            )
            return ShellCaptureProgress(
                output=truncation.content if truncated else tail_output,
                truncation=truncation,
                full_output_path=full_output_path,
                last_line_bytes=current_line_bytes,
            )

    def on_chunk(chunk: str) -> None:
        nonlocal tail_output, total_bytes, completed_lines, has_open_line, current_line_bytes, capture_error
        with state_lock:
            if not accepting_output:
                return
            try:
                text = sanitize_binary_output(chunk).replace("\r", "")
                text_bytes = utf8_byte_length(text)
                total_bytes += text_bytes
                completed_lines += text.count("\n")
                last_newline = text.rfind("\n")
                if last_newline >= 0:
                    trailing_text = text[last_newline + 1 :]
                    current_line_bytes = utf8_byte_length(trailing_text)
                    has_open_line = len(trailing_text) > 0
                elif text:
                    current_line_bytes += text_bytes
                    has_open_line = True

                tail_output += text
                total_lines = completed_lines + (1 if has_open_line else 0)
                if (total_bytes > DEFAULT_MAX_BYTES or total_lines > DEFAULT_MAX_LINES) and not full_output_requested:
                    ensure_full_output_file(tail_output)
                elif full_output_requested:
                    append_full_output(text)
                tail_output = trim_to_last_utf8_bytes(tail_output, max_output_bytes)
                if options.on_chunk is not None:
                    options.on_chunk(text, create_progress)
            except Exception as error:
                capture_error = _to_execution_error(to_error(error))

    async def finish_writer() -> Result[None, ExecutionError]:
        write_sender.send(_WRITE_SENTINEL)
        await writer_join
        if write_error is not None:
            return err(write_error)
        return ok(None)

    try:
        result = await env.exec(
            command,
            ShellExecOptions(
                cwd=options.cwd,
                env=options.env,
                inherit_env=options.inherit_env,
                timeout=options.timeout,
                cancel=options.cancel,
                on_stdout=on_chunk,
                on_stderr=on_chunk,
            ),
        )
    except Exception as error:
        with state_lock:
            accepting_output = False
        await finish_writer()
        return err(_to_execution_error(to_error(error)))

    with state_lock:
        accepting_output = False
    progress = create_progress()
    if progress.truncation.truncated and not full_output_requested:
        with state_lock:
            ensure_full_output_file(tail_output)
    write_result = await finish_writer()
    if not write_result.ok:
        return write_result
    if capture_error is not None:
        return err(capture_error)
    progress = create_progress()

    if not result.ok:
        if result.error.code == "aborted" or (options.cancel is not None and options.cancel.cancelled):
            return ok(
                ShellCaptureResult(
                    output=progress.output,
                    truncation=progress.truncation,
                    full_output_path=progress.full_output_path,
                    last_line_bytes=progress.last_line_bytes,
                    exit_code=None,
                    cancelled=True,
                    truncated=progress.truncation.truncated,
                )
            )
        if options.return_execution_errors:
            return ok(
                ShellCaptureResult(
                    output=progress.output,
                    truncation=progress.truncation,
                    full_output_path=progress.full_output_path,
                    last_line_bytes=progress.last_line_bytes,
                    exit_code=None,
                    cancelled=False,
                    truncated=progress.truncation.truncated,
                    execution_error=result.error,
                )
            )
        return err(result.error)

    cancelled = options.cancel.cancelled if options.cancel is not None else False
    return ok(
        ShellCaptureResult(
            output=progress.output,
            truncation=progress.truncation,
            full_output_path=progress.full_output_path,
            last_line_bytes=progress.last_line_bytes,
            exit_code=None if cancelled else result.value.exit_code,
            cancelled=cancelled,
            truncated=progress.truncation.truncated,
        )
    )
