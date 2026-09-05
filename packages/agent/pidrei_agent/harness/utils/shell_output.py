"""Compatibility collector over bounded shell output (port of pi `harness/utils/shell-output.ts`).

Source-side capture, adaptive publication and spilling are owned by the
execution environment (`OutputCapture`); this wraps `env.exec` for callers
that need one bounded final view plus pi's older `on_chunk` progress callback.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from pidrei_ai.utils.cancel import CancelToken

from ..types import (
    ExecutionError,
    Result,
    ShellExecOptions,
    ShellOutputCaptureOptions,
    ShellOutputLimits,
    ShellOutputTruncation,
    ShellOutputView as _ShellOutputView,
    err,
    ok,
)
from .output_capture import apply_shell_output_update, sanitize_shell_output
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, TruncationResult, truncate_tail


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
    # (chunk, get_progress, cancel) — pi passes the call's context last.
    on_chunk: Callable[[str, Callable[[], ShellCaptureProgress], CancelToken | None], None] | None = None
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


# pi: `export { sanitizeShellOutput as sanitizeBinaryOutput }`.
sanitize_binary_output = sanitize_shell_output


def _progress_from(output: _ShellOutputView) -> ShellCaptureProgress:
    truncation = output.truncation
    return ShellCaptureProgress(
        output=output.text,
        truncation=TruncationResult(
            content=output.text,
            truncated=truncation.truncated,
            truncated_by=truncation.truncated_by,
            total_lines=truncation.total_lines,
            total_bytes=truncation.total_bytes,
            output_lines=truncation.output_lines,
            output_bytes=truncation.output_bytes,
            last_line_partial=truncation.last_line_partial,
            first_line_exceeds_limit=truncation.first_line_exceeds_limit,
            max_lines=truncation.max_lines,
            max_bytes=truncation.max_bytes,
        ),
        full_output_path=output.spill_path,
        last_line_bytes=output.last_line_bytes if output.last_line_bytes is not None else 0,
    )


async def execute_shell_with_capture(
    env,
    command: str,
    options: ShellCaptureOptions | None = None,
    cancel: CancelToken | None = None,
) -> Result[ShellCaptureResult, ExecutionError]:
    options = options if options is not None else ShellCaptureOptions()
    output: _ShellOutputView | None = None

    def on_update(update, update_cancel) -> None:
        nonlocal output
        previous = output
        output = apply_shell_output_update(output, update)
        if update.kind in ("append", "slide"):
            chunk: str | None = update.text
        elif update.kind == "replace" and previous is None:
            chunk = output.text
        else:
            chunk = None
        # A metadata-only update and a post-cap replacement contain no new
        # incremental chunk. Reporting their complete view would duplicate bytes
        # for callers that accumulate this compatibility callback.
        if chunk and options.on_chunk is not None:
            current = output
            options.on_chunk(chunk, lambda: _progress_from(current), update_cancel)

    result = await env.exec(
        command,
        ShellExecOptions(
            cwd=options.cwd,
            env=options.env,
            inherit_env=options.inherit_env,
            timeout=options.timeout,
            capture=ShellOutputCaptureOptions(
                limits=ShellOutputLimits(max_bytes=DEFAULT_MAX_BYTES, max_lines=DEFAULT_MAX_LINES, retain="tail"),
                spill=True,
            ),
            on_update=on_update,
        ),
        cancel,
    )

    if output is None:
        empty = truncate_tail("")
        output = _progress_view(empty)
    progress = _progress_from(output)
    if not result.ok:
        if result.error.code == "aborted" or (cancel is not None and cancel.cancelled):
            return ok(_capture_result(progress, exit_code=None, cancelled=True))
        if options.return_execution_errors:
            return ok(_capture_result(progress, exit_code=None, cancelled=False, execution_error=result.error))
        return err(result.error)
    return ok(
        ShellCaptureResult(
            output=progress.output,
            truncation=progress.truncation,
            full_output_path=progress.full_output_path,
            last_line_bytes=progress.last_line_bytes,
            exit_code=result.value.exit_code,
            cancelled=False,
            truncated=result.value.truncation.truncated,
        )
    )


def _progress_view(empty: TruncationResult) -> _ShellOutputView:
    return _ShellOutputView(
        text=empty.content,
        truncation=ShellOutputTruncation(
            truncated=empty.truncated,
            truncated_by=empty.truncated_by,
            total_lines=empty.total_lines,
            total_bytes=empty.total_bytes,
            output_lines=empty.output_lines,
            output_bytes=empty.output_bytes,
            last_line_partial=empty.last_line_partial,
            first_line_exceeds_limit=empty.first_line_exceeds_limit,
            max_lines=empty.max_lines,
            max_bytes=empty.max_bytes,
        ),
    )


def _capture_result(
    progress: ShellCaptureProgress,
    *,
    exit_code: int | None,
    cancelled: bool,
    execution_error: ExecutionError | None = None,
) -> ShellCaptureResult:
    return ShellCaptureResult(
        output=progress.output,
        truncation=progress.truncation,
        full_output_path=progress.full_output_path,
        last_line_bytes=progress.last_line_bytes,
        exit_code=exit_code,
        cancelled=cancelled,
        truncated=progress.truncation.truncated,
        execution_error=execution_error,
    )
