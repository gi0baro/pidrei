"""Mirror of pi coding-agent src/core/bash-executor.ts.

Bash command execution with streaming support and cancellation, used by
AgentSession.executeBash and modes that need direct bash execution.
"""

import codecs
import threading
from dataclasses import dataclass
from typing import Any

from ..utils.ansi import split_incomplete_ansi_suffix, strip_ansi
from ..utils.output_files import create_output_file_stream
from ..utils.shell import sanitize_binary_output
from ..utils.temp_file_writer import TempFileWriter
from .tools.truncate import DEFAULT_MAX_BYTES, truncate_tail


@dataclass(slots=True)
class BashResult:
    # Combined stdout + stderr output (sanitized, possibly truncated)
    output: str
    # Process exit code (None if killed/cancelled)
    exit_code: int | None
    # Whether the command was cancelled via signal
    cancelled: bool
    # Whether the output was truncated
    truncated: bool
    # Path to temp file containing full output (if output exceeded truncation threshold)
    full_output_path: str | None = None


async def execute_bash_with_operations(
    command: str,
    cwd: str,
    operations: Any,
    *,
    on_chunk=None,
    cancel=None,
) -> BashResult:
    """Execute a bash command using custom BashOperations."""
    output_chunks: list[str] = []
    output_bytes = 0
    max_output_bytes = DEFAULT_MAX_BYTES * 2

    temp_file_path: str | None = None
    temp_file: TempFileWriter | None = None
    total_bytes = 0

    def ensure_temp_file() -> None:
        nonlocal temp_file_path, temp_file
        if temp_file_path is not None:
            return
        # Channel-backed, so `write` below stays non-blocking on the
        # streaming path — pi uses createWriteStream here for the same reason.
        temp_file_path, temp_file = create_output_file_stream("pidrei-bash", ".log")
        for chunk in output_chunks:
            temp_file.write(chunk.encode("utf-8", "replace"))

    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    # Unfinished escape sequence at the end of the previous chunk, completed by the next chunk.
    pending_ansi = ""
    # Backends may deliver stdout and stderr from parallel readers; the
    # decoder, `pending_ansi` and rolling buffer below assume pi's single-thread ordering.
    data_lock = threading.Lock()

    def on_data(data: bytes) -> None:
        nonlocal total_bytes, pending_ansi
        with data_lock:
            total_bytes += len(data)
            complete, pending_ansi = split_incomplete_ansi_suffix(pending_ansi + decoder.decode(data))
            _append_text(complete)

    def flush_output() -> None:
        nonlocal pending_ansi
        with data_lock:
            rest = pending_ansi + decoder.decode(b"", final=True)
            pending_ansi = ""
            _append_text(rest)

    def _append_text(raw_text: str) -> None:
        nonlocal output_bytes
        # Sanitize: strip ANSI, replace binary garbage, normalize newlines
        text = sanitize_binary_output(strip_ansi(raw_text)).replace("\r", "")
        if not text:
            return

        # Start writing to temp file if exceeds threshold
        if total_bytes > DEFAULT_MAX_BYTES:
            ensure_temp_file()

        if temp_file is not None:
            temp_file.write(text.encode("utf-8", "replace"))

        # Keep rolling buffer
        output_chunks.append(text)
        output_bytes += len(text)
        while output_bytes > max_output_bytes and len(output_chunks) > 1:
            removed = output_chunks.pop(0)
            output_bytes -= len(removed)

        # Stream to callback
        if on_chunk is not None:
            on_chunk(text)

    async def settle_output() -> tuple[str, Any]:
        full_output = "".join(output_chunks)
        truncation_result = truncate_tail(full_output)
        if truncation_result.truncated:
            ensure_temp_file()
        if temp_file is not None:
            # Drains before returning: `full_output_path` is handed back to
            # the caller to read. pi does not wait here; see TempFileWriter.
            await temp_file.close()
        return full_output, truncation_result

    exit_code: int | None = None
    try:
        exit_code = (await operations.exec(command, cwd, on_data=on_data, cancel=cancel)).exit_code
    except Exception:
        # An aborted command still returns the output it produced so far
        if cancel is None or not cancel.cancelled:
            if temp_file is not None:
                await temp_file.close()
            raise

    flush_output()
    full_output, truncation_result = await settle_output()
    cancelled = cancel.cancelled if cancel is not None else False

    return BashResult(
        output=truncation_result.content if truncation_result.truncated else full_output,
        exit_code=None if cancelled else exit_code,
        cancelled=cancelled,
        truncated=truncation_result.truncated,
        full_output_path=temp_file_path,
    )
