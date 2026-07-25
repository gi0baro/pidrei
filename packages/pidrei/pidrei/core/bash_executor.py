"""Mirror of pi coding-agent src/core/bash-executor.ts.

Bash command execution with streaming support and cancellation, used by
AgentSession.executeBash and modes that need direct bash execution.
"""

import codecs
import os
import secrets
import tempfile
from dataclasses import dataclass
from typing import Any, BinaryIO

from ..utils.ansi import strip_ansi
from ..utils.shell import sanitize_binary_output
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
    temp_file: BinaryIO | None = None
    total_bytes = 0

    def ensure_temp_file() -> None:
        nonlocal temp_file_path, temp_file
        if temp_file_path is not None:
            return
        temp_file_path = os.path.join(tempfile.gettempdir(), f"pidrei-bash-{secrets.token_hex(8)}.log")
        temp_file = open(temp_file_path, "wb")  # noqa: SIM115
        temp_file.writelines(chunk.encode("utf-8", "replace") for chunk in output_chunks)

    decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def on_data(data: bytes) -> None:
        nonlocal total_bytes, output_bytes
        total_bytes += len(data)

        # Sanitize: strip ANSI, replace binary garbage, normalize newlines
        text = sanitize_binary_output(strip_ansi(decoder.decode(data))).replace("\r", "")

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

    def settle_output() -> tuple[str, Any]:
        full_output = "".join(output_chunks)
        truncation_result = truncate_tail(full_output)
        if truncation_result.truncated:
            ensure_temp_file()
        if temp_file is not None:
            temp_file.close()
        return full_output, truncation_result

    try:
        result = await operations.exec(command, cwd, on_data=on_data, cancel=cancel)

        full_output, truncation_result = settle_output()
        cancelled = cancel.cancelled if cancel is not None else False

        return BashResult(
            output=truncation_result.content if truncation_result.truncated else full_output,
            exit_code=None if cancelled else result.exit_code,
            cancelled=cancelled,
            truncated=truncation_result.truncated,
            full_output_path=temp_file_path,
        )
    except Exception:
        # Check if it was an abort
        if cancel is not None and cancel.cancelled:
            full_output, truncation_result = settle_output()
            return BashResult(
                output=truncation_result.content if truncation_result.truncated else full_output,
                exit_code=None,
                cancelled=True,
                truncated=truncation_result.truncated,
                full_output_path=temp_file_path,
            )

        if temp_file is not None:
            temp_file.close()

        raise
