"""JSONL backend error helpers (port of pi `session/jsonl/errors.ts`)."""

from ...types import FileError, Result
from ..types import SessionError


def file_result[T](result: Result[T, FileError], message: str) -> T:
    if not result.ok:
        raise SessionError(
            "not_found" if result.error.code == "not_found" else "storage",
            f"{message}: {result.error.message}",
            result.error,
        )
    return result.value


def invalid_file(path: str, line: int, message: str, cause: Exception | None = None) -> SessionError:
    return SessionError("invalid_entry", f"Invalid JSONL v4 session {path}: line {line} {message}", cause)
