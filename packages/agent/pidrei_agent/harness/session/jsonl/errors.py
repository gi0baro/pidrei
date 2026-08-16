"""JSONL backend error helpers (port of pi `session/jsonl/errors.ts`)."""

from typing import Literal

from ...types import FileError, Result
from ..types import SessionError


class JsonlDecodeError(Exception):
    def __init__(self, kind: Literal["syntax", "schema"], message: str, cause: Exception | None = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        if cause is not None:
            self.__cause__ = cause


def file_result[T](result: Result[T, FileError], message: str) -> T:
    if not result.ok:
        raise SessionError(
            "not_found" if result.error.code == "not_found" else "storage",
            f"{message}: {result.error.message}",
            result.error,
        )
    return result.value


def invalid_file(path: str, line: int, cause: Exception) -> SessionError:
    return SessionError("invalid_entry", f"Invalid JSONL v4 session {path}: line {line} {cause}", cause)
