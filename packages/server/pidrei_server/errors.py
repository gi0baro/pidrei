"""Server error types (port of pi server `errors.ts`).

`NotImplementedError` deliberately reuses pi's class name even though it
shadows the Python builtin at `from pidrei_server import ...` sites; it is
still an exception subclass, and the protocol boundary depends on its
sanitized code/message pair.
"""

from typing import Literal

from pidrei_protocol import JsonValue


type PiServerOperationErrorCode = Literal["busy", "session_locked", "not_found", "invalid_request", "not_implemented"]

INTERNAL_SERVER_ERROR_MESSAGE = "Internal server error"
NOT_IMPLEMENTED_MESSAGE = "Operation is not implemented"


class PiServerError(Exception):
    """A service/runtime error that can safely cross the protocol boundary."""

    code: PiServerOperationErrorCode
    details: JsonValue | None

    def __init__(self, code: PiServerOperationErrorCode, message: str, details: JsonValue | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class SessionBusyError(PiServerError):
    def __init__(self, message: str = "Session is busy", details: JsonValue | None = None) -> None:
        super().__init__("busy", message, details)


class SessionLockedError(PiServerError):
    def __init__(self, message: str = "Session is locked", details: JsonValue | None = None) -> None:
        super().__init__("session_locked", message, details)


class SessionNotFoundError(PiServerError):
    def __init__(self, message: str = "Session was not found", details: JsonValue | None = None) -> None:
        super().__init__("not_found", message, details)


class NotImplementedError(PiServerError):
    def __init__(self) -> None:
        super().__init__("not_implemented", NOT_IMPLEMENTED_MESSAGE)


class InternalServerError(Exception):
    """An unsafe failure whose cause is retained for reporting but never serialized."""

    cause: object

    def __init__(self, cause: object) -> None:
        super().__init__(INTERNAL_SERVER_ERROR_MESSAGE)
        self.cause = cause
        if isinstance(cause, BaseException):
            self.__cause__ = cause
