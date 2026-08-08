"""Client error types (port of pi client `errors.ts`).

pi's `toError` (wrap non-Error throwables) has no Python counterpart — every
raised object is already an exception — so call sites use the caught exception
directly.
"""

from pidrei_protocol import JsonValue, ProtocolError, ProtocolErrorCode


class PiServerError(Exception):
    code: ProtocolErrorCode
    details: JsonValue | None

    def __init__(self, error: ProtocolError) -> None:
        super().__init__(error["message"])
        self.code = error["code"]
        self.details = error.get("details")


class PiDisconnectedError(Exception):
    def __init__(self, message: str = "Pi client is disconnected") -> None:
        super().__init__(message)


class PiClientDisposedError(Exception):
    def __init__(self) -> None:
        super().__init__("Pi client is disposed")


class PiSessionOwnershipError(Exception):
    session_id: str

    def __init__(self, session_id: str, message: str) -> None:
        super().__init__(message)
        self.session_id = session_id


class PiSessionDetachedError(Exception):
    session_id: str

    def __init__(self, session_id: str) -> None:
        super().__init__(f"Session {session_id} is not attached")
        self.session_id = session_id


def to_disconnected_error(error: BaseException) -> PiDisconnectedError:
    return error if isinstance(error, PiDisconnectedError) else PiDisconnectedError(str(error))
