"""Port of pi's diagnostics helpers (packages/ai/src/utils/diagnostics.ts)."""

import traceback
from typing import Any

from pidrei_ai.types import AssistantMessageDiagnostic, DiagnosticErrorInfo
from pidrei_http import http
from pidrei_utils import clock


def format_thrown_value(value: Any) -> str:
    if isinstance(value, BaseException):
        return str(value) or type(value).__name__
    if isinstance(value, str):
        return value
    return str(value)


def extract_diagnostic_error(error: Any) -> DiagnosticErrorInfo:
    if not isinstance(error, BaseException):
        return DiagnosticErrorInfo(name="ThrownValue", message=format_thrown_value(error))
    code = getattr(error, "code", None)
    stack = "".join(traceback.format_exception(error)) if error.__traceback__ is not None else None
    return DiagnosticErrorInfo(
        name=type(error).__name__,
        message=str(error) or type(error).__name__,
        stack=stack,
        code=code if isinstance(code, str | int) else None,
    )


def create_assistant_message_diagnostic(
    type: str,
    error: Any,
    details: dict[str, Any] | None = None,
) -> AssistantMessageDiagnostic:
    return AssistantMessageDiagnostic(
        type=type,
        timestamp=clock.now_ms(),
        error=extract_diagnostic_error(error),
        details=details,
    )


def append_assistant_message_diagnostic(message: Any, diagnostic: AssistantMessageDiagnostic) -> None:
    message.diagnostics = [*(message.diagnostics or []), diagnostic]


# The diagnostic of a response that failed in the exchange itself: connect,
# read, write, protocol, proxy or timeout. pi learns this from the error text
# (its SDKs' "Connection error.", undici's "terminated"); here the adapter
# still holds the exception when it reports the failure, so the class decides
# and `is_retryable_assistant_error` reads the diagnostic.
NETWORK_ERROR_DIAGNOSTIC = "network_error"


def append_network_error_diagnostic(message: Any, error: BaseException) -> None:
    """Mark `message` as failed by a network error, if `error` is one.

    Called by an adapter's error arm once the stop reason is set: an aborted
    request is not a failure. The diagnostic keeps the transport's own error
    (class, text, stack), which the reported message leaves out.
    """
    if message.stop_reason == "error" and isinstance(error, http.TransportError):
        append_assistant_message_diagnostic(
            message, create_assistant_message_diagnostic(NETWORK_ERROR_DIAGNOSTIC, error)
        )
