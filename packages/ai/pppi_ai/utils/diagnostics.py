"""Port of pi's diagnostics helpers (packages/ai/src/utils/diagnostics.ts)."""

import time
import traceback
from typing import Any

from pppi_ai.types import AssistantMessageDiagnostic, DiagnosticErrorInfo


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
        timestamp=int(time.time() * 1000),
        error=extract_diagnostic_error(error),
        details=details,
    )


def append_assistant_message_diagnostic(message: Any, diagnostic: AssistantMessageDiagnostic) -> None:
    message.diagnostics = [*(message.diagnostics or []), diagnostic]
