"""Port of pi's models error (packages/ai/src/utils/models-error.ts)."""

from pidrei_ai.utils.diagnostics import format_thrown_value


type ModelsErrorCode = str  # "model_source" | "model_validation" | "provider" | "stream" | "auth" | "oauth"


class ModelsError(Exception):
    def __init__(self, code: ModelsErrorCode, message: str, *, cause: BaseException | object | None = None):
        super().__init__(_with_cause_detail(message, cause))
        self.code = code
        if isinstance(cause, BaseException):
            self.__cause__ = cause


def _with_cause_detail(message: str, cause: BaseException | object | None) -> str:
    """Callers surface `str(error)` only, so keep the underlying reason in it."""
    if cause is None:
        return message
    detail = format_thrown_value(cause).strip()
    if not detail or detail in message:
        return message
    return f"{message}: {detail}"
