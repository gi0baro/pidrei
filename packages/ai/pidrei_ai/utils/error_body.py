"""Port of pi's provider HTTP error normalization (packages/ai/src/utils/error-body.ts).

Endpoints behind a proxy/gateway may return a non-2xx response whose body the
error object doesn't fold into its message. `normalize_provider_error` probes
the known field shapes and returns a struct each adapter composes into its
display string; `message_carries_body` captures the happy path where the
message already contains the body.

pi probes JS-SDK-specific fields (Mistral `statusCode`, openai `status`/`error`,
Bedrock `$metadata`/`$response`); the Python port probes the snake_case
equivalents our adapters and punkreq raise. Bedrock shapes are added with the
Bedrock adapter (PLAN.md Phase 5).
"""

import json
from dataclasses import dataclass
from typing import Any


MAX_PROVIDER_ERROR_BODY_CHARS = 4000


@dataclass(slots=True)
class NormalizedProviderError:
    # `str(error)`, or `safe_json_stringify(error)` for a non-exception value.
    message: str
    # True when `message` already contains the body (no separate body to add).
    message_carries_body: bool
    # HTTP status code, when one could be extracted from the error object.
    status: int | None = None
    # Raw HTTP body reason, already trimmed and truncated to the cap.
    body: str | None = None


def safe_json_stringify(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(value)


def truncate_error_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _extract_status(error: BaseException) -> int | None:
    # First numeric hit wins: `status_code` -> `status`.
    for attribute in ("status_code", "status"):
        value = getattr(error, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_plain_non_empty_object(value: Any) -> bool:
    """Only a PLAIN object counts as an HTTP body. Error fields can hold class
    instances instead of parsed bodies, and stringifying one produces
    internals-noise which then REPLACES `str(error)` in the composed display
    string — the one place the real deserialized exception text lives. A class
    instance yields no body, `message_carries_body` stays True, and the real
    message survives. Parsed JSON bodies are plain dicts by construction; pi's
    `Object.getPrototypeOf(value) === Object.prototype` check maps to an exact
    `dict` type check (subclasses are wrapper classes, not parsed JSON).
    """
    return type(value) is dict and len(value) > 0


def _pick_body_text(error: BaseException) -> str | None:
    body = getattr(error, "body", None)
    if isinstance(body, str):
        return body
    error_field = getattr(error, "error", None)
    if _is_plain_non_empty_object(error_field):
        return safe_json_stringify(error_field)
    return None


def _extract_body(error: BaseException) -> str | None:
    body_text = _pick_body_text(error)
    if body_text is None:
        return None
    trimmed = body_text.strip()
    if not trimmed:
        return None
    return truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)


def normalize_provider_error(error: Any) -> NormalizedProviderError:
    if not isinstance(error, BaseException):
        return NormalizedProviderError(message=safe_json_stringify(error), message_carries_body=False)

    message = str(error)
    status = _extract_status(error)
    body = _extract_body(error)
    message_carries_body = body is None or body in message

    return NormalizedProviderError(
        message=message,
        message_carries_body=message_carries_body,
        status=status,
        body=body,
    )


def format_provider_error(norm: NormalizedProviderError, prefix: str | None = None) -> str:
    """Compose a display string from a normalized error.

    - no prefix: `"<status>: <body>"`
    - prefix:    `"<prefix> (<status>): <body>"`

    When the message already carries the body or no body/status was extracted,
    the message is returned (still prefixed when a status exists).
    """
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({norm.status}): {norm.message}"
        return norm.message
    if prefix is not None:
        return f"{prefix} ({norm.status}): {norm.body}"
    return f"{norm.status}: {norm.body}"
