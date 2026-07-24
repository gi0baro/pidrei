"""The punkreq seam: the only pppi module that imports punkreq.

Adapters obtain clients and HTTP types exclusively from here, so alpha-stage
punkreq API churn stays contained in one file (see PLAN.md). The defaults
encode the LLM-streaming idiom: bound connect and per-chunk reads, never the
whole request — a legitimately long SSE stream must not hit a total deadline.
"""

import threading
from collections.abc import Mapping

from punkreq import Limits, Timeout
from punkreq.tonio import Client


STREAMING_TIMEOUT = Timeout(connect=30.0, read=600.0, pool=30.0, total=None)
DEFAULT_LIMITS = Limits(max_connections=64)

_shared_client: Client | None = None
_shared_client_guard = threading.Lock()


def shared_client() -> Client:
    """Process-wide pooled client used by the API adapters."""
    global _shared_client
    with _shared_client_guard:
        if _shared_client is None:
            _shared_client = create_client()
        return _shared_client


def request_timeout(timeout_ms: float | None) -> Timeout:
    """Per-request timeout for streaming LLM calls.

    pi forwards `timeoutMs` to the SDK's whole-request timeout; for streaming
    the practical bound is per-chunk idleness, so it maps to `read` here while
    `total` stays disabled (a legitimately long stream must not be cut off).
    """
    if timeout_ms is None:
        return STREAMING_TIMEOUT
    return Timeout(connect=30.0, read=timeout_ms / 1000, pool=30.0, total=None)


def create_client(
    *,
    base_url: str = "",
    headers: Mapping[str, str] | None = None,
    proxy: str | None = None,
    verify: bool = True,
    timeout: Timeout = STREAMING_TIMEOUT,
    limits: Limits = DEFAULT_LIMITS,
) -> Client:
    return Client(
        base_url=base_url,
        headers=headers,
        proxy=proxy,
        verify=verify,
        timeout=timeout,
        limits=limits,
    )
