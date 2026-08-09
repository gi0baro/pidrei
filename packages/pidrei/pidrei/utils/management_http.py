"""Mirror of pi coding-agent src/utils/management-http.ts.

pi wraps `fetch`; pidrei's management requests go through the punkreq seam
(`utils/http.shared_client`), so the helper is a GET wrapper rather than a
`fetch` wrapper. pi's `AbortSignal.timeout` overall budget becomes a monotonic
deadline: each attempt is given the remaining budget as its request timeout.

pi's `formatVersionCheckError` is not ported — its only caller is the npm
self-update planner, which pidrei does not have.
"""

import time
from typing import Any


__all__ = ["RETRYABLE_STATUS_CODES", "fetch_with_retry"]

RETRYABLE_STATUS_CODES = frozenset({408, 425, 429, 500, 502, 503, 504})

DEFAULT_MAX_RETRIES = 2


async def fetch_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_on_status: bool = True,
    timeout_ms: float | None = None,
) -> Any:
    """GET a management HTTP resource with a bounded immediate retry.

    This is intentionally a transport-level helper for idempotent management
    requests (version checks and catalogs). It must not be used for agent/model
    operations: those can fail after the request starts and are retried by their
    semantic caller instead.

    When `timeout_ms` is supplied it is the overall budget shared by all
    attempts.
    """
    # lazy: resolved per call so the http seam stays swappable in tests
    from pidrei_ai.utils.http import RequestTimeout, request_timeout, shared_client

    max_retries = max(0, int(max_retries))
    deadline = None if timeout_ms is None or timeout_ms <= 0 else time.monotonic() + timeout_ms / 1000

    attempt = 0
    while True:
        remaining_ms: float | None = None
        if deadline is not None:
            remaining_ms = (deadline - time.monotonic()) * 1000
            if remaining_ms <= 0:
                raise RequestTimeout(f"Timed out after {timeout_ms:.0f}ms: {url}")

        try:
            response = await shared_client().get(url, headers=headers, timeout=request_timeout(remaining_ms))
        except Exception:
            if attempt >= max_retries or (deadline is not None and time.monotonic() >= deadline):
                raise
        else:
            if not (retry_on_status and response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries):
                return response
            # The response is being discarded before a retry; nothing useful to
            # do if releasing it also fails.
            aclose = getattr(response, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    pass

        attempt += 1
