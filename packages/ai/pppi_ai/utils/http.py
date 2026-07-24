"""The punkreq seam: the only pppi module that imports punkreq.

Adapters obtain clients and HTTP types exclusively from here, so alpha-stage
punkreq API churn stays contained in one file (see PLAN.md). The defaults
encode the LLM-streaming idiom: bound connect and per-chunk reads, never the
whole request — a legitimately long SSE stream must not hit a total deadline.
"""

from collections.abc import Mapping

from punkreq import Limits, Timeout
from punkreq.tonio import Client


STREAMING_TIMEOUT = Timeout(connect=30.0, read=600.0, pool=30.0, total=None)
DEFAULT_LIMITS = Limits(max_connections=64)


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
