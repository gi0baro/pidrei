"""The HTTP requests the MCP client makes: pi's injectable `McpFetch`.

pi calls the platform `fetch`; here the default goes through the pidrei HTTP
seam (`pidrei_http.http.client_for`) and follows redirects as `fetch` does.
A caller injects its own as anything with the same call shape.

Every request is bounded by `MCP_TIMEOUT`, the limits pi runs with (undici's
defaults): 10 s to connect, 300 s without a byte while waiting for the head or
reading the body, no wait limit for a pooled connection (every request in
flight is bounded, so one frees up) and no total limit (a stream that keeps
delivering is legitimate). `timeout_ms` instead bounds the whole request, for
one that is given up on after a while (pi's `AbortSignal.timeout(...)`).

Nothing here cancels a request: a response is stopped by closing it, which
closes its connection and ends a read parked on it. A network failure is
reported as `pidrei_http.http.TransportError`, fetch's `TypeError`.
"""

from collections.abc import AsyncIterator, Awaitable, Mapping
from typing import Any, Protocol

from pidrei_http import http


MCP_TIMEOUT = http.timeout(connect=10.0, read=300.0, pool=None, total=None)


class McpResponseHeaders(Protocol):
    def get(self, name: str, /) -> str | None:
        """Case-insensitive; repeated fields come back comma-joined, as
        `Headers.get` returns them."""
        ...


class McpResponse(Protocol):
    @property
    def status(self) -> int: ...

    @property
    def headers(self) -> McpResponseHeaders: ...

    def iter_bytes(self) -> AsyncIterator[bytes]: ...

    async def read(self) -> bytes: ...

    async def close(self) -> None:
        """Release the response; one not read to the end is aborted."""
        ...


class McpFetch(Protocol):
    def __call__(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
        timeout_ms: float | None = None,
    ) -> Awaitable[McpResponse]: ...


class _PunkreqResponse:
    __slots__ = ("_response",)

    def __init__(self, response: Any) -> None:
        self._response = response

    @property
    def status(self) -> int:
        return self._response.status_code

    @property
    def headers(self) -> McpResponseHeaders:
        return self._response.headers

    def iter_bytes(self) -> AsyncIterator[bytes]:
        return self._response.iter_bytes()

    async def read(self) -> bytes:
        return await self._response.read()

    async def close(self) -> None:
        await self._response.close()


async def default_fetch(
    url: str,
    *,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    body: bytes | None = None,
    timeout_ms: float | None = None,
) -> McpResponse:
    client = http.client_for(url)
    response = await client.request(
        method,
        url,
        headers=dict(headers) if headers else None,
        content=body,
        timeout=MCP_TIMEOUT if timeout_ms is None else http.oneshot_timeout(timeout_ms),
    )
    return _PunkreqResponse(response)
