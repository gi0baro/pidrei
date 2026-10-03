"""Mirror of pi mcp src/auth-provider.ts."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .fetch import McpFetch, McpResponse


@dataclass(frozen=True, slots=True)
class UnauthorizedContext:
    # The 401 response, or a 403 response whose challenge reports `insufficient_scope`.
    response: McpResponse
    server_url: str
    fetch: McpFetch
    # Access token the rejected request carried, if any. A different current
    # token means another request already refreshed it.
    token: str | None = None


@dataclass(frozen=True, slots=True)
class AuthProvider:
    """Supplies bearer tokens to an MCP HTTP transport and may refresh them
    after a 401 response."""

    token: Callable[[], Awaitable[str | None]]
    on_unauthorized: Callable[[UnauthorizedContext], Awaitable[None]] | None = None
