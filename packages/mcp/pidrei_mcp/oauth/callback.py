"""Mirror of pi mcp src/oauth/callback.ts.

Served by pidrei-http's loopback callback server (httpunk underneath).
`close()` fails the waiting callbacks, stops accepting and drops every
accepted connection, as the provider flows close theirs. A waiter is
settled only once its page is written, so no answer is cut off.
"""

import threading
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from typing import Any

from pidrei_http.callback_server import (
    CallbackRequest,
    CallbackResponse,
    CallbackServer,
    OneShotValue,
    start_callback_server,
)
from pidrei_utils import timers


_PLAIN_TEXT_HEADERS = {"content-type": "text/plain; charset=utf-8"}
_HTML_HEADERS = {"content-type": "text/html; charset=utf-8", "cache-control": "no-store"}


@dataclass(frozen=True, slots=True)
class OAuthCallback:
    code: str
    state: str
    iss: str | None = None


@dataclass(frozen=True, slots=True)
class OAuthCallbackPage:
    """Outcome shown on the browser page after the redirect."""

    ok: bool
    message: str | None = None
    details: str | None = None


def _plain_text(page: OAuthCallbackPage) -> str:
    if page.ok:
        return "Authorization complete. You may close this window."
    return f"{page.message}\n\n{page.details}" if page.details else page.message or ""


class _PendingCallback:
    __slots__ = ("outcome", "timer")

    def __init__(self) -> None:
        # `(callback, None)` or `(None, error)`.
        self.outcome = OneShotValue()
        self.timer: timers.Timeout | None = None

    async def wait(self) -> OAuthCallback:
        callback, error = await self.outcome.wait()
        if error is not None:
            raise error
        return callback


class OAuthCallbackServer:
    """`callback = await OAuthCallbackServer(...)` listens; pi's `listen()`.

    `host` is the address to listen on (default `127.0.0.1`); `redirect_host`
    the host name in `redirect_url`, for example `localhost` for a client
    registered with it while listening on `127.0.0.1` (default: `host`).
    `render_page` renders the browser page as HTML; the default is a plain
    text message."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        redirect_host: str | None = None,
        port: int | None = None,
        path: str = "/callback",
        timeout_ms: float | None = None,
        render_page: Callable[[OAuthCallbackPage], str] | None = None,
    ) -> None:
        self._host = host
        self._redirect_host = redirect_host if redirect_host is not None else host
        self._port = port
        self._path = path
        self._timeout_ms = timeout_ms if timeout_ms is not None else 5 * 60_000
        self._render_page = render_page
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingCallback] = {}
        self._server: CallbackServer | None = None
        self.redirect_url = ""

    def __await__(self) -> Generator[Any, None, OAuthCallbackServer]:
        return self._start().__await__()

    async def _start(self) -> OAuthCallbackServer:
        server = await start_callback_server(host=self._host, port=self._port or 0, handle=self._handle)
        redirect_host = f"[{self._redirect_host}]" if ":" in self._redirect_host else self._redirect_host
        self.redirect_url = f"http://{redirect_host}:{server.port}{self._path}"
        self._server = server
        return self

    def wait_for_callback(self, state: str) -> Awaitable[OAuthCallback]:
        """Register `state` now, so a redirect arriving before the returned
        wait is awaited is not lost."""
        with self._lock:
            if state in self._pending:
                raise RuntimeError("OAuth state is already pending")
            pending = _PendingCallback()
            self._pending[state] = pending
            pending.timer = timers.Timeout(self._timeout_ms, lambda: self._expire(state, pending))
        return pending.wait()

    def close(self) -> None:
        with self._lock:
            pending = list(self._pending.values())
            self._pending.clear()
            server = self._server
        for entry in pending:
            if entry.timer is not None:
                entry.timer.cancel()
            entry.outcome.settle((None, RuntimeError("OAuth callback server closed")))
        if server is not None:
            server.close()
            server.close_all_connections()

    def _expire(self, state: str, pending: _PendingCallback) -> None:
        with self._lock:
            if self._pending.get(state) is not pending:
                return
            del self._pending[state]
        pending.outcome.settle((None, RuntimeError("OAuth callback timed out")))

    def _reply(
        self, status: int, page: OAuthCallbackPage, after_sent: Callable[[], None] | None = None
    ) -> CallbackResponse:
        if self._render_page is not None:
            return CallbackResponse(status, self._render_page(page), after_sent, _HTML_HEADERS)
        return CallbackResponse(status, _plain_text(page), after_sent, _PLAIN_TEXT_HEADERS)

    async def _handle(self, request: CallbackRequest) -> CallbackResponse:
        if request.path != self._path:
            return self._reply(404, OAuthCallbackPage(ok=False, message="Not found"))
        state = request.get("state")
        with self._lock:
            pending = self._pending.pop(state, None) if state else None
        if not state or pending is None:
            return self._reply(400, OAuthCallbackPage(ok=False, message="Invalid or expired OAuth state"))
        if pending.timer is not None:
            pending.timer.cancel()
        # The waiter is settled once the page is written: it wakes on another
        # thread and may close this server.
        error = request.get("error")
        if error:
            description = request.get("error_description")
            if description is None:
                description = error
            failure = RuntimeError(description)
            return self._reply(
                200,
                OAuthCallbackPage(
                    ok=False, message="Authorization failed. You may close this window.", details=description
                ),
                lambda: pending.outcome.settle((None, failure)),
            )
        code = request.get("code")
        if not code:
            missing = RuntimeError("OAuth callback did not include an authorization code")
            return self._reply(
                400,
                OAuthCallbackPage(ok=False, message="Missing authorization code"),
                lambda: pending.outcome.settle((None, missing)),
            )
        callback = OAuthCallback(code=code, state=state, iss=request.get("iss") or None)
        return self._reply(200, OAuthCallbackPage(ok=True), lambda: pending.outcome.settle((callback, None)))
