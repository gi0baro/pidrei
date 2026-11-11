"""Mirror of pi mcp src/oauth/loopback.ts.

How the authorization response of a sign-in reaches a local application: a
loopback callback server the browser redirects to, or a redirect URL the
user pastes from the browser address bar.

pi's `waitForLoopbackSignIn` registers the callback wait and races it in one
turn, after the caller showed the authorization URL. Here the browser runs
on its own and may reach the callback before the waiting coroutine runs
again, so `wait_for_loopback_sign_in` is a plain function: its synchronous
prefix registers the callback wait, and the caller calls it before sending
the browser to the authorization page and awaits what it returns after. The
race is two detached coroutines settling a first outcome.
"""

import threading
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import tonio.colored as tonio

from pidrei_utils.cancel import CancelToken, combine_cancel_tokens

from ..url import parse_url
from .callback import OAuthCallback, OAuthCallbackPage, OAuthCallbackServer
from .client_settings import (
    McpClientMetadataDocuments,
    McpOAuthSettings,
    loopback_settings,
    offers_client_metadata_document,
    registered_redirect_urls,
    server_specific_path,
)
from .provider import McpOAuthStateStore
from .sign_in import McpAuthorizationResponse, McpSignInCancelledError, McpSignInRedirect, cancellable


@dataclass(frozen=True, slots=True)
class McpLoopbackCallback:
    """A loopback callback server and the redirect URI it serves."""

    callback: OAuthCallbackServer
    # Pass to `start_mcp_sign_in` and `complete_mcp_sign_in`.
    redirect_url: str


async def listen_for_loopback_sign_in(
    *,
    server_url: str,
    store: McpOAuthStateStore,
    settings: McpOAuthSettings,
    client_metadata_documents: McpClientMetadataDocuments | None = None,
    render_page: Callable[[OAuthCallbackPage], str] | None = None,
) -> McpLoopbackCallback:
    """Listen for the authorization response of a sign-in on loopback, where
    `settings` say: `callback_url` and `callback_port`, else the port of the
    stored client's redirect URI (so that client stays valid) or a free one.
    Pass the sign-in's `client_metadata_documents`, so the callback server
    also serves the server-specific redirect URI. `render_page` renders the
    browser page after the redirect (default: plain text). Close the callback
    server when the sign-in ends."""
    loopback = loopback_settings(settings)
    registered = registered_redirect_urls(((await store.load()) or {}).get("clientInformation"))
    port = loopback.port
    if port is None and registered:
        registered_port = parse_url(registered[0]).port
        port = int(registered_port) if registered_port and int(registered_port) else None
    offers_document = offers_client_metadata_document(settings, client_metadata_documents, loopback.url)
    # The redirect URI of a server-specific Client ID Metadata Document.
    extra_paths = [server_specific_path(loopback.path, server_url)] if offers_document else []

    def server(port: int | None) -> OAuthCallbackServer:
        return OAuthCallbackServer(
            host=loopback.host,
            redirect_host=loopback.redirect_host,
            path=loopback.path,
            extra_paths=extra_paths,
            port=port,
            render_page=render_page,
        )

    try:
        callback = await server(port if port is not None else 0)
    except Exception:
        # A configured port is required; the stored client's port is only preferred.
        if loopback.port is not None or port is None:
            raise
        callback = await server(None)
    redirect_url = loopback.fixed_redirect_url if loopback.fixed_redirect_url is not None else callback.redirect_url
    return McpLoopbackCallback(callback=callback, redirect_url=redirect_url)


class _FirstOutcome:
    """The first of several racing outcomes; later ones are dropped."""

    __slots__ = ("_guard", "done", "error", "value")

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.done = tonio.Event()
        self.value: Any = None
        self.error: Exception | None = None

    def settle(self, value: Any = None, error: Exception | None = None) -> None:
        with self._guard:
            if self.done.is_set():
                return
            self.value, self.error = value, error
            self.done.set()


def wait_for_loopback_sign_in(
    *,
    callback: OAuthCallbackServer,
    authorization: McpSignInRedirect,
    prompt_for_redirect_url: Callable[[CancelToken], Awaitable[str | None]] | None = None,
    cancel: CancelToken | None = None,
) -> Coroutine[Any, Any, McpAuthorizationResponse]:
    """Wait for the authorization response on a loopback callback server, or
    for a redirect URL the user pastes from the browser address bar,
    whichever comes first. Cancelling `cancel` stops the wait with
    `McpSignInCancelledError`.

    The callback wait is registered when this is called; await the result.
    `prompt_for_redirect_url` asks for the redirect URL, for when the browser
    cannot reach the loopback callback (for example over SSH). Its token is
    cancelled once the callback arrives or the sign-in is cancelled.
    Resolving to None or an empty string cancels the sign-in."""
    outcome = _FirstOutcome()
    from_browser = callback.wait_for_callback(authorization["state"], parse_url(authorization["redirectUrl"]).pathname)

    async def browser() -> None:
        try:
            outcome.settle(await from_browser)
        except Exception as error:
            outcome.settle(error=error)

    tonio.spawn.without_tracking(browser())
    return _race(outcome, authorization, prompt_for_redirect_url, cancel)


async def _race(
    outcome: _FirstOutcome,
    authorization: McpSignInRedirect,
    prompt_for_redirect_url: Callable[[CancelToken], Awaitable[str | None]] | None,
    cancel: CancelToken | None,
) -> McpAuthorizationResponse:
    local = CancelToken()
    combined = combine_cancel_tokens(local, cancel)
    prompt_cancel = combined.token if combined.token is not None else local

    async def user(prompt: Callable[[CancelToken], Awaitable[str | None]]) -> None:
        try:
            text = await prompt(prompt_cancel)
            if not text or not text.strip():
                raise McpSignInCancelledError()
            outcome.settle(response_from_redirect_url(text, authorization))
        except Exception as error:
            outcome.settle(error=error)

    async def first() -> McpAuthorizationResponse:
        if prompt_for_redirect_url is not None:
            tonio.spawn.without_tracking(user(prompt_for_redirect_url))
        if cancel is None:
            await outcome.done.wait()
        else:
            await tonio.Waiter.any(outcome.done, cancel.event)
        if not outcome.done.is_set():
            raise McpSignInCancelledError()
        if outcome.error is not None:
            raise outcome.error
        return outcome.value

    try:
        return await cancellable(cancel, first)
    finally:
        # The losing side ends once the prompt is cancelled or the callback server closes.
        local.cancel()
        combined.cleanup()


def response_from_redirect_url(text: str, authorization: McpSignInRedirect) -> McpAuthorizationResponse:
    """The authorization response in a redirect URL the user pasted from the
    browser address bar. Fails when the URL is not on the sign-in's redirect
    URI, carries another `state`, or reports an error. Only the `state` and
    `redirectUrl` of `authorization` are read."""
    try:
        url = parse_url(text.strip())
    except ValueError:
        raise RuntimeError("Expected the full redirect URL from the browser address bar") from None
    redirect_url = parse_url(authorization["redirectUrl"])
    # A server-specific redirect URI tells authorization servers apart, so it must match exactly.
    if url.origin != redirect_url.origin or url.pathname != redirect_url.pathname:
        raise RuntimeError("The redirect URL does not match this sign-in's redirect URI")
    error = url.search_param("error")
    if error:
        description = url.search_param("error_description")
        raise RuntimeError(description if description is not None else error)
    state = url.search_param("state")
    if state != authorization["state"]:
        raise RuntimeError("The redirect URL belongs to a different sign-in")
    code = url.search_param("code")
    if not code:
        raise RuntimeError("The redirect URL does not contain an authorization code")
    return OAuthCallback(code=code, state=state, iss=url.search_param("iss") or None)
