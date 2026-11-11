"""Mirror of pi mcp src/oauth/sign-in.ts.

Sign-in to an OAuth-protected MCP server, in two halves so an application
can complete it in another request with the same state store:
`start_mcp_sign_in` refreshes the stored tokens when it can, or else
prepares the browser step and returns the authorization URL;
`complete_mcp_sign_in` exchanges the authorization code. The code arrives
through a loopback callback server (`loopback.py`), the application's own
https callback route, or a redirect URL the user pastes.

pi's `signal` is `cancel`, a `CancelToken`, handed to the flow
(`OAuthFlowOptions.cancel`), whose fetch runs each request under
`run_cancellable`: only the requests are torn, the store's saves between
them always complete. pi's `untilAborted` is the same mechanism as the auth
provider's shared refresh: the locked refresh runs detached and settles an
Event, and the sign-in waits for it or for the token, whichever comes first.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Literal, NotRequired, TypedDict

import tonio.colored as tonio

from pidrei_utils.cancel import CancelToken

from ..url import parse_url
from .auth_provider import refresh_stored_tokens
from .callback import OAuthCallback
from .client_settings import (
    McpOAuthClientOptions,
    create_oauth_client_provider,
    offers_client_metadata_document,
    registered_redirect_urls,
    timed_fetch,
)
from .errors import McpRefreshLockBusyError
from .flow import OAuthFlowOptions, OAuthFlowResult, authorize_mcp, step_up_scope
from .provider import McpOAuthState
from .types import OAuthChallenge


@dataclass(frozen=True, slots=True, kw_only=True)
class McpSignInOptions(McpOAuthClientOptions):
    """Options of both halves of a sign-in. Pass the same values to
    `start_mcp_sign_in` and `complete_mcp_sign_in`."""

    # Redirect URI chosen by the caller: a loopback one
    # (`listen_for_loopback_sign_in`) or an https callback route of the
    # application. Pass the same one to both halves: the authorization
    # request may use a variant specific to the MCP server
    # (`McpSignInRedirect.redirectUrl`), which the flow derives from it.
    redirect_url: str
    # The server's challenge that asked for the sign-in; its resource metadata URL and scope apply.
    challenge: OAuthChallenge | None = None
    # Cancelling stops the sign-in at once with `McpSignInCancelledError`,
    # also while `start_mcp_sign_in` waits for the store's refresh lock. A
    # refresh already sent still completes and saves its tokens.
    cancel: CancelToken | None = None


class McpSignInRedirect(TypedDict):
    """The browser step of a sign-in, from `start_mcp_sign_in`. Plain data, so
    an application can persist it."""

    kind: Literal["redirect"]
    # Open in the user's browser.
    authorizationUrl: str
    # The redirect URI of the authorization request; the authorization response must arrive on it.
    redirectUrl: str
    # The `state` parameter the authorization response must carry.
    state: str
    # The scope requested, if any. Saved with the tokens when the token response names none.
    scope: NotRequired[str]


class _McpSignInAuthorized(TypedDict):
    kind: Literal["authorized"]


# Result of `start_mcp_sign_in`: signed in by refreshing the stored tokens, or the browser step.
type McpSignInStart = _McpSignInAuthorized | McpSignInRedirect

# An authorization response: from a callback server, a callback route, or `response_from_redirect_url`.
type McpAuthorizationResponse = OAuthCallback


class McpSignInCancelledError(Exception):
    """The user or the application cancelled a sign-in (its `cancel` fired, or the user gave up)."""

    def __init__(self) -> None:
        super().__init__("Sign-in cancelled")


async def start_mcp_sign_in(options: McpSignInOptions) -> McpSignInStart:
    """First half of a sign-in. Without a step-up (`insufficient_scope`), it
    first tries to refresh the stored tokens with the stored client, whatever
    `redirect_url` is (`authorized`). The refresh takes the store's refresh
    lock, so never call this while holding it; when another process holds
    the lock for the whole wait (`McpRefreshLockBusyError`), the refresh is
    skipped, and any other failure of the lock fails the sign-in. Otherwise it
    prepares the browser step (`redirect`): a fresh `state`, and a stored
    client that cannot use `redirect_url` is dropped with its tokens. Requests
    to the authorization server are time-limited."""
    if options.cancel is not None and options.cancel.cancelled:
        raise McpSignInCancelledError()
    # A refresh keeps the granted scope; a server asking for more needs the browser step.
    step_up = (options.challenge or {}).get("error") == "insufficient_scope"
    if not step_up and await _refresh_before_sign_in(options) == "AUTHORIZED":
        return {"kind": "authorized"}
    return await cancellable(options.cancel, lambda: _start_browser_step(options, step_up))


class _LockedRefresh:
    """The outcome of the locked refresh, which runs detached. Set before `done`."""

    __slots__ = ("done", "error", "result")

    def __init__(self) -> None:
        self.done = tonio.Event()
        self.result: OAuthFlowResult = "REDIRECT"
        self.error: Exception | None = None


async def _refresh_before_sign_in(options: McpSignInOptions) -> OAuthFlowResult:
    """Refresh the stored tokens, unless `cimd` is to replace a registered
    client. The refresh uses `redirect_url` when the tokens came from a Client
    ID Metadata Document whose documents list it."""
    store, settings, cancel = options.store, options.settings, options.cancel

    async def locked() -> OAuthFlowResult:
        # The lock may arrive after the sign-in was cancelled: return at once, so it is released.
        if cancel is not None and cancel.cancelled:
            return "REDIRECT"
        state = await store.load()
        if state is None or not (state.get("tokens") or {}).get("refresh_token"):
            return "REDIRECT"
        if settings.client_registration == "cimd" and state.get("clientInformation") and not settings.client_id:
            return "REDIRECT"
        return await refresh_stored_tokens(
            options, state, challenge=options.challenge, redirect_url=options.redirect_url
        )

    refresh = _LockedRefresh()

    async def run() -> None:
        try:
            refresh.result = await store.with_refresh_lock(locked)
        except Exception as error:
            refresh.error = error
        finally:
            refresh.done.set()

    # Detached even without a token: a refresh the server answered must save
    # its tokens, also when the sign-in stops waiting for it.
    tonio.spawn.without_tracking(run())
    if cancel is None:
        await refresh.done.wait()
    else:
        await tonio.Waiter.any(refresh.done, cancel.event)
    if not refresh.done.is_set():
        raise McpSignInCancelledError()
    if refresh.error is not None:
        # Another process is refreshing and took too long. The user signs in instead.
        if isinstance(refresh.error, McpRefreshLockBusyError):
            return "REDIRECT"
        raise refresh.error
    return refresh.result


async def _start_browser_step(options: McpSignInOptions, step_up: bool) -> McpSignInRedirect:
    store, settings, redirect_url, challenge = options.store, options.settings, options.redirect_url, options.challenge
    stored = await store.load()
    if stored:
        await store.save(_state_for_sign_in(stored, options))
    authorization_urls: list[str] = []

    async def on_redirect(url: str) -> None:
        authorization_urls.append(url)

    provider = create_oauth_client_provider(options, redirect_url, on_redirect)
    # A server asking for more scope gets it on top of the configured scope
    # and, since the challenge may list only the missing scopes, on top of the
    # scope granted so far.
    challenged = (challenge or {}).get("scope")
    granted = ((stored or {}).get("tokens") or {}).get("scope")
    scope = _merge_scopes(settings.scope, step_up_scope(granted, challenged) if step_up else challenged)
    await authorize_mcp(provider, replace(_flow_options(options), scope=scope, skip_refresh=True))
    if not authorization_urls:
        raise RuntimeError("OAuth flow did not produce an authorization URL")
    authorization_url = parse_url(authorization_urls[-1])
    requested_scope = authorization_url.search_param("scope")
    chosen_redirect_url = authorization_url.search_param("redirect_uri")
    redirect: McpSignInRedirect = {
        "kind": "redirect",
        "authorizationUrl": authorization_urls[-1],
        # The flow picks the redirect URI, which may be specific to the MCP server.
        "redirectUrl": chosen_redirect_url if chosen_redirect_url is not None else redirect_url,
        "state": await provider.state(),
    }
    if requested_scope:
        redirect["scope"] = requested_scope
    return redirect


def _state_for_sign_in(stored: McpOAuthState, options: McpSignInOptions) -> McpOAuthState:
    """The stored state for a browser step: without the last sign-in's
    `state`, and without a client that cannot use `redirect_url`, along with
    its tokens."""
    settings, redirect_url = options.settings, options.redirect_url
    next_state: McpOAuthState = {key: value for key, value in stored.items() if key != "oauthState"}  # type: ignore[assignment]
    # A registered client cannot use another redirect URI, and its tokens belong to it. `cimd` replaces a
    # registered client; `auto` keeps it. A Client ID Metadata Document is not stored, so tokens without
    # a stored client belong to the document.
    keep_client = settings.client_id or (
        settings.client_registration != "cimd"
        and redirect_url in registered_redirect_urls(stored.get("clientInformation"))
        if stored.get("clientInformation")
        else offers_client_metadata_document(settings, options.client_metadata_documents, redirect_url)
    )
    if not keep_client:
        for key in ("clientInformation", "tokens", "tokensExpireAt"):
            next_state.pop(key, None)  # type: ignore[misc]
    return next_state


def _merge_scopes(*scopes: str | None) -> str | None:
    """Scopes of both lists, each once."""
    merged = list(dict.fromkeys(item for scope in scopes if scope for item in scope.split()))
    return " ".join(merged) if merged else None


async def complete_mcp_sign_in(
    options: McpSignInOptions, *, authorization: McpSignInRedirect, response: McpAuthorizationResponse
) -> None:
    """Second half of a sign-in: exchanges the authorization code of
    `response` and saves the tokens. Pass the options given to
    `start_mcp_sign_in`, including its `redirect_url` (not
    `authorization["redirectUrl"]`). Fails when the response carries another
    `state`, when another sign-in started with the store since, or when the
    exchange would use another redirect URI than the authorization request
    (the options changed). A redirect URL the user pasted goes through
    `response_from_redirect_url` first, which checks its path."""

    async def complete() -> None:
        if response.state != authorization["state"]:
            raise RuntimeError("The authorization response belongs to a different sign-in")
        # The stored PKCE verifier belongs to the latest sign-in; exchanging with it would fail with
        # `invalid_grant`, which discards the stored tokens.
        if ((await options.store.load()) or {}).get("oauthState") != authorization["state"]:
            raise RuntimeError("Another sign-in to this MCP server started since; sign in again")
        await authorize_mcp(
            create_oauth_client_provider(options, options.redirect_url),
            replace(
                _flow_options(options),
                # Saved with the tokens when the token response names no scope, so it must be the requested one.
                scope=authorization.get("scope"),
                authorization_redirect_url=authorization["redirectUrl"],
                authorization_code=response.code,
                iss=response.iss,
            ),
        )

    await cancellable(options.cancel, complete)


def _flow_options(options: McpSignInOptions) -> OAuthFlowOptions:
    """`authorize_mcp` options of a sign-in, without the scope."""
    return OAuthFlowOptions(
        server_url=options.server_url,
        fetch=timed_fetch(),
        cancel=options.cancel,
        resource_metadata_url=(options.challenge or {}).get("resourceMetadataUrl"),
        authorization_server_metadata_url=options.settings.auth_server_metadata_url,
    )


async def cancellable[T](cancel: CancelToken | None, fn: Callable[[], Awaitable[T]]) -> T:
    """Run `fn`, reporting a cancellation of `cancel` as `McpSignInCancelledError`."""
    if cancel is not None and cancel.cancelled:
        raise McpSignInCancelledError()
    try:
        return await fn()
    except Exception:
        # Cancelled requests fail with the token's reason; report them as the cancellation they are.
        if cancel is not None and cancel.cancelled:
            raise McpSignInCancelledError() from None
        raise
