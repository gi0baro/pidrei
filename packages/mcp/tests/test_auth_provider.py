"""Mirror of pi mcp test/auth-provider.test.ts.

The authorization server runs on `loopback_servers`, entered in each test
body. `settings` is async here (`create_mcp_auth_provider` resolves it only
for a refresh).
"""

from dataclasses import replace

import pytest
from mcp_authorization_server import (
    AuthorizationServer,
    authorize_in_browser,
    memory_store,
    start_authorization_server,
)
from mcp_helpers import loopback_servers

from pidrei_http import http
from pidrei_mcp import UnauthorizedContext, default_fetch
from pidrei_mcp.oauth import (
    McpAuthorizationResponse,
    McpClientMetadataDocuments,
    McpOAuthAuthorizationRequiredError,
    McpOAuthSettings,
    McpSignInOptions,
    McpSignInRedirect,
    OAuthCallback,
    complete_mcp_sign_in,
    create_mcp_auth_provider,
    start_mcp_sign_in,
)
from pidrei_mcp.url import parse_url


ROUTE = "https://app.example/mcp/callback"
LOOPBACK = "http://127.0.0.1:1/callback"


async def _start_redirect(options: McpSignInOptions) -> McpSignInRedirect:
    started = await start_mcp_sign_in(options)
    if started["kind"] != "redirect":
        raise AssertionError("Expected the browser step")
    return started


async def _browse(authorization: McpSignInRedirect) -> McpAuthorizationResponse:
    location = parse_url(await authorize_in_browser(authorization["authorizationUrl"]))
    return OAuthCallback(
        code=location.search_param("code") or "",
        state=location.search_param("state") or "",
        iss=location.search_param("iss") or None,
    )


async def _sign_in(options: McpSignInOptions) -> None:
    authorization = await _start_redirect(options)
    await complete_mcp_sign_in(options, authorization=authorization, response=await _browse(authorization))


class _Unauthorized:
    """pi's `new Response(null, { status: 401 })`."""

    status = 401

    @property
    def headers(self) -> dict[str, str]:
        return {}


def _unauthorized(server: AuthorizationServer, token: str, fetch=default_fetch) -> UnauthorizedContext:
    """What a connection's request reports after its token was rejected."""
    return UnauthorizedContext(response=_Unauthorized(), server_url=server.server_url, fetch=fetch, token=token)  # type: ignore[arg-type]


def _options(server: AuthorizationServer, **changes) -> McpSignInOptions:
    options = McpSignInOptions(
        server_url=server.server_url,
        store=memory_store(),
        settings=McpOAuthSettings(),
        redirect_url=ROUTE,
        client_name="app",
    )
    return replace(options, **changes)


def _provider(options: McpSignInOptions, settings: McpOAuthSettings | None = None, **extra):
    async def resolved() -> McpOAuthSettings:
        return settings if settings is not None else McpOAuthSettings()

    return create_mcp_auth_provider(
        server_url=options.server_url,
        store=options.store,
        settings=resolved,
        on_challenge=lambda _challenge: None,
        client_name=options.client_name,
        redirect_url=options.redirect_url,
        **extra,
    )


@pytest.mark.tonio
async def test_refreshes_a_rejected_token(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        await _sign_in(options)
        provider = _provider(options)

        await provider.on_unauthorized(_unauthorized(server, "access-1"))
        assert await provider.token() == "access-2"
        assert server.token_requests[1]["client_id"] == "client-1"


@pytest.mark.tonio
async def test_leaves_a_pending_sign_in_usable_when_the_refresh_fails(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        await _sign_in(options)
        # A step-up keeps the tokens while the user is in the browser.
        step_up = replace(options, challenge={"error": "insufficient_scope", "scope": "more"})
        authorization = await _start_redirect(step_up)
        pending = await options.store.load()

        server.revoke_refresh_tokens()
        provider = _provider(options)
        with pytest.raises(McpOAuthAuthorizationRequiredError):
            await provider.on_unauthorized(_unauthorized(server, "access-1"))

        state = await options.store.load()
        assert state["oauthState"] == pending["oauthState"]
        assert state["codeVerifier"] == pending["codeVerifier"]
        assert len(server.registrations) == 1
        await complete_mcp_sign_in(step_up, authorization=authorization, response=await _browse(authorization))
        assert (await options.store.load())["tokens"]["access_token"] == "access-2"


@pytest.mark.tonio
async def test_does_not_register_a_client_to_refresh(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True, iss_parameter=True)
        options = _options(server, redirect_url=LOOPBACK)
        documents = McpClientMetadataDocuments(
            base_url="https://app.example/oauth", lists_redirect_url=lambda _url: True
        )
        await _sign_in(replace(options, client_metadata_documents=documents))
        assert "clientInformation" not in await options.store.load()

        # Without the documents, the tokens have no client to refresh them with.
        provider = _provider(options)
        with pytest.raises(McpOAuthAuthorizationRequiredError):
            await provider.on_unauthorized(_unauthorized(server, "access-1"))
        assert server.registrations == []
        assert len(server.authorizations) == 1


@pytest.mark.tonio
@pytest.mark.parametrize("client_registration", ["auto", "cimd"])
async def test_refreshes_the_tokens_of_a_document_that_lists_only_the_redirect_uri_passed(
    monkeypatch, client_registration
):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True, iss_parameter=True)
        documents = McpClientMetadataDocuments(
            base_url="https://app.example/oauth", lists_redirect_url=lambda redirect_url: redirect_url == ROUTE
        )
        settings = McpOAuthSettings(client_registration=client_registration)
        options = _options(server, settings=settings, client_metadata_documents=documents)
        await _sign_in(options)
        assert "clientInformation" not in await options.store.load()

        provider = _provider(options, settings, client_metadata_documents=documents)
        await provider.on_unauthorized(_unauthorized(server, "access-1"))
        assert await provider.token() == "access-2"
        assert server.token_requests[1]["client_id"] == "https://app.example/oauth/client.json"
        assert len(server.authorizations) == 1


async def _pending_sign_in(servers):
    """Signed in, with a step-up sign-in waiting for the browser, which keeps the tokens meanwhile."""
    server = await start_authorization_server(servers)
    options = _options(server)
    await _sign_in(options)
    step_up = replace(options, challenge={"error": "insufficient_scope", "scope": "more"})
    await _start_redirect(step_up)
    pending = await options.store.load()
    return server, options.store, pending, _provider(options)


@pytest.mark.tonio
async def test_drops_a_rejected_client_and_its_tokens_but_keeps_the_pending_sign_in(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server, store, pending, provider = await _pending_sign_in(servers)
        server.fail_next_refresh("invalid_client")
        with pytest.raises(McpOAuthAuthorizationRequiredError):
            await provider.on_unauthorized(_unauthorized(server, "access-1"))
        state = await store.load()
        assert state["oauthState"] == pending["oauthState"]
        assert state["codeVerifier"] == pending["codeVerifier"]
        assert "clientInformation" not in state
        assert "tokens" not in state
        assert len(server.registrations) == 1


@pytest.mark.tonio
async def test_keeps_a_client_stored_while_the_rejected_refresh_ran(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server, store, _pending, provider = await _pending_sign_in(servers)

        # Another sign-in completes with a new client while the server rejects the old one.
        async def replace_client() -> None:
            state = await store.load()
            if state:
                await store.save({**state, "clientInformation": {"client_id": "client-new", "redirect_uris": [ROUTE]}})

        server.fail_next_refresh("invalid_client", replace_client)
        await provider.on_unauthorized(_unauthorized(server, "access-1"))
        assert (await store.load())["clientInformation"]["client_id"] == "client-new"
        assert server.token_requests[-1]["client_id"] == "client-new"
        assert await provider.token() == "access-2"


@pytest.mark.tonio
async def test_keeps_the_tokens_and_the_pending_sign_in_after_a_server_error(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server, store, pending, provider = await _pending_sign_in(servers)
        server.fail_next_refresh("server_error")
        with pytest.raises(McpOAuthAuthorizationRequiredError):
            await provider.on_unauthorized(_unauthorized(server, "access-1"))
        state = await store.load()
        assert state["oauthState"] == pending["oauthState"]
        assert state["codeVerifier"] == pending["codeVerifier"]
        assert state["tokens"]["access_token"] == "access-1"


@pytest.mark.tonio
async def test_keeps_the_tokens_and_the_pending_sign_in_after_a_network_error(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server, store, pending, provider = await _pending_sign_in(servers)

        def offline(url, **options):
            if url.endswith("/token"):
                raise http.TransportError("fetch failed")
            return default_fetch(url, **options)

        with pytest.raises(McpOAuthAuthorizationRequiredError):
            await provider.on_unauthorized(_unauthorized(server, "access-1", offline))
        state = await store.load()
        assert state["oauthState"] == pending["oauthState"]
        assert state["codeVerifier"] == pending["codeVerifier"]
        assert state["tokens"]["access_token"] == "access-1"
