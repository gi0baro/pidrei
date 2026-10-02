"""Mirror of pi's mcp-oauth-refresh.test.ts.

The fake authorization server is `mcp_oauth_server.py`; the browser that
follows the authorization redirect is `oauth_servers.browse`. The prompt for
a pasted redirect URL waits (bounded) until sign-in cancels it, as pi's waits
for its abort signal.
"""

import os

import pytest
import tonio.colored as tonio

from pidrei.core.auth_storage import InMemoryAuthStorageBackend
from pidrei.extensions.mcp.oauth import (
    McpOAuthCredentialStore,
    McpOAuthSettings,
    create_mcp_auth_provider,
    sign_in_mcp_server,
)
from pidrei_mcp import UnauthorizedContext, default_fetch
from pidrei_mcp.oauth import OAuthIssuerMismatchError
from pidrei_mcp.url import parse_url
from pidrei_utils.cancel import CancelToken

from .mcp_oauth_server import oauth_mcp_servers


_WAIT_S = 10


class _Unauthorized:
    """pi's `new Response(null, { status: 401 })`."""

    status = 401

    @property
    def headers(self) -> dict[str, str]:
        return {}


class _Prompt:
    def __init__(self, browse, opened: list[str] | None = None) -> None:
        self._browse = browse
        self._opened = opened

    def show_authorization_url(self, url: str) -> None:
        if self._opened is not None:
            self._opened.append(url)
        self._browse(url)

    async def prompt_for_redirect_url(self, cancel: CancelToken) -> str | None:
        await cancel.event.wait(_WAIT_S)
        return None


async def _no_settings() -> McpOAuthSettings:
    return McpOAuthSettings()


def _unauthorized(server, token: str) -> UnauthorizedContext:
    return UnauthorizedContext(response=_Unauthorized(), server_url=server.url, fetch=default_fetch, token=token)


async def _signed_in(oauth_servers, tmp_path):
    server = await oauth_servers.start()
    lock_dir = str(tmp_path / "locks")
    os.mkdir(lock_dir)
    # Stores sharing the credential file and lock directory stand in for separate pidrei processes.
    backend = InMemoryAuthStorageBackend()

    def process():
        store = McpOAuthCredentialStore(backend, lock_dir).for_server("test", server.url)
        provider = create_mcp_auth_provider(
            server_url=server.url, store=store, settings=_no_settings, on_challenge=lambda _challenge: None
        )
        return store, provider

    await sign_in_mcp_server(
        server_url=server.url,
        store=process()[0],
        settings=McpOAuthSettings(),
        prompt=_Prompt(oauth_servers.browse),
    )
    return server, lock_dir, process


@pytest.mark.tonio
async def test_refreshes_once_when_several_processes_find_the_same_token_rejected(monkeypatch, tmp_path):
    async with oauth_mcp_servers(monkeypatch) as oauth_servers:
        server, lock_dir, process = await _signed_in(oauth_servers, tmp_path)
        server.expire_access_tokens()
        processes = [process(), process(), process()]

        # The server rotates refresh tokens: a second refresh with refresh-1 would fail with invalid_grant.
        refreshes = [
            tonio.spawn(provider.on_unauthorized(_unauthorized(server, "access-1"))) for _, provider in processes
        ]
        for refresh in refreshes:
            await refresh

        assert [entry for entry in server.log if entry == "token refresh"] == ["token refresh"]
        for _, provider in processes:
            assert await provider.token() == "access-2"
        # The lock is released.
        assert os.listdir(lock_dir) == []


@pytest.mark.tonio
async def test_waits_for_a_running_refresh_to_save_the_new_tokens(monkeypatch, tmp_path):
    async with oauth_mcp_servers(monkeypatch) as oauth_servers:
        server, _lock_dir, process = await _signed_in(oauth_servers, tmp_path)
        server.expire_access_tokens()
        store, provider = process()

        refresh = provider.on_unauthorized(_unauthorized(server, "access-1"))
        await provider.settled()
        assert (await store.load())["tokens"]["access_token"] == "access-2"
        await refresh


async def _sign_in(oauth_servers, *, iss: str | None = None, settings=lambda _url: McpOAuthSettings()) -> None:
    server = await oauth_servers.start(iss=iss)
    await sign_in_mcp_server(
        server_url=server.url,
        store=McpOAuthCredentialStore(InMemoryAuthStorageBackend()).for_server("test", server.url),
        settings=settings(server.url),
        prompt=_Prompt(oauth_servers.browse),
    )


@pytest.mark.tonio
async def test_rejects_an_authorization_response_from_another_issuer(monkeypatch):
    async with oauth_mcp_servers(monkeypatch) as oauth_servers:
        with pytest.raises(OAuthIssuerMismatchError):
            await _sign_in(oauth_servers, iss="https://attacker.example")


@pytest.mark.tonio
async def test_keeps_the_granted_scope_when_the_server_asks_for_more(monkeypatch):
    async with oauth_mcp_servers(monkeypatch) as oauth_servers:
        server = await oauth_servers.start()
        store = McpOAuthCredentialStore(InMemoryAuthStorageBackend()).for_server("test", server.url)
        opened: list[str] = []

        async def sign_in_with(challenge):
            await sign_in_mcp_server(
                server_url=server.url,
                store=store,
                settings=McpOAuthSettings(),
                challenge=challenge,
                prompt=_Prompt(oauth_servers.browse, opened),
            )

        await sign_in_with({"scope": "issues:read"})
        # The token response names no scope, so the grant has the requested one.
        assert (await store.load())["tokens"]["scope"] == "issues:read"
        # The step-up challenge lists only the missing scope. Requesting just that
        # would lose issues:read, so the next request would ask for sign-in again.
        await sign_in_with({"error": "insufficient_scope", "scope": "issues:write"})
        assert [parse_url(url).search_param("scope") for url in opened] == ["issues:read", "issues:read issues:write"]
        assert (await store.load())["tokens"]["scope"] == "issues:read issues:write"


@pytest.mark.tonio
async def test_uses_the_configured_authorization_server_metadata_url(monkeypatch):
    async with oauth_mcp_servers(monkeypatch) as oauth_servers:
        # #10172
        def settings(server_url: str) -> McpOAuthSettings:
            return McpOAuthSettings(auth_server_metadata_url=server_url.removesuffix("/mcp") + "/missing")

        with pytest.raises(Exception, match="HTTP 404 loading authorization server metadata"):
            await _sign_in(oauth_servers, settings=settings)
