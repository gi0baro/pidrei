"""Mirror of pi mcp test/loopback.test.ts.

pi closes the callback servers from `afterEach`; here each test closes its
own before the body returns, inside `loopback_servers`.
`wait_for_loopback_sign_in` registers the callback wait when it is called
and returns the rest to await.
"""

import base64
import contextlib
import hashlib

import pytest
from mcp_authorization_server import authorize_in_browser, memory_store, start_authorization_server
from mcp_helpers import loopback_servers

from pidrei_http import http
from pidrei_mcp.oauth import (
    McpClientMetadataDocuments,
    McpLoopbackCallback,
    McpOAuthSettings,
    McpSignInCancelledError,
    McpSignInOptions,
    McpSignInRedirect,
    OAuthCallback,
    complete_mcp_sign_in,
    listen_for_loopback_sign_in,
    response_from_redirect_url,
    start_mcp_sign_in,
    wait_for_loopback_sign_in,
)
from pidrei_utils.cancel import CancelToken


@contextlib.asynccontextmanager
async def _listen_at(server_url: str, *, documents: bool = False):
    loopback: McpLoopbackCallback = await listen_for_loopback_sign_in(
        server_url=server_url,
        store=memory_store(),
        settings=McpOAuthSettings(),
        client_metadata_documents=(
            McpClientMetadataDocuments(base_url="https://app.example/oauth", lists_redirect_url=lambda _url: True)
            if documents
            else None
        ),
    )
    try:
        yield loopback
    finally:
        loopback.callback.close()


def _pending(redirect_url: str) -> McpSignInRedirect:
    return {
        "kind": "redirect",
        "authorizationUrl": "https://auth.example/authorize",
        "redirectUrl": redirect_url,
        "state": "state-1",
    }


# loopback sign-in


@pytest.mark.tonio
async def test_completes_through_the_loopback_callback_server(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        store = memory_store()
        loopback = await listen_for_loopback_sign_in(
            server_url=server.server_url, store=store, settings=McpOAuthSettings()
        )
        try:
            callback, redirect_url = loopback.callback, loopback.redirect_url
            assert redirect_url.startswith("http://127.0.0.1:")
            assert redirect_url.endswith("/callback")

            options = McpSignInOptions(
                server_url=server.server_url,
                store=store,
                settings=McpOAuthSettings(),
                redirect_url=redirect_url,
                client_name="app",
            )
            authorization = await start_mcp_sign_in(options)
            if authorization["kind"] != "redirect":
                raise AssertionError("Expected the browser step")
            response = wait_for_loopback_sign_in(callback=callback, authorization=authorization)
            page = await http.shared_client().get(await authorize_in_browser(authorization["authorizationUrl"]))
            await page.read()
            await complete_mcp_sign_in(options, authorization=authorization, response=await response)

            assert (await store.load())["tokens"]["access_token"] == "access-1"
            [registration] = server.registrations
            assert registration["redirect_uris"] == [redirect_url]
            assert registration["application_type"] == "native"
        finally:
            loopback.callback.close()


@pytest.mark.tonio
async def test_rejects_a_response_on_another_redirect_path(monkeypatch):
    async with loopback_servers(monkeypatch):
        server_url = "https://mcp.example/mcp"
        async with _listen_at(server_url, documents=True) as loopback:
            digest = hashlib.sha256(server_url.encode()).digest()[:9]
            callback_id = base64.urlsafe_b64encode(digest).decode().rstrip("=")
            waiting = wait_for_loopback_sign_in(
                callback=loopback.callback, authorization=_pending(f"{loopback.redirect_url}/{callback_id}")
            )
            # A mixed-up authorization server redirects to the shared callback path.
            page = await http.shared_client().get(f"{loopback.redirect_url}?code=c&state=state-1")
            await page.read()
            with pytest.raises(Exception, match="arrived on another redirect URI"):
                await waiting


@pytest.mark.tonio
async def test_takes_a_pasted_redirect_url_while_waiting_for_the_callback():
    async with _listen_at("https://mcp.example/mcp") as loopback:

        async def prompt(_cancel: CancelToken) -> str:
            return f"{loopback.redirect_url}?code=pasted&state=state-1"

        response = await wait_for_loopback_sign_in(
            callback=loopback.callback, authorization=_pending(loopback.redirect_url), prompt_for_redirect_url=prompt
        )
        assert response == OAuthCallback(code="pasted", state="state-1")


@pytest.mark.tonio
async def test_cancels_the_wait_and_an_empty_paste_cancels_too():
    async with _listen_at("https://mcp.example/mcp") as loopback:
        cancel = CancelToken()
        waiting = wait_for_loopback_sign_in(
            callback=loopback.callback, authorization=_pending(loopback.redirect_url), cancel=cancel
        )
        cancel.cancel()
        with pytest.raises(McpSignInCancelledError):
            await waiting

        async def empty(_cancel: CancelToken) -> str:
            return ""

        pasted = wait_for_loopback_sign_in(
            callback=loopback.callback,
            authorization={**_pending(loopback.redirect_url), "state": "state-2"},
            prompt_for_redirect_url=empty,
        )
        with pytest.raises(McpSignInCancelledError):
            await pasted


# pasted redirect URLs

_AUTHORIZATION = _pending("http://127.0.0.1:1234/callback/server-id")
_REDIRECT_URL = _AUTHORIZATION["redirectUrl"]


def test_reads_the_code_state_and_issuer():
    assert response_from_redirect_url(f" {_REDIRECT_URL}?code=c&state=state-1&iss=x ", _AUTHORIZATION) == (
        OAuthCallback(code="c", state="state-1", iss="x")
    )


def test_rejects_another_state_another_redirect_path_and_errors():
    with pytest.raises(Exception, match="belongs to a different sign-in"):
        response_from_redirect_url(f"{_REDIRECT_URL}?code=c&state=other", _AUTHORIZATION)
    with pytest.raises(Exception, match="does not match this sign-in's redirect URI"):
        response_from_redirect_url("http://127.0.0.1:1234/callback?code=c&state=state-1", _AUTHORIZATION)
    with pytest.raises(Exception, match="access_denied"):
        response_from_redirect_url(f"{_REDIRECT_URL}?error=access_denied", _AUTHORIZATION)
    with pytest.raises(Exception, match="Expected the full redirect URL"):
        response_from_redirect_url("code=c", _AUTHORIZATION)
