"""Mirror of pi mcp test/sign-in.test.ts.

The authorization server runs on `loopback_servers`, entered in each test
body. pi's cancellation cases abort after `setTimeout(..., 50)` against a
request or lock wait that never ends; here the test waits (bounded) until
the request is parked on the stalled path, or the sign-in waits for the
lock, then cancels.
"""

import base64
import hashlib
import json
from dataclasses import replace

import pytest
import tonio.colored as tonio
from mcp_authorization_server import (
    WAIT_S,
    AuthorizationServer,
    RecordingStore,
    authorize_in_browser,
    memory_store,
    start_authorization_server,
)
from mcp_helpers import loopback_servers

from pidrei_mcp.oauth import (
    McpAuthorizationResponse,
    McpClientMetadataDocuments,
    McpOAuthSettings,
    McpRefreshLockBusyError,
    McpSignInCancelledError,
    McpSignInOptions,
    McpSignInRedirect,
    OAuthCallback,
    complete_mcp_sign_in,
    start_mcp_sign_in,
)
from pidrei_mcp.url import parse_url
from pidrei_utils.cancel import CancelToken


# Callback routes of the application on two origins, like p2 through a relay and on its public URL.
ROUTE = "https://app.example/mcp/callback"
OTHER_ROUTE = "https://relay.example/mcp/callback"
# A loopback redirect URI; tests read the redirect from the authorization server instead of following it.
LOOPBACK = "http://127.0.0.1:1/callback"


def _lists_loopback_callback(redirect_url: str) -> bool:
    url = parse_url(redirect_url)
    return url.hostname == "127.0.0.1" and url.pathname == "/callback"


# Documents that list `/callback` on 127.0.0.1, like pi's on pi.dev.
DOCUMENTS = McpClientMetadataDocuments(
    base_url="https://app.example/oauth", lists_redirect_url=_lists_loopback_callback
)


def _options(server: AuthorizationServer, settings: McpOAuthSettings | None = None, **changes) -> McpSignInOptions:
    options = McpSignInOptions(
        server_url=server.server_url,
        store=memory_store(),
        settings=settings if settings is not None else McpOAuthSettings(),
        redirect_url=ROUTE,
        client_name="app",
    )
    return replace(options, **changes)


async def _start_redirect(options: McpSignInOptions) -> McpSignInRedirect:
    started = await start_mcp_sign_in(options)
    if started["kind"] != "redirect":
        raise AssertionError("Expected the browser step")
    return started


def _route_response(location: str) -> McpAuthorizationResponse:
    """The response a callback route reads from the query of the redirect."""
    query = parse_url(location)
    return OAuthCallback(
        code=query.search_param("code") or "",
        state=query.search_param("state") or "",
        iss=query.search_param("iss") or None,
    )


async def _sign_in(options: McpSignInOptions) -> McpSignInRedirect:
    """Start, follow the browser step, and complete."""
    authorization = await _start_redirect(options)
    response = _route_response(await authorize_in_browser(authorization["authorizationUrl"]))
    await complete_mcp_sign_in(options, authorization=authorization, response=response)
    return authorization


def _search_param(url: str, name: str) -> str | None:
    return parse_url(url).search_param(name)


def _callback_id(server_url: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(server_url.encode()).digest()[:9]).decode().rstrip("=")


def _grant_types(server: AuthorizationServer) -> list[str | None]:
    return [request.get("grant_type") for request in server.token_requests]


def _assert_matches(actual, expected) -> None:
    """pi's `toMatchObject`: a subset check."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict), actual
        for key, value in expected.items():
            assert key in actual, (key, actual)
            _assert_matches(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), actual
        for item, value in zip(actual, expected, strict=True):
            _assert_matches(item, value)
    else:
        assert actual == expected


# start and complete


@pytest.mark.tonio
async def test_completes_in_a_second_call_with_the_same_store_and_a_persisted_redirect(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server, McpOAuthSettings(scope="issues:read"))

        started = await _start_redirect(options)
        # Plain data: the application may store it and read it back in another request.
        authorization = json.loads(json.dumps(started))
        assert authorization == started
        _assert_matches(authorization, {"redirectUrl": ROUTE, "scope": "issues:read"})
        assert _search_param(authorization["authorizationUrl"], "state") == authorization["state"]
        _assert_matches(
            server.registrations, [{"client_name": "app", "redirect_uris": [ROUTE], "application_type": "web"}]
        )

        response = _route_response(await authorize_in_browser(authorization["authorizationUrl"]))
        await complete_mcp_sign_in(options, authorization=authorization, response=response)

        _assert_matches((await options.store.load())["tokens"], {"access_token": "access-1", "scope": "issues:read"})
        assert server.token_requests[0]["redirect_uri"] == ROUTE
        assert server.token_requests[0]["client_id"] == "client-1"


@pytest.mark.tonio
async def test_refreshes_the_stored_tokens_instead_of_redirecting(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        await _sign_in(options)

        assert await start_mcp_sign_in(options) == {"kind": "authorized"}
        assert len(server.authorizations) == 1
        assert server.token_requests[1]["grant_type"] == "refresh_token"
        assert (await options.store.load())["tokens"]["access_token"] == "access-2"


@pytest.mark.tonio
async def test_refreshes_under_the_stores_refresh_lock(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        recording = RecordingStore()
        options = _options(server, store=recording)
        await _sign_in(options)

        assert await start_mcp_sign_in(options) == {"kind": "authorized"}
        assert [save for save in recording.saves if save["access_token"] == "access-2"] == [
            {"locked": True, "access_token": "access-2"}
        ]


@pytest.mark.tonio
async def test_goes_to_the_browser_step_when_another_process_holds_the_refresh_lock_for_too_long(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server, store=RecordingStore(lock_error=McpRefreshLockBusyError()))
        await _sign_in(options)

        assert (await start_mcp_sign_in(options))["kind"] == "redirect"
        assert _grant_types(server) == ["authorization_code"]


@pytest.mark.tonio
async def test_fails_when_the_refresh_lock_fails_for_another_reason(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        lock_error = PermissionError("EACCES: permission denied")
        options = _options(server, store=RecordingStore(lock_error=lock_error))

        with pytest.raises(PermissionError) as caught:
            await start_mcp_sign_in(options)
        assert caught.value is lock_error
        assert server.registrations == []
        assert server.authorizations == []


@pytest.mark.tonio
async def test_refuses_to_complete_with_another_redirect_uri_than_the_authorization_request(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True)
        options = _options(server, redirect_url=LOOPBACK, client_metadata_documents=DOCUMENTS)
        authorization = await _start_redirect(options)
        response = _route_response(await authorize_in_browser(authorization["authorizationUrl"]))
        # Another loopback port than the one passed to start: the server-specific variant differs too.
        wrong = replace(options, redirect_url="http://127.0.0.1:2/callback")
        with pytest.raises(Exception) as caught:
            await complete_mcp_sign_in(wrong, authorization=authorization, response=response)
        assert f"but the authorization request used {authorization['redirectUrl']}" in str(caught.value)
        assert server.token_requests == []


@pytest.mark.tonio
async def test_gives_every_sign_in_a_fresh_state_and_refuses_to_complete_a_superseded_one(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        first = await _start_redirect(options)
        first_response = _route_response(await authorize_in_browser(first["authorizationUrl"]))
        second = await _start_redirect(options)
        assert second["state"] != first["state"]

        with pytest.raises(Exception, match="Another sign-in to this MCP server started since"):
            await complete_mcp_sign_in(options, authorization=first, response=first_response)
        assert server.token_requests == []


@pytest.mark.tonio
async def test_refuses_a_response_with_another_state(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        authorization = await _start_redirect(options)
        response = _route_response(await authorize_in_browser(authorization["authorizationUrl"]))
        with pytest.raises(Exception, match="belongs to a different sign-in"):
            await complete_mcp_sign_in(options, authorization=authorization, response=replace(response, state="other"))
        assert server.token_requests == []


# stored client


@pytest.mark.tonio
async def test_refreshes_with_the_stored_client_from_another_redirect_uri(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        await _sign_in(options)

        assert await start_mcp_sign_in(replace(options, redirect_url=OTHER_ROUTE)) == {"kind": "authorized"}
        assert len(server.registrations) == 1
        assert server.token_requests[1]["client_id"] == "client-1"
        assert (await options.store.load())["clientInformation"]["client_id"] == "client-1"


@pytest.mark.tonio
async def test_keeps_the_client_for_a_browser_step_on_its_redirect_uri(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        await _sign_in(options)
        server.revoke_refresh_tokens()

        authorization = await _start_redirect(options)
        assert _search_param(authorization["authorizationUrl"], "client_id") == "client-1"
        assert len(server.registrations) == 1


@pytest.mark.tonio
async def test_drops_the_client_and_its_tokens_for_a_browser_step_on_another_redirect_uri(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        await _sign_in(options)
        server.revoke_refresh_tokens()

        authorization = await _start_redirect(replace(options, redirect_url=OTHER_ROUTE))
        assert _search_param(authorization["authorizationUrl"], "client_id") == "client-2"
        _assert_matches(server.registrations[1], {"redirect_uris": [OTHER_ROUTE]})
        state = await options.store.load()
        assert state["clientInformation"]["client_id"] == "client-2"
        assert "tokens" not in state


@pytest.mark.tonio
async def test_does_not_refresh_with_a_registered_client_that_cimd_replaces(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True, iss_parameter=True)
        options = _options(server, redirect_url=LOOPBACK, client_metadata_documents=DOCUMENTS)
        await _sign_in(replace(options, settings=McpOAuthSettings(client_registration="dcr")))
        assert (await options.store.load())["clientInformation"]["client_id"] == "client-1"

        authorization = await _start_redirect(replace(options, settings=McpOAuthSettings(client_registration="cimd")))
        assert _search_param(authorization["authorizationUrl"], "client_id") == "https://app.example/oauth/client.json"
        assert _grant_types(server) == ["authorization_code"]
        state = await options.store.load()
        assert "clientInformation" not in state
        assert "tokens" not in state


@pytest.mark.tonio
async def test_asks_for_the_granted_scope_plus_the_challenged_one_on_a_step_up_without_refreshing(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server, challenge={"scope": "issues:read"})
        await _sign_in(options)
        assert (await options.store.load())["tokens"]["scope"] == "issues:read"

        # The challenge lists only the missing scope; requesting just that would lose issues:read.
        step_up = replace(options, challenge={"error": "insufficient_scope", "scope": "issues:write"})
        authorization = await _sign_in(step_up)
        assert authorization["scope"] == "issues:read issues:write"
        assert _grant_types(server) == ["authorization_code", "authorization_code"]
        assert (await options.store.load())["tokens"]["scope"] == "issues:read issues:write"


# client metadata documents


def _loopback(server: AuthorizationServer, settings: McpOAuthSettings | None = None) -> McpSignInOptions:
    return _options(server, settings, redirect_url=LOOPBACK, client_metadata_documents=DOCUMENTS)


@pytest.mark.tonio
async def test_uses_a_document_when_the_server_supports_them_and_the_documents_list_the_redirect_uri(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True, iss_parameter=True)
        options = _loopback(server)
        authorization = await _sign_in(options)
        assert _search_param(authorization["authorizationUrl"], "client_id") == "https://app.example/oauth/client.json"
        assert authorization["redirectUrl"] == LOOPBACK
        assert server.registrations == []
        assert server.token_requests[0]["client_id"] == "https://app.example/oauth/client.json"
        assert "clientInformation" not in await options.store.load()


@pytest.mark.tonio
async def test_uses_a_document_and_redirect_uri_specific_to_the_mcp_server_without_the_iss_parameter(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True)
        authorization = await _sign_in(_loopback(server))
        callback_id = _callback_id(server.server_url)
        assert (
            _search_param(authorization["authorizationUrl"], "client_id")
            == f"https://app.example/oauth/{callback_id}/client.json"
        )
        assert authorization["redirectUrl"] == f"{LOOPBACK}/{callback_id}"
        assert server.token_requests[0]["redirect_uri"] == f"{LOOPBACK}/{callback_id}"


@pytest.mark.tonio
async def test_appends_the_server_specific_path_without_doubling_a_trailing_slash(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True)
        authorization = await _start_redirect(
            replace(
                _loopback(server),
                redirect_url=f"{LOOPBACK}/",
                client_metadata_documents=replace(DOCUMENTS, lists_redirect_url=lambda _url: True),
            )
        )
        assert authorization["redirectUrl"] == f"{LOOPBACK}/{_callback_id(server.server_url)}"


@pytest.mark.tonio
async def test_registers_dynamically_when_the_server_does_not_support_documents(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, iss_parameter=True)
        await _sign_in(_loopback(server))
        assert len(server.registrations) == 1
        assert server.authorizations[0]["client_id"] == "client-1"


@pytest.mark.tonio
async def test_registers_dynamically_when_the_documents_do_not_list_the_redirect_uri(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True, iss_parameter=True)
        await _sign_in(replace(_loopback(server), redirect_url=ROUTE))
        assert len(server.registrations) == 1


@pytest.mark.tonio
async def test_registers_dynamically_without_documents_or_with_a_client_name(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True, iss_parameter=True)
        await _sign_in(replace(_loopback(server), client_metadata_documents=None))
        await _sign_in(_loopback(server, McpOAuthSettings(client_name="Other")))
        _assert_matches(server.registrations, [{"client_name": "app"}, {"client_name": "Other"}])


# Documents that list only the application's https callback route.
ROUTE_DOCUMENTS = McpClientMetadataDocuments(
    base_url="https://app.example/oauth", lists_redirect_url=lambda redirect_url: redirect_url == ROUTE
)


@pytest.mark.tonio
@pytest.mark.parametrize("client_registration", ["auto", "cimd"])
async def test_refreshes_the_tokens_of_a_document_that_lists_only_an_https_route(monkeypatch, client_registration):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True, iss_parameter=True)
        options = _options(
            server,
            McpOAuthSettings(client_registration=client_registration),
            client_metadata_documents=ROUTE_DOCUMENTS,
        )
        authorization = await _sign_in(options)
        assert _search_param(authorization["authorizationUrl"], "client_id") == "https://app.example/oauth/client.json"
        assert "clientInformation" not in await options.store.load()

        assert await start_mcp_sign_in(options) == {"kind": "authorized"}
        assert len(server.authorizations) == 1
        assert server.token_requests[1]["grant_type"] == "refresh_token"
        assert server.token_requests[1]["client_id"] == "https://app.example/oauth/client.json"
        assert (await options.store.load())["tokens"]["access_token"] == "access-2"


@pytest.mark.tonio
async def test_fails_with_cimd_when_the_server_does_not_support_documents(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, iss_parameter=True)
        with pytest.raises(Exception, match="does not support Client ID Metadata Documents"):
            await start_mcp_sign_in(_loopback(server, McpOAuthSettings(client_registration="cimd")))
        assert server.registrations == []


@pytest.mark.tonio
async def test_fails_with_cimd_when_the_documents_do_not_list_the_redirect_uri(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers, cimd=True, iss_parameter=True)
        options = replace(_loopback(server, McpOAuthSettings(client_registration="cimd")), redirect_url=ROUTE)
        with pytest.raises(Exception, match="needs Client ID Metadata Documents that list the redirect URI"):
            await start_mcp_sign_in(options)
        assert server.registrations == []


# cancellation


@pytest.mark.tonio
async def test_cancels_a_start_whose_request_does_not_answer(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        server.stall.add("/register")
        cancel = CancelToken()
        started = tonio.spawn(start_mcp_sign_in(_options(server, cancel=cancel)))
        await server.stalled_arrived.wait(WAIT_S)
        assert server.stalled_arrived.is_set()
        cancel.cancel()
        with pytest.raises(BaseExceptionGroup) as caught:
            await started
        assert caught.group_contains(McpSignInCancelledError)


@pytest.mark.tonio
async def test_cancels_a_completion_whose_token_request_does_not_answer(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        options = _options(server)
        authorization = await _start_redirect(options)
        response = _route_response(await authorize_in_browser(authorization["authorizationUrl"]))
        server.stall.add("/token")
        cancel = CancelToken()
        completed = tonio.spawn(
            complete_mcp_sign_in(replace(options, cancel=cancel), authorization=authorization, response=response)
        )
        await server.stalled_arrived.wait(WAIT_S)
        assert server.stalled_arrived.is_set()
        cancel.cancel()
        with pytest.raises(BaseExceptionGroup) as caught:
            await completed
        assert caught.group_contains(McpSignInCancelledError)


class _LockWaitStore(RecordingStore):
    """A `RecordingStore` that reports when a run starts waiting for the lock."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting = tonio.Event()

    def with_refresh_lock(self, fn):
        run = super().with_refresh_lock(fn)
        self.waiting.set()
        return run


@pytest.mark.tonio
async def test_cancels_at_once_while_another_process_holds_the_refresh_lock_and_refreshes_nothing(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        recording = _LockWaitStore()
        options = _options(server, store=recording)
        await _sign_in(options)

        release = recording.hold()
        cancel = CancelToken()
        started = tonio.spawn(start_mcp_sign_in(replace(options, cancel=cancel)))
        await recording.waiting.wait(WAIT_S)
        assert recording.waiting.is_set()
        cancel.cancel()
        with pytest.raises(BaseExceptionGroup) as caught:
            await started
        assert caught.group_contains(McpSignInCancelledError)

        # The lock arrives after the cancellation: nothing runs under it, and it is released.
        release()
        for run in recording.lock_runs:
            await run.wait(WAIT_S)
            assert run.is_set()
        assert not recording.is_locked()
        assert _grant_types(server) == ["authorization_code"]
        assert len(server.authorizations) == 1


@pytest.mark.tonio
async def test_fails_at_once_with_a_cancelled_token(monkeypatch):
    async with loopback_servers(monkeypatch) as servers:
        server = await start_authorization_server(servers)
        cancel = CancelToken()
        cancel.cancel()
        with pytest.raises(McpSignInCancelledError):
            await start_mcp_sign_in(_options(server, cancel=cancel))
        assert server.registrations == []
