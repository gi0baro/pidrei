"""Mirror of pi mcp test/oauth.test.ts."""

import base64
import hashlib
import json
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pytest
import tonio.colored as tonio
from mcp_helpers import loopback_servers, read_body, read_json

from pidrei_http import http
from pidrei_mcp import McpClient, StreamableHttpTransport, default_fetch
from pidrei_mcp.auth_provider import UnauthorizedContext
from pidrei_mcp.oauth import (
    McpOAuthAuthorizationRequiredError,
    McpOAuthProvider,
    MemoryOAuthStateStore,
    OAuthCallbackPage,
    OAuthCallbackServer,
    OAuthClientProvider,
    OAuthFlowOptions,
    OAuthInsecureEndpointError,
    OAuthIssuerMismatchError,
    adapt_oauth_provider,
    authorize_mcp,
    discover_authorization_server_metadata,
    register_client,
)
from pidrei_mcp.url import parse_url
from pidrei_utils import clock


class TestOAuthProvider(OAuthClientProvider):
    __test__ = False

    def __init__(self, redirect_url: str) -> None:
        self.redirect_url = redirect_url
        self.client_metadata = {
            "redirect_uris": [redirect_url],
            "client_name": "pi-mcp-test",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        self.client: dict[str, Any] | None = None
        self.token_set: dict[str, Any] | None = None
        self.verifier: str | None = None
        self.discovery: dict[str, Any] | None = None
        self.authorization_url: str | None = None

    async def state(self) -> str:  # type: ignore[override]
        return "expected-state"

    async def client_information(self):
        return self.client

    async def save_client_information(self, information) -> None:  # type: ignore[override]
        self.client = information

    async def tokens(self):
        return self.token_set

    async def save_tokens(self, tokens) -> None:
        self.token_set = tokens

    async def redirect_to_authorization(self, url: str) -> None:
        self.authorization_url = url

    async def save_code_verifier(self, verifier: str) -> None:
        self.verifier = verifier

    async def code_verifier(self) -> str:
        if not self.verifier:
            raise RuntimeError("Missing code verifier")
        return self.verifier

    async def invalidate_credentials(self, kind) -> None:  # type: ignore[override]
        if kind in ("all", "client"):
            self.client = None
        if kind in ("all", "tokens"):
            self.token_set = None
        if kind in ("all", "verifier"):
            self.verifier = None
        if kind in ("all", "discovery"):
            self.discovery = None

    async def save_discovery_state(self, state) -> None:  # type: ignore[override]
        self.discovery = state

    async def discovery_state(self):  # type: ignore[override]
        return self.discovery


class _Response:
    """A response handed to `on_unauthorized`, as pi builds one with `new Response(...)`."""

    def __init__(self, status: int, www_authenticate: str) -> None:
        self.status = status
        self._headers = {"www-authenticate": www_authenticate}

    @property
    def headers(self):
        return self._headers


def _path(request: Any) -> str:
    return urlsplit(request.target).path


def _query(request: Any) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(request.target).query, keep_blank_values=True))


async def _respond_json(request: Any, value: Any, status: int = 200) -> None:
    await request.respond(status, headers={"content-type": "application/json"}, body=json.dumps(value).encode())


def _search_param(url: str | None, name: str) -> str | None:
    assert url is not None
    return parse_url(url).search_param(name)


@pytest.mark.tonio
async def test_discovers_registers_authorizes_with_pkce_and_refreshes_on_401(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        expected_challenge: str | None = None
        refreshes = 0

        async def handler(request, origin):
            nonlocal expected_challenge, refreshes
            path = _path(request)
            if path == "/.well-known/oauth-protected-resource/mcp":
                await _respond_json(
                    request,
                    {"resource": f"{origin}/mcp", "authorization_servers": [origin], "scopes_supported": ["org:read"]},
                )
                return
            if path == "/.well-known/oauth-authorization-server":
                await _respond_json(
                    request,
                    {
                        "issuer": origin,
                        "authorization_endpoint": f"{origin}/authorize",
                        "token_endpoint": f"{origin}/token",
                        "registration_endpoint": f"{origin}/register",
                        "response_types_supported": ["code"],
                        "grant_types_supported": ["authorization_code", "refresh_token"],
                        "token_endpoint_auth_methods_supported": ["none"],
                        "code_challenge_methods_supported": ["S256"],
                    },
                )
                return
            if path == "/register":
                metadata = await read_json(request)
                # Empty and null optional fields count as absent (#10266).
                await _respond_json(request, {**metadata, "client_id": "test-client", "client_secret": ""}, 201)
                return
            if path == "/authorize":
                query = _query(request)
                expected_challenge = query.get("code_challenge")
                redirect = parse_url(query.get("redirect_uri", ""))
                redirect = redirect.with_search_param("code", "test-code").with_search_param(
                    "state", query.get("state", "")
                )
                await request.respond(302, headers={"location": redirect.href})
                return
            if path == "/token":
                params = dict(parse_qsl(await read_body(request), keep_blank_values=True))
                if params.get("grant_type") == "refresh_token":
                    refreshes += 1
                    await _respond_json(
                        request,
                        {
                            "access_token": "refreshed-token",
                            "token_type": "Bearer",
                            "refresh_token": "",
                            "expires_in": None,
                        },
                    )
                    return
                digest = hashlib.sha256(params.get("code_verifier", "").encode()).digest()
                challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")
                if params.get("code") != "test-code" or challenge != expected_challenge:
                    await _respond_json(request, {"error": "invalid_grant"}, 400)
                    return
                await _respond_json(
                    request,
                    {
                        "access_token": "first-token",
                        "refresh_token": "refresh-token",
                        "token_type": "Bearer",
                        "scope": "",
                    },
                )
                return
            if path != "/mcp":
                await request.respond(404)
                return
            if request.method == "GET":
                await request.respond(405)
                return
            if request.method == "DELETE":
                await request.respond(200)
                return
            token = request.headers.get("authorization")
            token = None if token is None else bytes(token).decode()
            if token not in ("Bearer first-token", "Bearer refreshed-token"):
                await read_body(request)
                await request.respond(
                    401,
                    headers={
                        # An empty scope falls through to the resource metadata's scopes_supported.
                        "www-authenticate": (
                            f'Bearer resource_metadata="{origin}/.well-known/oauth-protected-resource/mcp", scope=""'
                        )
                    },
                    body=b"Unauthorized",
                )
                return
            message = await read_json(request)
            if "id" not in message:
                await request.respond(202)
                return
            if message["method"] == "initialize":
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "oauth-test", "version": "1.0.0"},
                }
            else:
                result = {"tools": [{"name": "issues", "inputSchema": {"type": "object"}}]}
            await _respond_json(request, {"jsonrpc": "2.0", "id": message["id"], "result": result})

        origin = await http_servers.listen(handler)
        callback = await OAuthCallbackServer()
        try:
            provider = TestOAuthProvider(callback.redirect_url)

            def transport():
                return StreamableHttpTransport(
                    f"{origin}/mcp",
                    headers={"Authorization": "Bearer caller-supplied-stale-token"},
                    auth_provider=adapt_oauth_provider(provider),
                    open_get_stream=False,
                )

            first_client = McpClient(name="oauth-test", version="1.0.0")
            with pytest.raises(McpOAuthAuthorizationRequiredError):
                await first_client.connect(transport())
            assert _search_param(provider.authorization_url, "scope") == "org:read"
            assert _search_param(provider.authorization_url, "resource") == f"{origin}/mcp"

            callback_result = callback.wait_for_callback("expected-state")
            authorization_response = await http.shared_client().get(provider.authorization_url, follow_redirects=False)
            await authorization_response.read()
            callback_page = await http.shared_client().get(authorization_response.headers["location"])
            await callback_page.read()
            code = (await callback_result).code
            assert await authorize_mcp(
                provider, OAuthFlowOptions(server_url=f"{origin}/mcp", authorization_code=code)
            ) == ("AUTHORIZED")

            client = McpClient(name="oauth-test", version="1.0.0")
            await client.connect(transport())
            assert await client.list_tools() == [{"name": "issues", "inputSchema": {"type": "object"}}]
            await client.close()

            provider.token_set = {**provider.token_set, "access_token": "stale-token"}
            refreshed_client = McpClient(name="oauth-test", version="1.0.0")
            await refreshed_client.connect(transport())
            # Neither token response names a scope, so the grant has the requested scope.
            assert provider.token_set == {
                "access_token": "refreshed-token",
                "refresh_token": "refresh-token",
                "token_type": "Bearer",
                "scope": "org:read",
            }
            assert refreshes == 1
            await refreshed_client.close()
        finally:
            callback.close()


@pytest.mark.tonio
async def test_shares_one_refresh_between_concurrent_401s_when_refresh_tokens_rotate(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        grants: list[str] = []

        async def handler(request, origin):
            path = _path(request)
            if path == "/.well-known/oauth-protected-resource/mcp":
                # Invalid resource metadata falls back to the server origin instead of failing discovery.
                await _respond_json(request, {"resource": f"{origin}/mcp", "authorization_servers": ["not a url"]})
                return
            if path == "/.well-known/oauth-authorization-server":
                await _respond_json(
                    request,
                    {
                        # Issuer without the trailing slash that URL parsing adds to the fallback server URL.
                        "issuer": origin,
                        "authorization_endpoint": f"{origin}/authorize",
                        "token_endpoint": f"{origin}/token",
                        "response_types_supported": ["code"],
                    },
                )
                return
            if path == "/token":
                params = dict(parse_qsl(await read_body(request), keep_blank_values=True))
                refresh_token = params.get("refresh_token", "")
                grants.append(refresh_token)
                if refresh_token != "r1":
                    await _respond_json(request, {"error": "invalid_grant"}, 400)
                    return
                await _respond_json(
                    request, {"access_token": "a2", "refresh_token": "r2", "token_type": "Bearer", "expires_in": 3600}
                )
                return
            await request.respond(404)

        origin = await http_servers.listen(handler)
        store = MemoryOAuthStateStore()

        async def on_redirect(_url: str) -> None:
            pass

        provider = McpOAuthProvider(
            server_url=f"{origin}/mcp",
            redirect_url="http://127.0.0.1/callback",
            client_metadata={"client_name": "test"},
            client_id="client",
            store=store,
            on_redirect=on_redirect,
        )
        await provider.save_tokens({"access_token": "a1", "refresh_token": "r1", "token_type": "Bearer"})
        auth = adapt_oauth_provider(provider)

        def unauthorized() -> UnauthorizedContext:
            return UnauthorizedContext(
                response=_Response(401, "Bearer"), server_url=f"{origin}/mcp", fetch=default_fetch, token="a1"
            )

        # Whichever way the two interleave here (joining the refresh in flight,
        # or arriving after it replaced the token), only one grant is spent.
        await tonio.spawn(auth.on_unauthorized(unauthorized()), auth.on_unauthorized(unauthorized()))
        # A late 401 for a request that still carried the old token must not refresh again.
        await auth.on_unauthorized(unauthorized())
        assert grants == ["r1"]
        assert await auth.token() == "a2"
        state = await store.load()
        assert state["tokens"]["refresh_token"] == "r2"
        assert state["tokensExpireAt"] > clock.now_ms() + 3_500_000


@pytest.mark.tonio
async def test_asks_for_authorization_instead_of_refreshing_when_the_server_needs_more_scope(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        provider = TestOAuthProvider("http://127.0.0.1/callback")
        provider.client = {"client_id": "client"}
        provider.token_set = {
            "access_token": "a1",
            "refresh_token": "r1",
            "token_type": "Bearer",
            "scope": "repo read:org",
        }

        async def handler(request, origin):
            path = _path(request)
            if path == "/.well-known/oauth-authorization-server":
                await _respond_json(
                    request,
                    {
                        "issuer": origin,
                        "authorization_endpoint": f"{origin}/authorize",
                        "token_endpoint": f"{origin}/token",
                        "response_types_supported": ["code"],
                    },
                )
                return
            await request.respond(500 if path == "/token" else 404)

        origin = await http_servers.listen(handler)
        auth = adapt_oauth_provider(provider)
        with pytest.raises(McpOAuthAuthorizationRequiredError):
            await auth.on_unauthorized(
                UnauthorizedContext(
                    response=_Response(403, 'Bearer error="insufficient_scope", scope="repo admin"'),
                    server_url=f"{origin}/mcp",
                    fetch=default_fetch,
                    token="a1",
                )
            )
        # The challenge may list only the missing scopes; the new grant keeps the old ones too.
        assert _search_param(provider.authorization_url, "scope") == "repo read:org admin"
        # The working grant is kept until the user authorizes the new scope.
        assert provider.token_set["access_token"] == "a1"


@pytest.mark.tonio
async def test_binds_persisted_credentials_to_the_exact_mcp_server_url():
    store = MemoryOAuthStateStore()

    async def on_redirect(_url: str) -> None:
        pass

    first = McpOAuthProvider(
        server_url="https://one.example/mcp",
        redirect_url="http://127.0.0.1/callback",
        client_metadata={"client_name": "test"},
        store=store,
        on_redirect=on_redirect,
    )
    await first.save_tokens({"access_token": "secret", "token_type": "Bearer"})
    assert (await first.tokens())["access_token"] == "secret"

    second = McpOAuthProvider(
        server_url="https://two.example/mcp",
        redirect_url="http://127.0.0.1/callback",
        client_metadata={"client_name": "test"},
        store=store,
        on_redirect=on_redirect,
    )
    assert await second.tokens() is None


# #10493
@pytest.mark.tonio
async def test_registers_with_an_application_type_derived_from_the_redirect_uris_unless_one_is_set(monkeypatch):
    bodies: list[dict[str, Any]] = []
    async with loopback_servers(monkeypatch) as http_servers:

        async def handler(request, _origin):
            metadata = await read_json(request)
            bodies.append(metadata)
            await _respond_json(request, {**metadata, "client_id": "client"}, 201)

        origin = await http_servers.listen(handler)

        async def register(redirect_uris: list[str], application_type: str | None = None) -> None:
            await register_client(
                origin,
                client_metadata={
                    "redirect_uris": redirect_uris,
                    **({"application_type": application_type} if application_type else {}),
                },
            )

        await register(["http://127.0.0.1:1234/callback"])
        await register(["http://[::1]/callback"])
        await register(["com.example.app:/callback"])
        await register(["https://app.example/callback"])
        await register(["http://localhost/callback"], "web")
    assert [body["application_type"] for body in bodies] == ["native", "native", "native", "web", "web"]


@pytest.mark.tonio
async def test_rejects_authorization_metadata_whose_issuer_does_not_match_discovery(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:

        async def handler(request, origin):
            if _path(request) == "/.well-known/oauth-authorization-server":
                await _respond_json(
                    request,
                    {
                        "issuer": "https://attacker.example",
                        "authorization_endpoint": f"{origin}/authorize",
                        "token_endpoint": f"{origin}/token",
                        "response_types_supported": ["code"],
                    },
                )
                return
            await request.respond(404)

        origin = await http_servers.listen(handler)
        with pytest.raises(OAuthIssuerMismatchError):
            await discover_authorization_server_metadata(origin)


@pytest.mark.tonio
async def test_uses_a_configured_authorization_server_metadata_document_as_is(monkeypatch):
    """#10172"""
    async with loopback_servers(monkeypatch) as http_servers:

        async def handler(request, origin):
            path = _path(request)
            if path == "/.well-known/oauth-protected-resource/mcp":
                # Names the MCP server itself, which serves no authorization server metadata.
                await _respond_json(request, {"resource": f"{origin}/mcp", "authorization_servers": [origin]})
            elif path == "/idp/metadata.json":
                await _respond_json(
                    request,
                    {
                        # Not derivable from the document URL; a configured document is not checked.
                        "issuer": "https://idp.example",
                        "authorization_endpoint": f"{origin}/idp/authorize",
                        "token_endpoint": f"{origin}/idp/token",
                        "response_types_supported": ["code"],
                    },
                )
            else:
                await request.respond(404, headers={"content-type": "application/json"})

        origin = await http_servers.listen(handler)
        provider = TestOAuthProvider("http://127.0.0.1/callback")
        provider.client = {"client_id": "client"}
        options = OAuthFlowOptions(
            server_url=f"{origin}/mcp", authorization_server_metadata_url=f"{origin}/idp/metadata.json"
        )
        assert await authorize_mcp(provider, options) == "REDIRECT"
        authorization_url = parse_url(provider.authorization_url)
        assert f"{authorization_url.origin}{authorization_url.pathname}" == f"{origin}/idp/authorize"
        assert authorization_url.search_param("resource") == f"{origin}/mcp"

        insecure = OAuthFlowOptions(
            server_url=f"{origin}/mcp", authorization_server_metadata_url="http://idp.example/metadata.json"
        )
        with pytest.raises(OAuthInsecureEndpointError):
            await authorize_mcp(provider, insecure)


@pytest.mark.tonio
async def test_exchanges_a_code_only_when_its_iss_parameter_names_the_authorization_server(monkeypatch):
    async with loopback_servers(monkeypatch) as http_servers:
        codes: list[str] = []

        async def handler(request, _origin):
            codes.append(dict(parse_qsl(await read_body(request), keep_blank_values=True)).get("code", ""))
            await _respond_json(request, {"access_token": "token", "token_type": "Bearer"})

        origin = await http_servers.listen(handler)

        def exchange(code: str, iss: str | None, iss_parameter_supported: bool):
            provider = TestOAuthProvider("http://127.0.0.1/callback")
            provider.client = {"client_id": "client"}
            provider.verifier = "verifier"
            provider.discovery = {
                "authorizationServerUrl": origin,
                "authorizationServerMetadata": {
                    "issuer": origin,
                    "authorization_endpoint": f"{origin}/authorize",
                    "token_endpoint": f"{origin}/token",
                    "response_types_supported": ["code"],
                    "authorization_response_iss_parameter_supported": iss_parameter_supported,
                },
            }
            return authorize_mcp(
                provider, OAuthFlowOptions(server_url=f"{origin}/mcp", authorization_code=code, iss=iss)
            )

        with pytest.raises(OAuthIssuerMismatchError):
            await exchange("other", "https://attacker.example", False)
        with pytest.raises(OAuthIssuerMismatchError):
            await exchange("missing", None, True)
        assert await exchange("matching", origin, True) == "AUTHORIZED"
        # Servers that do not promise the parameter may omit it.
        assert await exchange("omitted", None, False) == "AUTHORIZED"
        assert codes == ["matching", "omitted"]


@pytest.mark.tonio
async def test_callback_server_renders_plain_text_by_default(monkeypatch):
    async with loopback_servers(monkeypatch):
        callback = await OAuthCallbackServer()
        try:
            pending = callback.wait_for_callback("s1")
            response = await http.shared_client().get(f"{callback.redirect_url}?code=abc&state=s1")
            assert response.headers["content-type"] == "text/plain; charset=utf-8"
            assert (await response.read()).decode() == "Authorization complete. You may close this window."
            assert (await pending).code == "abc"
        finally:
            callback.close()


# #10302
@pytest.mark.tonio
async def test_callback_server_rejects_a_response_on_another_path_than_the_expected_one(monkeypatch):
    async with loopback_servers(monkeypatch):
        callback = await OAuthCallbackServer(extra_paths=["/callback/server-id"])
        try:
            origin = parse_url(callback.redirect_url).origin
            mixed_up = callback.wait_for_callback("s1", "/callback/server-id")
            wrong = await http.shared_client().get(f"{origin}/callback?code=abc&state=s1")
            await wrong.read()
            assert wrong.status_code == 400
            with pytest.raises(RuntimeError, match="arrived on another redirect URI"):
                await mixed_up

            pending = callback.wait_for_callback("s2", "/callback/server-id")
            right = await http.shared_client().get(f"{origin}/callback/server-id?code=abc&state=s2")
            await right.read()
            assert right.status_code == 200
            assert (await pending).code == "abc"
        finally:
            callback.close()


@pytest.mark.tonio
async def test_callback_server_renders_pages_through_render_page(monkeypatch):
    async with loopback_servers(monkeypatch):
        pages: list[OAuthCallbackPage] = []

        def render_page(page: OAuthCallbackPage) -> str:
            pages.append(page)
            return "<p>ok</p>" if page.ok else f"<p>{page.message}</p>"

        callback = await OAuthCallbackServer(render_page=render_page)
        try:
            denied = callback.wait_for_callback("s1")
            failure = await http.shared_client().get(
                f"{callback.redirect_url}?error=access_denied&error_description=Denied&state=s1"
            )
            await failure.read()
            assert failure.headers["content-type"] == "text/html; charset=utf-8"
            with pytest.raises(RuntimeError, match="Denied"):
                await denied
            assert pages[-1] == OAuthCallbackPage(
                ok=False, message="Authorization failed. You may close this window.", details="Denied"
            )

            pending = callback.wait_for_callback("s2")
            success = await http.shared_client().get(f"{callback.redirect_url}?code=abc&state=s2")
            assert (await success.read()).decode() == "<p>ok</p>"
            assert (await pending).code == "abc"
        finally:
            callback.close()
