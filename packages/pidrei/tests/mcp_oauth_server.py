"""Mirror of pi's suite/mcp-oauth-server.ts: an MCP server protected by
OAuth, with its own authorization server (discovery, DCR, PKCE, refresh).

pi's is a node `http` server; this one runs on the mcp package's loopback
test servers (`mcp_helpers.HttpServers`). `oauth_mcp_servers` (over
`mcp_helpers.loopback_servers`) gives the test a shared HTTP client without
keep-alive, so a connection ends with its exchange and the servers shut down
with nothing left open, as pi's helper force-closes them. `browse(url)` plays
the browser that follows the authorization redirect to the loopback callback
(pi's `void fetch(url)`); every browse is waited for before the servers shut
down. Entered inside the test body, never from a fixture (see
`mcp_helpers`): everything it opens is closed before the body returns.
"""

import base64
import contextlib
import hashlib
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import tonio.colored as tonio

from pidrei_mcp import LATEST_PROTOCOL_VERSION, default_fetch
from pidrei_mcp.url import parse_url


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mcp" / "tests"))
from mcp_helpers import HttpServers, header, loopback_servers, read_body


__all__ = ["OAuthMcpServer", "OAuthServers", "loopback_servers", "oauth_mcp_servers"]


async def _json(request: Any, status: int, body: Any, headers: dict[str, str] | None = None) -> None:
    await request.respond(
        status, headers={"content-type": "application/json", **(headers or {})}, body=json.dumps(body).encode()
    )


class OAuthMcpServer:
    """`iss` is sent as the `iss` parameter of authorization responses (RFC
    9207). `iss_parameter` advertises that parameter and sends the server's
    issuer. `cimd` advertises Client ID Metadata Documents. `redirect_path`
    replaces the path of the redirect URI, like a mixed-up authorization
    server. The MCP endpoint assigns a session, so closing a connection sends
    a DELETE. Paths added to `stall` accept requests and never answer them,
    like an unresponsive server."""

    def __init__(
        self, iss: str | None, *, iss_parameter: bool = False, cimd: bool = False, redirect_path: str | None = None
    ) -> None:
        self._iss = iss
        self._iss_parameter = iss_parameter
        self._cimd = cimd
        self._redirect_path = redirect_path
        self.log: list[str] = []
        # Client metadata of dynamic client registrations.
        self.registrations: list[dict[str, Any]] = []
        # Query parameters of authorization requests.
        self.authorizations: list[dict[str, str]] = []
        # Parameters of token requests.
        self.token_requests: list[dict[str, str]] = []
        # Access tokens of session DELETE requests.
        self.deletes: list[str | None] = []
        self.stall: set[str] = set()
        # Requests to stalled paths, each with an Event set once the client gave up (hung up).
        self.stalled: list[tuple[str, tonio.Event]] = []
        # Set when a request reached a stalled path.
        self.stalled_arrived = tonio.Event()
        self._valid_tokens: set[str] = set()
        self._refresh_tokens: set[str] = set()
        self._challenges: dict[str, str] = {}
        self._issued = 0
        self.origin = ""
        self.url = ""

    def expire_access_tokens(self) -> None:
        """Simulates access token expiry."""
        self._valid_tokens.clear()

    def _issue_tokens(self) -> dict[str, Any]:
        self._issued += 1
        tokens = {"access_token": f"access-{self._issued}", "refresh_token": f"refresh-{self._issued}"}
        self._valid_tokens.add(tokens["access_token"])
        self._refresh_tokens.add(tokens["refresh_token"])
        return {**tokens, "token_type": "Bearer", "expires_in": 3600}

    async def _handle_mcp(self, request: Any) -> None:
        authorization = header(request, "authorization")
        token = authorization.removeprefix("Bearer ") if authorization else None
        if request.method == "DELETE":
            self.deletes.append(token)
        if request.method != "POST":
            await request.respond(405 if request.method == "GET" else 200)
            return
        if not token or token not in self._valid_tokens:
            self.log.append(f"401 {token or 'none'}")
            await request.respond(
                401,
                headers={
                    "www-authenticate": f'Bearer resource_metadata="{self.origin}/.well-known/oauth-protected-resource/mcp"'
                },
            )
            return
        message = json.loads(await read_body(request))
        if message.get("id") is None:
            await request.respond(202)
            return
        method = message["method"]
        if method == "initialize":
            result: Any = {
                "protocolVersion": LATEST_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "issues", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": "whoami", "inputSchema": {"type": "object", "properties": {}}}]}
        elif method == "tools/call":
            self.log.append(f"call {token}")
            result = {"content": [{"type": "text", "text": f"token {token}"}]}
        else:
            result = {}
        await _json(
            request, 200, {"jsonrpc": "2.0", "id": message["id"], "result": result}, {"mcp-session-id": "session-1"}
        )

    async def handle(self, request: Any, origin: str) -> None:
        url = urlsplit(request.target)
        query = dict(parse_qsl(url.query, keep_blank_values=True))
        if url.path in self.stall:
            hung_up = tonio.Event()
            self.stalled.append((url.path, hung_up))
            # Read first: the client's hang-up shows only once nothing is left to read.
            await request.read()
            self.stalled_arrived.set()
            await request.peer_closed()
            hung_up.set()
            return
        match url.path:
            case "/mcp":
                await self._handle_mcp(request)
            case "/.well-known/oauth-protected-resource/mcp":
                await _json(request, 200, {"resource": f"{origin}/mcp", "authorization_servers": [origin]})
            case "/.well-known/oauth-authorization-server":
                await _json(
                    request,
                    200,
                    {
                        "issuer": origin,
                        "authorization_endpoint": f"{origin}/authorize",
                        "token_endpoint": f"{origin}/token",
                        "registration_endpoint": f"{origin}/register",
                        "response_types_supported": ["code"],
                        "code_challenge_methods_supported": ["S256"],
                        "token_endpoint_auth_methods_supported": ["none"],
                        **({"client_id_metadata_document_supported": True} if self._cimd else {}),
                        **({"authorization_response_iss_parameter_supported": True} if self._iss_parameter else {}),
                    },
                )
            case "/register":
                metadata = json.loads(await read_body(request))
                self.log.append("register")
                self.registrations.append(metadata)
                await _json(request, 201, {**metadata, "client_id": "client-1"})
            case "/authorize":
                self.authorizations.append(query)
                code = f"code-{len(self._challenges) + 1}"
                self._challenges[code] = query.get("code_challenge", "")
                redirect = parse_url(query.get("redirect_uri", ""))
                if self._redirect_path:
                    redirect = replace(redirect, pathname=self._redirect_path)
                redirect = redirect.with_search_param("code", code).with_search_param("state", query.get("state", ""))
                iss = self._iss if self._iss is not None else origin if self._iss_parameter else None
                if iss:
                    redirect = redirect.with_search_param("iss", iss)
                await request.respond(302, headers={"location": redirect.href})
            case "/token":
                params = dict(parse_qsl(await read_body(request), keep_blank_values=True))
                self.token_requests.append(params)
                if params.get("grant_type") == "authorization_code":
                    challenge = self._challenges.get(params.get("code", ""))
                    digest = hashlib.sha256(params.get("code_verifier", "").encode()).digest()
                    verifier = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                    if not challenge or challenge != verifier:
                        await _json(request, 400, {"error": "invalid_grant"})
                        return
                    del self._challenges[params["code"]]
                    self.log.append("token code")
                    await _json(request, 200, self._issue_tokens())
                    return
                refresh = params.get("refresh_token", "")
                if refresh not in self._refresh_tokens:
                    await _json(request, 400, {"error": "invalid_grant"})
                    return
                self._refresh_tokens.discard(refresh)
                self.log.append("token refresh")
                await _json(request, 200, self._issue_tokens())
            case _:
                await request.respond(404)


class OAuthServers:
    """`await servers.start(iss=...)` starts an `OAuthMcpServer`."""

    def __init__(self, servers: HttpServers) -> None:
        self._servers = servers
        self._browsing: list[Any] = []

    async def start(
        self,
        *,
        iss: str | None = None,
        iss_parameter: bool = False,
        cimd: bool = False,
        redirect_path: str | None = None,
    ) -> OAuthMcpServer:
        server = OAuthMcpServer(iss, iss_parameter=iss_parameter, cimd=cimd, redirect_path=redirect_path)

        async def handle(request: Any, origin: str) -> None:
            try:
                await server.handle(request, origin)
            except Exception as error:
                await request.respond(500, body=str(error).encode())

        server.origin = await self._servers.listen(handle)
        server.url = f"{server.origin}/mcp"
        return server

    def browse(self, url: str) -> None:
        """The browser opening `url` and following its redirects."""

        async def follow() -> None:
            try:
                response = await default_fetch(url)
                await response.read()
                await response.close()
            except Exception:
                pass

        self._browsing.append(tonio.spawn(follow()))

    async def finish_browsing(self) -> None:
        for browsing in self._browsing:
            await browsing


@contextlib.asynccontextmanager
async def oauth_mcp_servers(monkeypatch) -> AsyncIterator[OAuthServers]:
    async with loopback_servers(monkeypatch) as servers:
        oauth = OAuthServers(servers)
        try:
            yield oauth
        finally:
            await oauth.finish_browsing()
