"""Mirror of pi mcp test/authorization-server.ts: the authorization server of
an MCP server, and the stores the sign-in tests use.

pi's is a node `http` server closed from `afterEach`; this one runs on the
test's `mcp_helpers.loopback_servers`, which the test enters in its body and
which shuts every server down before the body returns. Paths in `stall`
accept a request and never answer it: the handler sets `stalled_arrived` and
waits for the client to hang up, so a test cancels once the request is
parked there (pi's `setTimeout(abort, 50)`).
"""

import base64
import hashlib
import json
import threading
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import tonio.colored as tonio
from mcp_helpers import HttpServers, read_body

from pidrei_http import http
from pidrei_mcp.oauth import McpOAuthServerStore, McpOAuthState, MemoryOAuthStateStore
from pidrei_mcp.url import parse_url


WAIT_S = 5.0


async def _json(request: Any, status: int, body: Any) -> None:
    await request.respond(status, headers={"content-type": "application/json"}, body=json.dumps(body).encode())


class AuthorizationServer:
    """Authorization server of an MCP server at `<origin>/mcp`: discovery,
    dynamic client registration, PKCE, and rotating refresh tokens. Token
    responses name no scope. `cimd` advertises Client ID Metadata Documents,
    `iss_parameter` the `iss` parameter of authorization responses."""

    def __init__(self, *, cimd: bool, iss_parameter: bool) -> None:
        self._cimd = cimd
        self._iss_parameter = iss_parameter
        self.server_url = ""
        # Client metadata of dynamic client registrations.
        self.registrations: list[dict[str, Any]] = []
        # Query parameters of authorization requests.
        self.authorizations: list[dict[str, str]] = []
        # Parameters of token requests.
        self.token_requests: list[dict[str, str]] = []
        self.stall: set[str] = set()
        self.stalled_arrived = tonio.Event()
        self._challenges: dict[str, str] = {}
        self._refresh_tokens: set[str] = set()
        self._refresh_failure: tuple[str, Callable[[], Awaitable[None]] | None] | None = None
        self._issued = 0

    def revoke_refresh_tokens(self) -> None:
        """Makes every refresh fail with `invalid_grant`, like a revoked grant."""
        self._refresh_tokens.clear()

    def fail_next_refresh(self, error: str, on_request: Callable[[], Awaitable[None]] | None = None) -> None:
        """Answers the next refresh with the OAuth `error`, after running `on_request`."""
        self._refresh_failure = (error, on_request)

    def _issue_tokens(self) -> dict[str, Any]:
        self._issued += 1
        self._refresh_tokens.add(f"refresh-{self._issued}")
        return {
            "access_token": f"access-{self._issued}",
            "refresh_token": f"refresh-{self._issued}",
            "token_type": "Bearer",
        }

    async def handle(self, request: Any, origin: str) -> None:
        url = urlsplit(request.target)
        query = dict(parse_qsl(url.query, keep_blank_values=True))
        if url.path in self.stall:
            # Read first: the client's hang-up shows only once nothing is left to read.
            await request.read()
            self.stalled_arrived.set()
            await request.peer_closed()
            return
        match url.path:
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
                self.registrations.append(metadata)
                await _json(request, 201, {**metadata, "client_id": f"client-{len(self.registrations)}"})
            case "/authorize":
                self.authorizations.append(query)
                code = f"code-{len(self.authorizations)}"
                self._challenges[code] = query.get("code_challenge", "")
                redirect = parse_url(query.get("redirect_uri", ""))
                redirect = redirect.with_search_param("code", code).with_search_param("state", query.get("state", ""))
                if self._iss_parameter:
                    redirect = redirect.with_search_param("iss", origin)
                await request.respond(302, headers={"location": redirect.href})
            case "/token":
                params = dict(parse_qsl(await read_body(request), keep_blank_values=True))
                self.token_requests.append(params)
                if params.get("grant_type") == "authorization_code":
                    code = params.get("code", "")
                    digest = hashlib.sha256(params.get("code_verifier", "").encode()).digest()
                    verifier = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
                    if self._challenges.get(code) != verifier:
                        await _json(request, 400, {"error": "invalid_grant"})
                        return
                    del self._challenges[code]
                    await _json(request, 200, self._issue_tokens())
                    return
                failure, self._refresh_failure = self._refresh_failure, None
                if failure is not None:
                    error, on_request = failure
                    if on_request is not None:
                        await on_request()
                    await _json(request, 400, {"error": error})
                    return
                refresh = params.get("refresh_token", "")
                if refresh not in self._refresh_tokens:
                    await _json(request, 400, {"error": "invalid_grant"})
                    return
                self._refresh_tokens.discard(refresh)
                await _json(request, 200, self._issue_tokens())
            case _:
                await request.respond(404)


async def start_authorization_server(
    servers: HttpServers, *, cimd: bool = False, iss_parameter: bool = False
) -> AuthorizationServer:
    server = AuthorizationServer(cimd=cimd, iss_parameter=iss_parameter)
    server.server_url = f"{await servers.listen(server.handle)}/mcp"
    return server


class _MemoryServerStore(MemoryOAuthStateStore):
    """An in-memory store whose refresh lock only runs `fn`: there is one process."""

    def with_refresh_lock[T](self, fn: Callable[[], Awaitable[T]]) -> Awaitable[T]:
        return fn()


def memory_store() -> McpOAuthServerStore:
    return _MemoryServerStore()


class RecordingStore:
    """An in-memory store that records whether each save happened while the
    refresh lock was held. With `lock_error`, taking the lock fails with it;
    `hold()` stands for another process holding the lock until the returned
    function releases it (bounded by `WAIT_S`). `lock_runs` has an Event per
    call of `with_refresh_lock`, set when that run ends."""

    def __init__(self, lock_error: Exception | None = None) -> None:
        self._memory = MemoryOAuthStateStore()
        self._lock_error = lock_error
        self._guard = threading.Lock()
        self._locked = False
        self._held: tonio.Event | None = None
        self.saves: list[dict[str, Any]] = []
        self.lock_runs: list[tonio.Event] = []

    def load(self) -> Awaitable[McpOAuthState | None]:
        return self._memory.load()

    def save(self, state: McpOAuthState) -> Awaitable[None]:
        self.saves.append({"locked": self._locked, "access_token": (state.get("tokens") or {}).get("access_token")})
        return self._memory.save(state)

    def with_refresh_lock[T](self, fn: Callable[[], Awaitable[T]]) -> Awaitable[T]:
        ended = tonio.Event()
        with self._guard:
            self.lock_runs.append(ended)
            held = self._held

        async def run() -> T:
            try:
                if held is not None:
                    await held.wait(WAIT_S)
                if self._lock_error is not None:
                    raise self._lock_error
                self._locked = True
                try:
                    return await fn()
                finally:
                    self._locked = False
            finally:
                ended.set()

        return run()

    def hold(self) -> Callable[[], None]:
        held = tonio.Event()
        with self._guard:
            self._held = held
        return held.set

    def is_locked(self) -> bool:
        return self._locked


async def authorize_in_browser(authorization_url: str) -> str:
    """Follows the authorization URL like a browser, up to the redirect back to the application."""
    response = await http.shared_client().get(authorization_url, follow_redirects=False)
    await response.read()
    return response.headers.get("location") or ""
