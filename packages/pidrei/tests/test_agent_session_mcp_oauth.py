"""Mirror of pi's suite/agent-session-mcp-oauth.test.ts.

The fake authorization server is `mcp_oauth_server.py`. The browser either
follows the authorization redirect to the loopback callback
(`oauth_servers.browse`), or cannot reach it, and the user pastes the URL it
was redirected to (the redirect is read without following it).

Servers and harnesses are opened and closed inside the test body (`_suite`),
never from a fixture: a fixture's teardown is a separate `run_until_complete`,
and nothing parked on a socket may cross that boundary (see `mcp_helpers`).
"""

import contextlib
import json
import re

import pytest
import tonio.colored as tonio
from tonio.colored import net

from pidrei.core.agent_session import ExtensionBindings
from pidrei.core.auth_storage import InMemoryAuthStorageBackend
from pidrei.core.extensions.runner import emit_session_shutdown_event
from pidrei.extensions.mcp import cli, create_mcp_extension
from pidrei.extensions.mcp.cli import McpCommandOptions, run_mcp_command
from pidrei.extensions.mcp.config import LoadedMcpConfig, McpServerEntry
from pidrei.extensions.mcp.oauth import McpOAuthCredentialStore
from pidrei_ai.providers.faux import faux_assistant_message, faux_tool_call
from pidrei_http import http
from pidrei_mcp.url import parse_url
from pidrei_utils import clock

from .harness import create_harness, create_test_ui_context, get_message_text
from .mcp_oauth_server import oauth_mcp_servers


_WAIT_S = 10


@contextlib.asynccontextmanager
async def _suite(monkeypatch):
    """The fake servers and the harnesses a test creates. The harnesses are
    shut down before the servers, so the connections' closes still reach them."""
    async with oauth_mcp_servers(monkeypatch) as servers:
        created: list = []
        try:
            yield servers, created
        finally:
            for harness in created:
                await emit_session_shutdown_event(
                    harness.session.extension_runner, {"type": "session_shutdown", "reason": "quit"}
                )
                harness.cleanup()


async def _redirect_location(url: str) -> str:
    response = await http.shared_client().get(url, follow_redirects=False)
    try:
        return response.headers.get("location") or ""
    finally:
        await response.close()


async def setup(oauth_servers, harnesses, browser: str, oauth: dict | None = None):
    server = await oauth_servers.start()
    backend = InMemoryAuthStorageBackend()
    config = {"url": server.url, "exposure": "direct", **({"oauth": oauth} if oauth else {})}
    entry = McpServerEntry(name="issues", config=config, source="test")
    notifications: list[str] = []
    opened: list[str] = []
    redirects: list = []

    def open_url(url: str) -> None:
        opened.append(url)
        if browser == "follow":
            # The browser follows the authorization redirect to the loopback callback.
            oauth_servers.browse(url)
        else:
            # The browser cannot reach the callback; the user pastes the redirect URL.
            redirects.append(tonio.spawn(_redirect_location(url)))

    def ask(_title, _placeholder=None, opts=None):
        # The paste prompt waits until sign-in completes unless the user pastes the redirect URL.
        async def pasted() -> str:
            return await redirects[-1]

        async def until_cancelled() -> None:
            await opts["signal"].event.wait(_WAIT_S)

        return pasted() if browser == "paste" else until_cancelled()

    async def load_config(_ctx):
        return LoadedMcpConfig(servers=[entry], errors=[])

    harness = await create_harness(
        initial_active_tool_names=[],
        extension_factories=[
            create_mcp_extension(
                load_config=load_config, credentials=McpOAuthCredentialStore(backend), open_url=open_url
            )
        ],
        extension_bindings=ExtensionBindings(
            ui_context=create_test_ui_context(notify=lambda message, *_rest: notifications.append(message), input=ask)
        ),
    )
    harnesses.append(harness)
    return harness, server, notifications, backend, opened


async def call_whoami(harness):
    harness.set_responses(
        [
            faux_assistant_message([faux_tool_call("mcp__issues__whoami", {})], stop_reason="toolUse"),
            faux_assistant_message("done"),
        ]
    )
    before = len(harness.session.messages)
    await harness.session.prompt("who am i")
    return next(message for message in harness.session.messages[before:] if message.role == "toolResult")


async def free_port() -> int:
    listeners = await net.open_tcp_listeners(0, host="127.0.0.1")
    port = listeners[0].socket.getsockname()[1]
    for listener in listeners:
        listener.close()
    return port


def _redirect_uri(url: str) -> str | None:
    return parse_url(url).search_param("redirect_uri")


@pytest.mark.tonio
async def test_signs_in_through_the_browser_refreshes_expired_tokens_and_signs_out(monkeypatch):
    async with _suite(monkeypatch) as (oauth_servers, harnesses):
        harness, server, notifications, backend, _opened = await setup(oauth_servers, harnesses, "follow")

        await harness.session.prompt("/mcp")
        # Startup problems are reported once, pointing to /mcp.
        assert "MCP servers need attention:\n  issues: needs sign-in\nRun /mcp to fix." in notifications
        assert notifications[-1] == "issues: needs sign-in, run /mcp login issues (direct)"

        await harness.session.prompt("/mcp login issues")
        assert notifications[-1] == 'Signed in to MCP server "issues" (1 tools).'
        assert server.log == ["401 none", "register", "token code"]
        assert '"access_token": "access-1"' in backend.with_lock(lambda current: (current, None))

        assert get_message_text(await call_whoami(harness)) == "token access-1"

        # An expired access token is refreshed without user interaction.
        server.expire_access_tokens()
        assert get_message_text(await call_whoami(harness)) == "token access-2"
        assert server.log[-3:] == ["401 access-1", "token refresh", "call access-2"]

        # A token past its expiry is refreshed before the request, without a 401 round trip.
        def expire(current):
            states = json.loads(current or "{}")
            for state in states.values():
                state["tokensExpireAt"] = clock.now_ms() - 1_000
            return None, json.dumps(states)

        backend.with_lock(expire)
        assert get_message_text(await call_whoami(harness)) == "token access-3"
        assert server.log[-2:] == ["token refresh", "call access-3"]

        await harness.session.prompt("/mcp logout issues")
        assert notifications[-1] == 'Signed out of MCP server "issues".'
        result = await call_whoami(harness)
        assert result.is_error is True
        assert get_message_text(result) == 'MCP server "issues" requires sign-in. Run /mcp to sign in.'


async def shutdown(harness) -> float:
    """Emit session_shutdown like quitting does; returns how long the handlers took, in seconds."""
    start = clock.monotonic()
    await emit_session_shutdown_event(harness.session.extension_runner, {"type": "session_shutdown", "reason": "quit"})
    return clock.monotonic() - start


# pi #10565
@pytest.mark.tonio
@pytest.mark.parametrize("path", ["/.well-known/oauth-authorization-server", "/token"])
async def test_cancels_a_sign_in_waiting_on_a_stalled_request_when_the_session_shuts_down(monkeypatch, path):
    async with _suite(monkeypatch) as (oauth_servers, harnesses):
        harness, server, notifications, _backend, _opened = await setup(oauth_servers, harnesses, "follow")
        server.stall.add(path)
        login = tonio.spawn(harness.session.prompt("/mcp login issues"))
        await server.stalled_arrived.wait(_WAIT_S)
        assert [stalled_path for stalled_path, _hung_up in server.stalled] == [path]

        # Shutdown waits for the sign-in to clean up, which cancelling makes immediate.
        assert await shutdown(harness) < 2
        await login
        # The request is aborted instead of left open, and the ended session reports nothing.
        hung_up = server.stalled[0][1]
        await hung_up.wait(_WAIT_S)
        assert hung_up.is_set()
        assert [message for message in notifications if message.startswith("Sign-in")] == []


# pi #10565
@pytest.mark.tonio
async def test_closes_the_session_without_refreshing_an_expiring_token(monkeypatch):
    async with _suite(monkeypatch) as (oauth_servers, harnesses):
        harness, server, notifications, backend, _opened = await setup(oauth_servers, harnesses, "follow")
        await harness.session.prompt("/mcp login issues")
        assert notifications[-1] == 'Signed in to MCP server "issues" (1 tools).'

        # Still accepted, but close enough to expiry that the next request would refresh it first.
        def expiring(current):
            states = json.loads(current or "{}")
            for state in states.values():
                state["tokensExpireAt"] = clock.now_ms() + 10_000
            return None, json.dumps(states)

        backend.with_lock(expiring)
        server.stall.add("/token")

        assert await shutdown(harness) < 2
        assert server.stalled == []
        assert server.deletes == ["access-1"]


@pytest.mark.tonio
async def test_accepts_a_pasted_redirect_url_when_the_browser_cannot_reach_the_callback(monkeypatch):
    async with _suite(monkeypatch) as (oauth_servers, harnesses):
        harness, server, notifications, _backend, _opened = await setup(oauth_servers, harnesses, "paste")

        await harness.session.prompt("/mcp login")
        assert notifications[-1] == 'Signed in to MCP server "issues" (1 tools).'
        assert get_message_text(await call_whoami(harness)) == "token access-1"
        assert "token code" in server.log


@pytest.mark.tonio
async def test_uses_the_configured_callback_url_and_scope(monkeypatch):
    async with _suite(monkeypatch) as (oauth_servers, harnesses):
        callback_url = f"http://localhost:{await free_port()}/callback"
        harness, _server, notifications, _backend, opened = await setup(
            oauth_servers, harnesses, "follow", {"callbackUrl": callback_url, "scope": "issues:read"}
        )

        await harness.session.prompt("/mcp login issues")
        assert notifications[-1] == 'Signed in to MCP server "issues" (1 tools).'
        assert _redirect_uri(opened[0]) == callback_url
        assert parse_url(opened[0]).search_param("scope") == "issues:read"
        assert get_message_text(await call_whoami(harness)) == "token access-1"


@pytest.mark.tonio
async def test_registers_with_the_configured_client_name(monkeypatch):
    async with _suite(monkeypatch) as (oauth_servers, harnesses):
        # #10226
        harness, server, notifications, _backend, _opened = await setup(
            oauth_servers, harnesses, "follow", {"clientName": "Claude Code"}
        )
        await harness.session.prompt("/mcp login issues")
        assert notifications[-1] == 'Signed in to MCP server "issues" (1 tools).'
        assert [metadata["client_name"] for metadata in server.registrations] == ["Claude Code"]

        fallback, fallback_server, *_ = await setup(oauth_servers, harnesses, "follow")
        await fallback.session.prompt("/mcp login issues")
        assert [metadata["client_name"] for metadata in fallback_server.registrations] == ["pidrei"]
        # OpenID Connect servers reject the loopback redirect URI of a `web` client (#10493).
        assert [metadata["application_type"] for metadata in fallback_server.registrations] == ["native"]


@pytest.mark.tonio
async def test_adds_the_listening_port_to_a_callback_url_without_one(monkeypatch):
    async with _suite(monkeypatch) as (oauth_servers, harnesses):
        harness, _server, notifications, _backend, opened = await setup(
            oauth_servers, harnesses, "follow", {"callbackUrl": "http://127.0.0.1/oauth/done"}
        )
        await harness.session.prompt("/mcp login issues")
        assert notifications[-1] == 'Signed in to MCP server "issues" (1 tools).'
        assert re.fullmatch(r"http://127\.0\.0\.1:\d+/oauth/done", _redirect_uri(opened[0]) or "")

        port = await free_port()
        fixed, _fixed_server, _fixed_notifications, _fixed_backend, fixed_opened = await setup(
            oauth_servers, harnesses, "follow", {"callbackUrl": "http://127.0.0.1/oauth/done", "callbackPort": port}
        )
        await fixed.session.prompt("/mcp login issues")
        assert _redirect_uri(fixed_opened[0]) == f"http://127.0.0.1:{port}/oauth/done"


class _ManualTimeout:
    """`cli.Timeout` that the test fires by hand (pi waits out a real 0.5 s)."""

    def __init__(self, delay_ms: float, fn) -> None:
        self.delay_ms = delay_ms
        self._fn = fn
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def fire(self) -> None:
        if not self._cancelled:
            self._fn()


@pytest.mark.tonio
async def test_gives_up_on_pidrei_mcp_login_after_timeout_also_while_the_authorization_server_hangs(
    monkeypatch, tmp_path
):
    timeouts: list[_ManualTimeout] = []

    def create_timeout(delay_ms: float, fn, on_error=None) -> _ManualTimeout:
        timeout = _ManualTimeout(delay_ms, fn)
        timeouts.append(timeout)
        return timeout

    monkeypatch.setattr(cli, "Timeout", create_timeout)
    async with oauth_mcp_servers(monkeypatch) as oauth_servers:
        server = await oauth_servers.start()
        agent_dir = str(tmp_path)
        (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": {"issues": {"url": server.url}}}))

        def login(open_url):
            output: list[str] = []

            async def run() -> tuple[int, list[str]]:
                exit_code = await run_mcp_command(
                    ["login", "issues", "--timeout", "0.5"],
                    McpCommandOptions(
                        cwd=agent_dir,
                        agent_dir=agent_dir,
                        credentials=McpOAuthCredentialStore(InMemoryAuthStorageBackend()),
                        open_url=open_url,
                        log=output.append,
                        error=output.append,
                    ),
                )
                return exit_code, output

            return run()

        # The user never approves in the browser: the timeout runs out while the sign-in waits for it.
        def never_approve(_url: str) -> None:
            timeouts[-1].fire()

        exit_code, output = await login(never_approve)
        assert timeouts[-1].delay_ms == 500
        assert exit_code == 1
        assert "was cancelled or not completed within" in output[-1]

        # pi #10565: the timeout also ends a request to an authorization server that hangs.
        server.stall.add("/.well-known/oauth-authorization-server")
        stalled = tonio.spawn(login(lambda _url: None))
        await server.stalled_arrived.wait(_WAIT_S)
        assert server.stalled_arrived.is_set()
        timeouts[-1].fire()
        exit_code, output = await stalled
        assert exit_code == 1
        assert "was cancelled or not completed within" in output[-1]


@pytest.mark.tonio
async def test_uses_credentials_from_pidrei_mcp_login_on_the_next_turn(monkeypatch, tmp_path):
    async with _suite(monkeypatch) as (oauth_servers, harnesses):
        harness, server, _notifications, backend, _opened = await setup(oauth_servers, harnesses, "follow")
        agent_dir = str(tmp_path)
        (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": {"issues": {"url": server.url}}}))

        # The agent runs `pidrei mcp login issues` through bash; the user approves in the browser.
        output: list[str] = []
        exit_code = await run_mcp_command(
            ["login", "issues"],
            McpCommandOptions(
                cwd=agent_dir,
                agent_dir=agent_dir,
                credentials=McpOAuthCredentialStore(backend),
                open_url=oauth_servers.browse,
                log=output.append,
                error=output.append,
            ),
        )
        assert exit_code == 0
        assert output[-1] == 'Signed in to MCP server "issues" (1 tools).'

        # The session still waits for a sign-in, and reconnects when the next turn starts.
        assert get_message_text(await call_whoami(harness)) == "token access-1"
