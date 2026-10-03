"""Mirror of pi's oauth-callback-server.test.ts. The lower layer, which pi gets
from node's `http.createServer`, is tested in pidrei-http
(`packages/http/tests/test_callback_server.py`).
"""

import errno
from dataclasses import dataclass
from urllib.parse import urlencode

import pytest
import tonio.colored as tonio
from tonio.colored import net

from pidrei_ai.auth.oauth.callback_server import (
    CallbackOrManualInput,
    start_oauth_callback_server,
    wait_for_callback_or_manual_input,
)
from pidrei_http import http
from pidrei_utils.cancel import CancelToken

from .oauth_helpers import RecordingInteraction


@dataclass(slots=True)
class _Page:
    status: int
    content_type: str | None
    body: str


def callback_url(redirect_uri: str, params: dict[str, str]) -> str:
    return f"{redirect_uri}?{urlencode(params)}"


async def fetch_page(url: str, method: str = "GET") -> _Page:
    client = http.create_client(timeout=http.oneshot_timeout(5_000), trust_env=False)
    try:
        response = await client.request(method, url)
        body = await response.read()
        return _Page(response.status_code, response.headers.get("content-type"), body.decode("utf-8"))
    finally:
        await client.close()


async def completed(code: str) -> str:
    return f"completed:{code}"


async def echo(code: str) -> str:
    return code


def start(**options):
    return start_oauth_callback_server(
        **{
            "provider_name": "Example",
            "host": "127.0.0.1",
            "port": 0,
            "path": "/callback",
            "state": "expected-state",
            "complete": completed,
            **options,
        }
    )


def pending_prompt(on_prompt=None):
    """A manual prompt that stays open until its cancel token fires."""

    async def prompt(prompt):
        if on_prompt is not None:
            on_prompt(prompt)
        done = tonio.Event()
        prompt.cancel.on_cancel(lambda _reason: done.set())
        await done.wait()
        raise RuntimeError("prompt aborted")

    return prompt


@pytest.mark.tonio
async def test_ignores_stray_requests_and_resolves_with_the_completed_code():
    server = await start()
    try:
        assert server.redirect_uri.startswith("http://127.0.0.1:")
        assert server.redirect_uri.endswith("/callback")

        wrong_path = await fetch_page(server.redirect_uri.removesuffix("/callback") + "/other")
        assert wrong_path.status == 404
        wrong_state = await fetch_page(callback_url(server.redirect_uri, {"code": "c", "state": "other"}))
        assert (wrong_state.status, wrong_state.content_type) == (400, "text/html; charset=utf-8")
        assert "State mismatch." in wrong_state.body
        post = await fetch_page(callback_url(server.redirect_uri, {"code": "c", "state": "expected-state"}), "POST")
        assert post.status == 404
        missing_code = await fetch_page(callback_url(server.redirect_uri, {"state": "expected-state"}))
        assert missing_code.status == 400

        success = await fetch_page(callback_url(server.redirect_uri, {"code": "the-code", "state": "expected-state"}))
        assert (success.status, success.content_type) == (200, "text/html; charset=utf-8")
        assert "Authentication successful" in success.body
        assert "Signed in to Example." in success.body
        assert 'fill="#F09082"' in success.body
        assert 'fill="#4D9ABF"' in success.body
        assert 'fill="#F1BE58"' in success.body
        assert await server.wait() == "completed:the-code"
    finally:
        server.close()


@pytest.mark.tonio
async def test_uses_the_redirect_host_and_skips_the_state_check_when_none_is_expected():
    server = await start(redirect_host="localhost", state=None)
    try:
        assert server.redirect_uri.startswith("http://localhost:")
        assert server.redirect_uri.endswith("/callback")
        # the listener is on 127.0.0.1; `localhost` is only what the provider is told
        response = await fetch_page(
            callback_url(server.redirect_uri.replace("localhost", "127.0.0.1", 1), {"code": "no-state"})
        )
        assert response.status == 200
        assert await server.wait() == "completed:no-state"
    finally:
        server.close()


@pytest.mark.tonio
async def test_shows_completion_failures_on_the_page_and_rejects_the_wait():
    async def failing(_code: str) -> str:
        raise RuntimeError("token exchange failed")

    server = await start(complete=failing)
    try:
        failure = await fetch_page(callback_url(server.redirect_uri, {"code": "c", "state": "expected-state"}))
        assert failure.status == 502
        assert "Example sign-in failed." in failure.body
        assert "token exchange failed" in failure.body
        with pytest.raises(RuntimeError, match="token exchange failed"):
            await server.wait()
    finally:
        server.close()


@pytest.mark.tonio
async def test_rejects_the_wait_when_the_provider_redirects_with_an_error():
    server = await start()
    try:
        failure = await fetch_page(
            callback_url(
                server.redirect_uri,
                {"error": "access_denied", "error_description": "User denied access", "state": "expected-state"},
            )
        )
        assert failure.status == 400
        assert "User denied access" in failure.body
        with pytest.raises(RuntimeError, match="Example authorization failed: User denied access"):
            await server.wait()
    finally:
        server.close()


@pytest.mark.tonio
async def test_completes_only_the_first_callback():
    exchange_started = tonio.Event()
    release = tonio.Event()

    async def blocking(_code: str) -> str:
        exchange_started.set()
        await release.wait(5)
        return "done"

    server = await start(complete=blocking)
    url = callback_url(server.redirect_uri, {"code": "c", "state": "expected-state"})

    async def second_request() -> int:
        await exchange_started.wait(5)
        assert exchange_started.is_set()
        second = await fetch_page(url)
        # A claimed callback keeps completing even when the caller switches to manual input.
        server.cancel()
        release.set()
        return second.status

    try:
        first, second_status = await tonio.spawn(fetch_page(url), second_request())
        assert second_status == 409
        assert first.status == 200
        assert await server.wait() == "done"
    finally:
        server.close()


@pytest.mark.tonio
async def test_resolves_with_none_after_cancel():
    server = await start()
    try:
        server.cancel()
        assert await server.wait() is None
        late = await fetch_page(callback_url(server.redirect_uri, {"code": "c", "state": "expected-state"}))
        assert late.status == 409
    finally:
        server.close()


@pytest.mark.tonio
async def test_rejects_the_wait_on_abort_and_on_timeout():
    cancel = CancelToken()
    aborted = await start(cancel=cancel)
    timed_out = await start(timeout_ms=10)
    try:
        cancel.cancel()
        with pytest.raises(RuntimeError, match="Login cancelled"):
            await aborted.wait()

        with pytest.raises(RuntimeError, match="Example sign-in timed out"):
            await timed_out.wait()

        already_aborted = CancelToken()
        already_aborted.cancel()
        with pytest.raises(RuntimeError, match="Login cancelled"):
            await start(cancel=already_aborted)
    finally:
        aborted.close()
        timed_out.close()


@pytest.mark.tonio
async def test_fails_instead_of_picking_another_port_when_the_requested_port_is_taken():
    blockers = await net.open_tcp_listeners(0, host="127.0.0.1")
    try:
        port = blockers[0].socket.getsockname()[1]
        with pytest.raises(OSError) as raised:
            await start(port=port)
        assert raised.value.errno == errno.EADDRINUSE
    finally:
        for blocker in blockers:
            blocker.close()


# waitForCallbackOrManualInput


@pytest.mark.tonio
async def test_returns_the_browser_callback_and_aborts_the_manual_prompt():
    manual_cancels = []
    server = await start_oauth_callback_server(
        provider_name="Example", host="127.0.0.1", port=0, path="/callback", complete=echo
    )
    try:
        interaction = RecordingInteraction(prompt=pending_prompt(lambda prompt: manual_cancels.append(prompt.cancel)))
        result, _ = await tonio.spawn(
            wait_for_callback_or_manual_input(interaction, server, message="paste", placeholder=server.redirect_uri),
            fetch_page(callback_url(server.redirect_uri, {"code": "from-browser"})),
        )
        assert result == CallbackOrManualInput(type="callback", value="from-browser")
        assert manual_cancels and manual_cancels[0].cancelled
    finally:
        server.close()


@pytest.mark.tonio
async def test_returns_pasted_input_and_stops_waiting_for_the_browser():
    server = await start_oauth_callback_server(
        provider_name="Example", host="127.0.0.1", port=0, path="/callback", complete=echo
    )
    try:
        result = await wait_for_callback_or_manual_input(
            RecordingInteraction(prompt=lambda _prompt: "pasted"),
            server,
            message="paste",
            placeholder=server.redirect_uri,
        )
        assert result == CallbackOrManualInput(type="manual", input="pasted")
    finally:
        server.close()


@pytest.mark.tonio
async def test_uses_only_the_manual_prompt_without_a_callback_server():
    result = await wait_for_callback_or_manual_input(
        RecordingInteraction(prompt=lambda _prompt: "pasted"),
        None,
        message="paste",
        placeholder="http://localhost/callback",
    )
    assert result == CallbackOrManualInput(type="manual", input="pasted")


@pytest.mark.tonio
async def test_propagates_manual_prompt_failures():
    def failing(_prompt):
        raise RuntimeError("prompt cancelled")

    server = await start_oauth_callback_server(
        provider_name="Example", host="127.0.0.1", port=0, path="/callback", complete=echo
    )
    try:
        with pytest.raises(RuntimeError, match="prompt cancelled"):
            await wait_for_callback_or_manual_input(
                RecordingInteraction(prompt=failing), server, message="paste", placeholder=server.redirect_uri
            )
    finally:
        server.close()
