"""Mirror of pi's openrouter-oauth.test.ts.

The callback really is fetched over the loopback interface, as in pi; only the
token exchange is stubbed. pi's `notify` starts that fetch without awaiting it,
which here means spawning it from the (synchronous) notify callback.

The two cases that also assert the *images* provider are ported for the text
provider only: `openrouter-images` lands with the image models in Phase 5d
(PLAN.md).
"""

import base64
import hashlib
import re
from typing import Any

import pytest
import tonio.colored as tonio

from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.oauth.openrouter import MAX_SAFE_INTEGER, openrouter_oauth
from pidrei_ai.auth.types import AuthEvent, OAuthCredential
from pidrei_ai.providers.openrouter import openrouter_provider
from pidrei_ai.registry import create_models
from pidrei_ai.utils import http
from pidrei_ai.utils.cancel import CancelToken

from .oauth_helpers import OAuthRequest, RecordingInteraction, json_response, process_env, stub_oauth_http


TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"


def base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


async def fetch_status(url: str) -> int:
    client = http.create_client(trust_env=False)
    try:
        response = await client.get(url)
        await response.read()
        return response.status_code
    finally:
        await client.close()


class _CallbackDriver:
    """Fetches the callback URL as soon as the flow announces it, like pi's notify."""

    def __init__(self, query: dict[str, str] | None = None):
        self.authorize_url: str | None = None
        self.callback_url: str | None = None
        self.status: list[Any] = []
        self.announced = tonio.Event()
        self.answered = tonio.Event()
        self._query = query or {"code": "authorization-code"}

    def notify(self, event: AuthEvent) -> None:
        if event.type != "auth_url":
            return
        self.authorize_url = event.url
        self.callback_url = callback_url_of(event.url)
        target = self.callback_url + "?" + "&".join(f"{name}={value}" for name, value in self._query.items())
        self.announced.set()

        async def run() -> None:
            try:
                self.status.append(await fetch_status(target))
            except Exception as error:  # pragma: no cover - reported through `status`
                self.status.append(error)
            self.answered.set()

        tonio.spawn.without_tracking(run())

    async def wait(self) -> None:
        """pi awaits its `callbackResponse` promise; the page is written after the
        flow settles, so the test has to wait for it too."""
        await self.answered.wait()


def callback_url_of(authorize_url: str) -> str:
    match = re.search(r"[?&]callback_url=([^&]+)", authorize_url)
    assert match, f"No callback_url in {authorize_url}"
    from urllib.parse import unquote

    return unquote(match.group(1))


def authorize_params(authorize_url: str) -> dict[str, str]:
    from urllib.parse import parse_qs, urlsplit

    query = parse_qs(urlsplit(authorize_url).query)
    return {name: values[0] for name, values in query.items()}


def test_is_exposed_by_the_openrouter_provider_alongside_api_key_auth():
    provider = openrouter_provider()
    assert provider.auth.api_key is not None
    assert provider.auth.oauth is not None
    assert provider.auth.oauth.login_label == "Sign in with OpenRouter"


@pytest.mark.tonio
async def test_resolves_the_stored_oauth_key_for_the_text_provider():
    credentials = InMemoryCredentialStore()
    await credentials.modify(
        "openrouter",
        lambda _current: _stored(OAuthCredential(access="sk-or-stored", refresh="", expires=MAX_SAFE_INTEGER)),
    )

    models = create_models(credentials=credentials)
    models.set_provider(openrouter_provider())

    result = await models.get_auth("openrouter")
    assert result is not None and result.auth.api_key == "sk-or-stored"


async def _stored(credential: OAuthCredential) -> OAuthCredential:
    return credential


@pytest.mark.tonio
async def test_runs_pkce_on_a_one_shot_callback_and_exchanges_the_code_for_a_permanent_key():
    exchange_bodies: list[Any] = []

    def handler(request: OAuthRequest):
        assert request.url == TOKEN_URL
        exchange_bodies.append(request.json_body)
        return json_response({"key": "sk-or-test"})

    driver = _CallbackDriver()
    interaction = RecordingInteraction()
    interaction.notify = driver.notify  # type: ignore[method-assign]

    with stub_oauth_http(handler) as calls:
        credential = await openrouter_oauth.login(interaction)

    assert credential == OAuthCredential(access="sk-or-test", refresh="", expires=MAX_SAFE_INTEGER)
    await driver.wait()
    assert driver.status == [200]

    params = authorize_params(driver.authorize_url or "")
    assert (driver.authorize_url or "").startswith("https://openrouter.ai/auth?")
    assert params["code_challenge_method"] == "S256"

    callback = driver.callback_url or ""
    assert callback.startswith("http://127.0.0.1:")
    assert re.search(r"/oauth/callback/[0-9a-f-]+$", callback)

    body = exchange_bodies[0]
    assert body["code"] == "authorization-code"
    assert body["code_challenge_method"] == "S256"
    verifier = body["code_verifier"]
    assert isinstance(verifier, str)
    assert params["code_challenge"] == base64url(hashlib.sha256(verifier.encode()).digest())
    assert len(calls) == 1


@pytest.mark.tonio
async def test_reports_token_exchange_failures_through_both_the_callback_page_and_login():
    def handler(_request: OAuthRequest):
        return json_response({"error": {"message": "invalid code"}}, 403)

    driver = _CallbackDriver({"code": "bad-code"})
    interaction = RecordingInteraction()
    interaction.notify = driver.notify  # type: ignore[method-assign]

    with (
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match=r"OpenRouter OAuth key exchange failed \(HTTP 403\): invalid code"),
    ):
        await openrouter_oauth.login(interaction)

    await driver.wait()
    assert driver.status == [502]


@pytest.mark.tonio
async def test_allows_only_one_token_exchange_for_a_callback():
    exchange_started = tonio.Event()
    release = tonio.Event()

    async def handler(_request: OAuthRequest):
        exchange_started.set()
        await release.wait()
        return json_response({"key": "sk-or-test"})

    driver = _CallbackDriver()
    interaction = RecordingInteraction()
    interaction.notify = driver.notify  # type: ignore[method-assign]

    second_status: list[int] = []

    async def drive() -> None:
        await exchange_started.wait()
        second_status.append(await fetch_status(driver.callback_url or ""))
        release.set()

    with stub_oauth_http(handler) as calls:
        credential, _ = await tonio.spawn(openrouter_oauth.login(interaction), drive())

    assert second_status == [409]
    assert len(calls) == 1
    assert credential.access == "sk-or-test"
    await driver.wait()
    assert driver.status == [200]


@pytest.mark.tonio
async def test_rejects_a_successful_response_that_does_not_contain_a_key():
    def handler(_request: OAuthRequest):
        return json_response({"user_id": "user-1"})

    driver = _CallbackDriver({"code": "code-without-key"})
    interaction = RecordingInteraction()
    interaction.notify = driver.notify  # type: ignore[method-assign]

    with (
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match='OpenRouter OAuth response carries no "key"'),
    ):
        await openrouter_oauth.login(interaction)

    await driver.wait()
    assert driver.status == [502]


@pytest.mark.tonio
async def test_closes_the_pending_callback_when_login_is_cancelled():
    cancel = CancelToken()
    seen: list[str] = []

    def notify(event: AuthEvent) -> None:
        if event.type != "auth_url":
            return
        seen.append(callback_url_of(event.url))
        cancel.cancel()

    interaction = RecordingInteraction(cancel=cancel)
    interaction.notify = notify  # type: ignore[method-assign]

    def handler(_request: OAuthRequest):
        raise AssertionError("Cancelled login must not exchange a code")

    with stub_oauth_http(handler), pytest.raises(RuntimeError, match="Login cancelled"):
        await openrouter_oauth.login(interaction)

    assert seen
    # the socket is gone; punkreq's connect error type is its own
    with pytest.raises(Exception):
        await fetch_status(seen[0])


@pytest.mark.tonio
async def test_rejects_before_opening_a_callback_server_when_login_is_already_cancelled():
    cancel = CancelToken()
    cancel.cancel()

    def notify(_event: AuthEvent) -> None:
        raise AssertionError("Cancelled login must not emit events")

    interaction = RecordingInteraction(cancel=cancel)
    interaction.notify = notify  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Login cancelled"):
        await openrouter_oauth.login(interaction)


@pytest.mark.tonio
async def test_uses_the_configured_oauth_callback_host():
    cancel = CancelToken()
    seen: list[str] = []

    def notify(event: AuthEvent) -> None:
        if event.type != "auth_url":
            return
        seen.append(callback_url_of(event.url))
        cancel.cancel()

    interaction = RecordingInteraction(cancel=cancel)
    interaction.notify = notify  # type: ignore[method-assign]

    with (
        process_env(PIDREI_OAUTH_CALLBACK_HOST="localhost"),
        pytest.raises(RuntimeError, match="Login cancelled"),
    ):
        await openrouter_oauth.login(interaction)

    assert seen and seen[0].startswith("http://localhost:")
