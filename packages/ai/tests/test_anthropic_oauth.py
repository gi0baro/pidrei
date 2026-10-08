"""Mirror of pi's anthropic-oauth.test.ts.

Every browser-login case really opens the loopback callback server on the
preferred port (or a free one when it is taken) — pi's suite does the same —
and the browser-callback cases fetch it.
"""

from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

import pytest
import tonio.colored as tonio
from tonio.colored import net

from pidrei_ai.auth.oauth.anthropic import anthropic_oauth
from pidrei_ai.auth.types import AuthEvent, AuthPrompt, AuthPromptOption, OAuthCredential
from pidrei_http import http

from .oauth_helpers import (
    DEFAULT_START_MS,
    OAuthRequest,
    RecordingInteraction,
    json_response,
    stub_oauth_http,
    virtual_clock,
)


TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
_WAIT_S = 10


def auth_url_params(interaction: RecordingInteraction) -> dict[str, str]:
    events = interaction.events_of("auth_url")
    assert events, "Expected an auth_url event"
    query = parse_qs(urlsplit(events[0].url).query)
    return {name: values[0] for name, values in query.items()}


@pytest.mark.tonio
async def test_keeps_the_localhost_redirect_uri_for_manual_callback_login():
    def handler(request: OAuthRequest):
        assert request.url == TOKEN_URL
        assert request.method == "POST"
        assert request.json_body["grant_type"] == "authorization_code"
        assert request.json_body["code"] == "manual-code"
        assert request.json_body["redirect_uri"] == "http://localhost:53692/callback"
        return json_response({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600})

    interaction = RecordingInteraction()

    def prompt(prompt: AuthPrompt) -> str:
        if prompt.type == "select":
            return "browser"
        if prompt.type != "manual_code":
            raise AssertionError(f"Unexpected prompt: {prompt.type}")
        params = auth_url_params(interaction)
        assert params["state"] and params["redirect_uri"]
        return f"{params['redirect_uri']}?code=manual-code&state={params['state']}"

    # the prompt needs the recorded auth_url, so it is attached after construction
    interaction._prompt = prompt

    with virtual_clock(), stub_oauth_http(handler) as calls:
        credential = await anthropic_oauth.login(interaction)

    assert credential.access == "access-token"
    assert credential.refresh == "refresh-token"
    assert len(calls) == 1


@pytest.mark.tonio
async def test_offers_browser_login_first_and_uses_the_selected_anthropic_copy_code_flow():
    interaction = RecordingInteraction()

    def handler(request: OAuthRequest):
        assert request.url == TOKEN_URL
        assert request.json_body["grant_type"] == "authorization_code"
        assert request.json_body["code"] == "copied-code"
        assert request.json_body["state"] == auth_url_params(interaction)["state"]
        assert request.json_body["redirect_uri"] == "https://platform.claude.com/oauth/code/callback"
        return json_response({"access_token": "access-token", "refresh_token": "refresh-token", "expires_in": 3600})

    select_prompts: list[AuthPrompt] = []

    def prompt(prompt: AuthPrompt) -> str:
        if prompt.type == "select":
            select_prompts.append(prompt)
            return "copy_code"
        if prompt.type != "manual_code":
            raise AssertionError(f"Unexpected prompt: {prompt.type}")
        return f"copied-code#{auth_url_params(interaction)['state']}"

    # the prompt needs the recorded auth_url, so it is attached after construction
    interaction._prompt = prompt

    with virtual_clock(), stub_oauth_http(handler) as calls:
        credential = await anthropic_oauth.login(interaction)

    assert credential.access == "access-token"
    assert credential.refresh == "refresh-token"
    assert auth_url_params(interaction)["redirect_uri"] == "https://platform.claude.com/oauth/code/callback"
    assert len(calls) == 1
    assert select_prompts == [
        AuthPrompt(
            type="select",
            message="Select Anthropic login method:",
            options=[
                AuthPromptOption(id="browser", label="Browser login (default)"),
                AuthPromptOption(id="copy_code", label="Copy code login (headless)"),
            ],
        )
    ]


@pytest.mark.tonio
async def test_cancels_when_anthropic_login_method_selection_is_cancelled():
    def prompt(_prompt: AuthPrompt) -> str:
        raise RuntimeError("Login cancelled")

    with pytest.raises(RuntimeError, match="Login cancelled"):
        await anthropic_oauth.login(RecordingInteraction(prompt=prompt))


@pytest.mark.tonio
async def test_omits_scope_from_refresh_token_requests():
    def handler(request: OAuthRequest):
        assert request.url == TOKEN_URL
        assert request.method == "POST"
        assert request.json_body["grant_type"] == "refresh_token"
        assert request.json_body["client_id"]
        assert request.json_body["refresh_token"] == "refresh-token"
        assert "scope" not in request.json_body
        return json_response(
            {"access_token": "new-access-token", "refresh_token": "new-refresh-token", "expires_in": 3600}
        )

    with virtual_clock(), stub_oauth_http(handler) as calls:
        credential = await anthropic_oauth.refresh(
            OAuthCredential(access="old-access-token", refresh="refresh-token", expires=0), None
        )

    assert credential.access == "new-access-token"
    assert credential.refresh == "new-refresh-token"
    assert credential.expires == DEFAULT_START_MS + 3600 * 1000 - 5 * 60 * 1000
    assert len(calls) == 1


@pytest.mark.tonio
async def test_login_resolves_through_the_manual_code_prompt_and_aborts_it_after_settling():
    def handler(request: OAuthRequest):
        assert "/oauth/token" in request.url
        return json_response({"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})

    def prompt(prompt: AuthPrompt) -> str:
        if prompt.type == "select":
            return "browser"
        if prompt.type != "manual_code":
            raise AssertionError(f"Unexpected prompt: {prompt.type}")
        return "the-code"

    interaction = RecordingInteraction(prompt=prompt)
    with virtual_clock(), stub_oauth_http(handler):
        credential = await anthropic_oauth.login(interaction)

    assert credential.type == "oauth"
    assert credential.access == "access"
    assert interaction.events_of("auth_url")
    manual_prompts = [prompt for prompt in interaction.prompts if prompt.type == "manual_code"]
    assert manual_prompts
    # the prompt's token is cancelled once login settles, so UIs can dismiss it
    assert manual_prompts[0].cancel is not None and manual_prompts[0].cancel.cancelled


@pytest.mark.tonio
async def test_completes_login_through_the_browser_callback_and_shows_the_sign_in_page():
    login = await login_through_browser_callback()

    assert login.credential.access == "access"
    assert login.exchanged_code == "browser-code"
    assert login.exchanged_redirect_uri == login.redirect_uri
    assert login.page_status == 200
    assert "Signed in to Anthropic." in login.page_text


# pi #10571
@pytest.mark.tonio
async def test_falls_back_to_a_free_callback_port_when_the_preferred_port_cannot_be_bound():
    # Occupy the preferred port unless something already holds it; the fallback runs either way.
    try:
        blockers = await net.open_tcp_listeners(53692, host="127.0.0.1")
    except OSError:
        blockers = []
    try:
        login = await login_through_browser_callback()
        redirect_uri = urlsplit(login.redirect_uri)

        assert redirect_uri.hostname == "localhost"
        assert redirect_uri.path == "/callback"
        assert redirect_uri.port != 53692
        assert login.credential.access == "access"
        assert login.exchanged_redirect_uri == login.redirect_uri
        assert login.page_status == 200
    finally:
        for blocker in blockers:
            blocker.close()


@dataclass(slots=True)
class BrowserCallbackLogin:
    credential: OAuthCredential
    redirect_uri: str
    exchanged_code: str | None
    exchanged_redirect_uri: str | None
    page_status: int | None
    page_text: str


async def login_through_browser_callback() -> BrowserCallbackLogin:
    exchanged: list[dict] = []

    def handler(request: OAuthRequest):
        assert request.url == TOKEN_URL
        exchanged.append(request.json_body)
        return json_response({"access_token": "access", "refresh_token": "refresh", "expires_in": 3600})

    redirect_uris: list[str] = []
    callback_page: list[tuple[int, str]] = []
    page_fetched = tonio.Event()

    async def fetch_callback(url: str) -> None:
        client = http.create_client(timeout=http.oneshot_timeout(5_000), trust_env=False)
        try:
            response = await client.get(url)
            callback_page.append((response.status_code, (await response.read()).decode("utf-8")))
        finally:
            await client.close()
            page_fetched.set()

    async def pending_prompt(prompt: AuthPrompt) -> str:
        if prompt.type == "select":
            return "browser"
        # The browser callback settles the login, which cancels this prompt.
        # Bounded: when no callback server could be started, the login falls
        # back to this prompt alone and would otherwise wait forever.
        done = tonio.Event()
        prompt.cancel.on_cancel(lambda _reason: done.set())
        await done.wait(_WAIT_S)
        raise RuntimeError("aborted" if done.is_set() else "the browser callback never settled the login")

    interaction = RecordingInteraction(prompt=pending_prompt)

    def notify(event: AuthEvent) -> None:
        if event.type != "auth_url":
            return
        params = {name: values[0] for name, values in parse_qs(urlsplit(event.url).query).items()}
        redirect_uri = params.get("redirect_uri", "")
        redirect_uris.append(redirect_uri)
        # The callback server listens on 127.0.0.1; the redirect URI names localhost.
        callback_url = urlsplit(redirect_uri)._replace(
            netloc=f"127.0.0.1:{urlsplit(redirect_uri).port}",
            query=urlencode({"code": "browser-code", "state": params.get("state", "")}),
        )
        tonio.spawn.without_tracking(fetch_callback(urlunsplit(callback_url)))

    interaction.notify = notify  # type: ignore[method-assign]

    with stub_oauth_http(handler):
        credential = await anthropic_oauth.login(interaction)

    await page_fetched.wait(_WAIT_S)
    assert page_fetched.is_set()
    return BrowserCallbackLogin(
        credential=credential,
        redirect_uri=redirect_uris[0],
        exchanged_code=exchanged[0]["code"] if exchanged else None,
        exchanged_redirect_uri=exchanged[0]["redirect_uri"] if exchanged else None,
        page_status=callback_page[0][0] if callback_page else None,
        page_text=callback_page[0][1] if callback_page else "",
    )
