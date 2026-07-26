"""Mirror of pi's anthropic-oauth.test.ts.

These cases exercise the manual-paste branch, so the flow really does open its
loopback callback server on the fixed port — pi's suite does the same.
"""

from urllib.parse import parse_qs, urlsplit

import pytest

from pidrei_ai.auth.oauth.anthropic import anthropic_oauth
from pidrei_ai.auth.types import AuthPrompt, OAuthCredential

from .oauth_helpers import (
    DEFAULT_START_MS,
    OAuthRequest,
    RecordingInteraction,
    json_response,
    stub_oauth_http,
    virtual_clock,
)


TOKEN_URL = "https://platform.claude.com/v1/oauth/token"


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
