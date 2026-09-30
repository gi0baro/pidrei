"""Mirror of pi's openai-chatgpt-oauth.test.ts.

The login cases open the real callback server on port 1455 (or fall back to the
paste prompt when it is taken), as pi's do; the prompt answers with the
callback URL, so the browser side is never needed. `agent_name_hint` is
pidrei's name where pi asserts "Pi" (see the flow module).
"""

from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from pidrei_ai.auth.oauth.openai_chatgpt import openai_chatgpt_oauth
from pidrei_ai.auth.types import AuthEvent, AuthPrompt, LoginOptions, OAuthCredential
from pidrei_ai.utils.user_agent import CLIENT_NAME

from .oauth_helpers import (
    DEFAULT_START_MS,
    OAuthRequest,
    RecordingInteraction,
    json_response,
    stub_oauth_http,
    virtual_clock,
)


TOKEN_URL = "https://auth.openai.com/api/accounts/oauth/token"
REQUIRED_SCOPE = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
DEVICE_ID = "e61bbe28-07ef-466d-8e5d-a344f94ab305"
OPTIONS = LoginOptions(get_device_id=lambda: DEVICE_ID)


def token_response(scope: str = REQUIRED_SCOPE) -> dict[str, Any]:
    return {
        "access_token": "access-token",
        "refresh_token": "refresh-token",
        "expires_in": 3600,
        "id_token": "id-token",
        "scope": scope,
    }


def stub_token_endpoint(response: dict[str, Any], inspect=None):
    def handler(request: OAuthRequest):
        assert request.url == TOKEN_URL
        if inspect is not None:
            inspect(request.form)
        return json_response(response)

    return stub_oauth_http(handler)


def query_of(url: str) -> dict[str, str]:
    return {name: values[0] for name, values in parse_qs(urlsplit(url).query).items()}


def login_interaction(*, callback_client_id: str | None = None, on_authorize=None) -> RecordingInteraction:
    authorize: list[dict[str, str]] = []

    def prompt(prompt: AuthPrompt) -> str:
        if prompt.type != "manual_code":
            raise AssertionError(f"Unexpected prompt: {prompt.type}")
        if not authorize:
            raise AssertionError("Authorization URL was not emitted before the callback prompt")
        params = {"code": "authorization-code", "state": authorize[0]["state"]}
        if callback_client_id:
            params["client_id"] = callback_client_id
        return f"{authorize[0]['redirect_uri']}?{urlencode(params)}"

    interaction = RecordingInteraction(prompt=prompt)

    def notify(event: AuthEvent) -> None:
        if event.type != "auth_url":
            return
        authorize.append(query_of(event.url))
        if on_authorize is not None:
            on_authorize(authorize[0])

    interaction.notify = notify  # type: ignore[method-assign]
    return interaction


def connected_credential() -> OAuthCredential:
    return OAuthCredential(
        access="old-access",
        refresh="old-refresh",
        expires=0,
        extra={"clientId": "oaiapp_existing", "scopes": REQUIRED_SCOPE.split(" ")},
    )


@pytest.mark.tonio
async def test_registers_a_user_owned_client_and_stores_its_issued_id_and_granted_scopes():
    authorize: list[dict[str, str]] = []
    exchange: list[dict[str, str]] = []

    with stub_token_endpoint(token_response(), exchange.append):
        credential = await openai_chatgpt_oauth.login(
            login_interaction(callback_client_id="oaiapp_issued", on_authorize=authorize.append), OPTIONS
        )

    params = authorize[0]
    assert params["client_id"] == "dynamic_agent_client"
    assert params["agent_name_hint"] == CLIENT_NAME
    assert params["ext_agent_host_id"] == f"urn:uuid:{DEVICE_ID}"
    assert params["scope"] == REQUIRED_SCOPE
    assert params["redirect_uri"] == "http://127.0.0.1:1455/auth/callback"
    assert params["resource"] == "https://api.openai.com/v1"
    assert params["code_challenge_method"] == "S256"
    body = exchange[0]
    assert body["client_id"] == "oaiapp_issued"
    assert body["code"] == "authorization-code"
    assert body["resource"] == "https://api.openai.com/v1"
    assert body["code_verifier"]
    assert credential.type == "oauth"
    assert (credential.access, credential.refresh) == ("access-token", "refresh-token")
    assert credential.extra == {"clientId": "oaiapp_issued", "scopes": REQUIRED_SCOPE.split(" ")}


@pytest.mark.tonio
async def test_rejects_registration_without_an_issued_client_id():
    with (
        stub_token_endpoint(token_response()) as calls,
        pytest.raises(RuntimeError, match="registration callback did not contain an issued client ID"),
    ):
        await openai_chatgpt_oauth.login(login_interaction(), OPTIONS)
    assert calls == []


@pytest.mark.tonio
async def test_rejects_a_token_response_that_did_not_grant_direct_token_use():
    with (
        stub_token_endpoint(token_response("openid profile email offline_access resource.invoke")),
        pytest.raises(RuntimeError, match="grant did not include chatgpt.tokens.use.direct"),
    ):
        await openai_chatgpt_oauth.login(login_interaction(callback_client_id="oaiapp_issued"), OPTIONS)


@pytest.mark.tonio
async def test_requires_a_device_id_before_starting_authorization():
    authorization_started: list[dict[str, str]] = []
    interaction = login_interaction(on_authorize=authorization_started.append)

    with pytest.raises(RuntimeError, match="requires a device ID"):
        await openai_chatgpt_oauth.login(interaction)
    with pytest.raises(RuntimeError, match="requires a device ID"):
        await openai_chatgpt_oauth.login(interaction, LoginOptions(get_device_id=lambda: "not-a-uuid"))
    assert authorization_started == []


@pytest.mark.tonio
async def test_requires_refresh_responses_to_rotate_the_refresh_token():
    response = token_response()
    del response["refresh_token"]

    with stub_token_endpoint(response), pytest.raises(RuntimeError, match="token response has invalid refresh_token"):
        await openai_chatgpt_oauth.refresh(connected_credential(), None)


@pytest.mark.tonio
async def test_refreshes_with_the_credentials_issued_client_id_and_stores_replacement_scopes():
    refresh: list[dict[str, str]] = []

    with (
        virtual_clock(),
        stub_token_endpoint(
            {**token_response(), "access_token": "new-access", "refresh_token": "new-refresh"}, refresh.append
        ),
    ):
        credential = await openai_chatgpt_oauth.refresh(connected_credential(), None)

    # expires_in is 3600 seconds; the credential expires 3 minutes early so it is refreshed in time.
    assert credential.expires == DEFAULT_START_MS + (3600 - 180) * 1000

    body = refresh[0]
    assert body["grant_type"] == "refresh_token"
    assert body["client_id"] == "oaiapp_existing"
    assert body["refresh_token"] == "old-refresh"
    assert body["resource"] == "https://api.openai.com/v1"
    assert "scope" not in body
    assert (credential.access, credential.refresh) == ("new-access", "new-refresh")
    assert credential.extra == {"clientId": "oaiapp_existing", "scopes": REQUIRED_SCOPE.split(" ")}
