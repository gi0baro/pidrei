"""Mirror of pi's github-copilot-oauth.test.ts."""

from typing import Any

import pytest

from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.oauth.github_copilot import github_copilot_oauth
from pidrei_ai.auth.types import AuthPrompt, OAuthCredential
from pidrei_ai.providers.github_copilot import github_copilot_provider
from pidrei_ai.registry import create_models
from pidrei_ai.utils import clock

from .oauth_helpers import (
    DEFAULT_START_MS,
    OAuthRequest,
    RecordingInteraction,
    json_response,
    stub_oauth_http,
    text_response,
    virtual_clock,
)


COPILOT_TOKEN = "tid=test;exp=9999999999;proxy-ep=proxy.individual.githubcopilot.com;"
COPILOT_MODELS_URL = "https://api.individual.githubcopilot.com/models"


def blank_enterprise_prompt(prompt: AuthPrompt) -> str:
    if prompt.type != "text":
        raise AssertionError(f"Unexpected prompt: {prompt.type}")
    return ""


def device_code_body(**overrides: Any) -> dict[str, Any]:
    return {
        "device_code": "device-code",
        "user_code": "ABCD-EFGH",
        "verification_uri": "https://github.com/login/device",
        "interval": 1,
        "expires_in": 900,
        **overrides,
    }


def device_code_events(interaction: RecordingInteraction) -> list[tuple]:
    return [
        (event.user_code, event.verification_uri, event.interval_seconds, event.expires_in_seconds)
        for event in interaction.events_of("device_code")
    ]


async def _store(credential: OAuthCredential) -> OAuthCredential:
    return credential


async def refresh_github_copilot_models_for_test(
    data: list[dict[str, Any]],
    proxy_host: str = "proxy.individual.githubcopilot.com",
) -> OAuthCredential:
    access_token = f"tid=test;exp=9999999999;proxy-ep={proxy_host};"
    models_url = f"https://{proxy_host.replace('proxy.', 'api.', 1)}/models"

    def handler(request: OAuthRequest):
        if "/copilot_internal/v2/token" in request.url:
            return json_response({"token": access_token, "expires_at": 9999999999})

        assert request.url == models_url
        assert request.headers["Authorization"] == f"Bearer {access_token}"
        return json_response({"data": data})

    with virtual_clock(), stub_oauth_http(handler):
        return await github_copilot_oauth.refresh(
            OAuthCredential(access="old-access-token", refresh="ghu_refresh_token", expires=0), None
        )


def _require_model_id(models: list, index: int) -> str:
    """Catalog-sensitive ids come from the generated models, not literals."""
    assert len(models) > index, f"Expected a GitHub Copilot model at index {index}"
    return models[index].id


@pytest.mark.tonio
async def test_filters_models_to_the_authenticated_account_picker_catalog():
    provider = github_copilot_provider()
    provider_models = provider.get_models()
    picker_model_id = _require_model_id(provider_models, 0)
    disabled_model_id = _require_model_id(provider_models, 1)
    hidden_model_id = _require_model_id(provider_models, 2)

    credential = await refresh_github_copilot_models_for_test(
        [
            {"id": picker_model_id, "model_picker_enabled": True, "capabilities": {"supports": {"tool_calls": True}}},
            {
                "id": disabled_model_id,
                "model_picker_enabled": True,
                "policy": {"state": "disabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": hidden_model_id,
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
        ]
    )
    assert credential.extra["availableModelIds"] == [picker_model_id]

    store = InMemoryCredentialStore()
    await store.modify("github-copilot", lambda _current: _store(credential))
    models = create_models(credentials=store)
    models.set_provider(provider)
    available = await models.get_available("github-copilot")

    assert [model.id for model in available] == [picker_model_id]


@pytest.mark.tonio
async def test_falls_back_to_explicitly_enabled_policy_models_when_the_picker_catalog_is_empty():
    provider = github_copilot_provider()
    enabled_model_id = _require_model_id(provider.get_models(), 0)
    credential = await refresh_github_copilot_models_for_test(
        [
            {
                "id": enabled_model_id,
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "policy-disabled-model",
                "model_picker_enabled": False,
                "policy": {"state": "disabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "unconfigured-model",
                "model_picker_enabled": False,
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "tool-incapable-model",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": False}},
            },
        ]
    )

    assert credential.extra["availableModelIds"] == [enabled_model_id]

    store = InMemoryCredentialStore()
    await store.modify("github-copilot", lambda _current: _store(credential))
    models = create_models(credentials=store)
    models.set_provider(provider)
    available = await models.get_available("github-copilot")

    assert [model.id for model in available] == [enabled_model_id]


@pytest.mark.tonio
async def test_does_not_fall_back_to_policy_models_for_non_individual_accounts():
    credential = await refresh_github_copilot_models_for_test(
        [
            {
                "id": "gpt-4.1",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
        ],
        "proxy.business.githubcopilot.com",
    )

    assert credential.extra["availableModelIds"] == []


@pytest.mark.tonio
async def test_reports_device_code_details_through_the_device_code_event():
    def handler(request: OAuthRequest):
        if request.url.endswith("/login/device/code"):
            return json_response(device_code_body())
        if request.url.endswith("/login/oauth/access_token"):
            return json_response({"access_token": "ghu_refresh_token"})
        if "/copilot_internal/v2/token" in request.url:
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        if request.url.endswith("/models"):
            return json_response({"data": []})
        assert "/models/" in request.url and request.url.endswith("/policy")
        return text_response("", 200)

    interaction = RecordingInteraction(prompt=blank_enterprise_prompt)
    with virtual_clock(), stub_oauth_http(handler):
        await github_copilot_oauth.login(interaction)

    assert device_code_events(interaction) == [("ABCD-EFGH", "https://github.com/login/device", 1, 900)]


@pytest.mark.tonio
async def test_rejects_a_non_http_verification_uri_before_it_reaches_the_device_code_event():
    """A malicious enterprise OAuth server could return a verification_uri that the
    browser launcher would otherwise hand to the OS."""

    def handler(request: OAuthRequest):
        assert request.url.endswith("/login/device/code")
        return json_response(device_code_body(verification_uri="$(id>/tmp/pwned)"))

    interaction = RecordingInteraction(prompt=blank_enterprise_prompt)
    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match="Untrusted verification_uri"),
    ):
        await github_copilot_oauth.login(interaction)

    assert device_code_events(interaction) == []


@pytest.mark.tonio
async def test_normalizes_verification_uri_before_it_reaches_the_device_code_event():
    raw_verification_uri = "https://github.com/login/\x1b]8;;evil"
    normalized_verification_uri = "https://github.com/login/%1B]8;;evil"
    assert normalized_verification_uri != raw_verification_uri

    def handler(request: OAuthRequest):
        if request.url.endswith("/login/device/code"):
            return json_response(device_code_body(verification_uri=raw_verification_uri))
        if request.url.endswith("/login/oauth/access_token"):
            return json_response({"access_token": "ghu_refresh_token"})
        if "/copilot_internal/v2/token" in request.url:
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        if request.url.endswith("/models"):
            return json_response({"data": []})
        return text_response("", 200)

    interaction = RecordingInteraction(prompt=blank_enterprise_prompt)
    with virtual_clock(), stub_oauth_http(handler):
        await github_copilot_oauth.login(interaction)

    assert device_code_events(interaction) == [("ABCD-EFGH", normalized_verification_uri, 1, 900)]


@pytest.mark.tonio
async def test_waits_before_polling_and_increases_the_interval_after_slow_down():
    poll_times: list[int] = []
    access_token_responses = [
        json_response({"error": "authorization_pending", "error_description": "pending"}),
        json_response({"error": "slow_down", "error_description": "slow down", "interval": 7}),
        json_response({"access_token": "ghu_refresh_token"}),
    ]

    def handler(request: OAuthRequest):
        if request.url.endswith("/login/device/code"):
            assert request.method == "POST"
            assert request.headers["Accept"] == "application/json"
            assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
            assert request.form["client_id"]
            assert request.form["scope"] == "read:user"
            return json_response(device_code_body(interval=5))

        if request.url.endswith("/login/oauth/access_token"):
            poll_times.append(clock.now_ms())
            assert request.method == "POST"
            assert request.headers["Accept"] == "application/json"
            assert request.headers["Content-Type"] == "application/x-www-form-urlencoded"
            assert request.form["client_id"]
            assert request.form["device_code"] == "device-code"
            assert request.form["grant_type"] == "urn:ietf:params:oauth:grant-type:device_code"
            if not access_token_responses:
                raise AssertionError("Unexpected extra access token poll")
            return access_token_responses.pop(0)

        if "/copilot_internal/v2/token" in request.url:
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        if request.url.endswith("/models"):
            return json_response({"data": []})
        return text_response("", 200)

    with virtual_clock(), stub_oauth_http(handler):
        await github_copilot_oauth.login(RecordingInteraction(prompt=blank_enterprise_prompt))

    # 5 s wait before the first poll, 5 s interval, then the server's 7 s slow_down.
    assert poll_times == [
        DEFAULT_START_MS + 5000,
        DEFAULT_START_MS + 10_000,
        DEFAULT_START_MS + 17_000,
    ]


@pytest.mark.tonio
async def test_times_out_after_repeated_slow_down_responses():
    poll_times: list[int] = []
    access_token_responses = [
        json_response({"error": "slow_down", "error_description": "slow down"}),
        json_response({"error": "slow_down", "error_description": "still too fast"}),
        json_response({"error": "authorization_pending", "error_description": "pending"}),
    ]

    def handler(request: OAuthRequest):
        if request.url.endswith("/login/device/code"):
            return json_response(device_code_body(interval=5, expires_in=25))
        assert request.url.endswith("/login/oauth/access_token")
        poll_times.append(clock.now_ms())
        if not access_token_responses:
            raise AssertionError("Unexpected extra access token poll")
        return access_token_responses.pop(0)

    with (
        virtual_clock(),
        stub_oauth_http(handler),
        pytest.raises(RuntimeError, match="Device flow timed out after one or more slow_down responses"),
    ):
        await github_copilot_oauth.login(RecordingInteraction(prompt=blank_enterprise_prompt))

    assert poll_times == [DEFAULT_START_MS + 5000, DEFAULT_START_MS + 15_000]


def login_handler(models, policy=None):
    """pi's `stubGitHubCopilotLoginFetch`: a full device-flow login whose model
    catalog and policy responses come from the caller."""

    def handler(request: OAuthRequest):
        if request.url.endswith("/login/device/code"):
            return json_response(device_code_body())
        if request.url.endswith("/login/oauth/access_token"):
            return json_response({"access_token": "ghu_refresh_token"})
        if "/copilot_internal/v2/token" in request.url:
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        if request.url == COPILOT_MODELS_URL:
            return models()
        if request.url.startswith(f"{COPILOT_MODELS_URL}/") and request.url.endswith("/policy"):
            if policy is None:
                raise AssertionError(f"Unexpected policy request: {request.url}")
            return policy(request.url[len(COPILOT_MODELS_URL) + 1 : -len("/policy")])
        raise AssertionError(f"Unexpected request URL: {request.url}")

    return handler


def account_model(model_id: str, policy_state: str, *, tool_calls: bool = True) -> dict[str, Any]:
    return {
        "id": model_id,
        "model_picker_enabled": True,
        "policy": {"state": policy_state},
        "capabilities": {"supports": {"tool_calls": tool_calls}},
    }


def throttled_response(retry_after: str):
    return text_response('{"error": "too many requests"}', 429, {"retry-after": retry_after})


@pytest.mark.tonio
async def test_does_not_retry_model_catalog_throttling_during_credential_refresh():
    catalog_request_count = 0

    def handler(request: OAuthRequest):
        nonlocal catalog_request_count
        if "/copilot_internal/v2/token" in request.url:
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        assert request.url == COPILOT_MODELS_URL
        catalog_request_count += 1
        return throttled_response("0")

    with virtual_clock(), stub_oauth_http(handler), pytest.raises(RuntimeError, match="429"):
        await github_copilot_oauth.refresh(
            OAuthCredential(access="old-access-token", refresh="ghu_refresh_token", expires=0), None
        )

    assert catalog_request_count == 1


@pytest.mark.tonio
async def test_updates_only_known_tool_capable_unconfigured_account_model_policies():
    provider_models = github_copilot_provider().get_models()
    configured_model_id = _require_model_id(provider_models, 0)
    unconfigured_model_id = _require_model_id(provider_models, 1)
    tool_incapable_model_id = _require_model_id(provider_models, 2)
    catalog_request_count = 0
    policy_model_ids: list[str] = []

    def models():
        nonlocal catalog_request_count
        catalog_request_count += 1
        return json_response(
            {
                "data": [
                    account_model(configured_model_id, "enabled"),
                    account_model(unconfigured_model_id, "unconfigured"),
                    account_model("remote-only-model", "unconfigured"),
                    account_model(tool_incapable_model_id, "unconfigured", tool_calls=False),
                ]
            }
        )

    def policy(model_id: str):
        policy_model_ids.append(model_id)
        return text_response("", 200)

    with virtual_clock(), stub_oauth_http(login_handler(models, policy)):
        credential = await github_copilot_oauth.login(RecordingInteraction(prompt=blank_enterprise_prompt))

    assert catalog_request_count == 1
    assert policy_model_ids == [unconfigured_model_id]
    assert credential.extra["availableModelIds"] == [
        configured_model_id,
        unconfigured_model_id,
        "remote-only-model",
    ]


@pytest.mark.tonio
async def test_retries_a_throttled_policy_update_after_retry_after():
    model_id = _require_model_id(github_copilot_provider().get_models(), 0)
    policy_request_times: list[int] = []

    def models():
        return json_response({"data": [account_model(model_id, "unconfigured")]})

    def policy(_model_id: str):
        policy_request_times.append(clock.now_ms())
        return throttled_response("1") if len(policy_request_times) == 1 else text_response("", 200)

    with virtual_clock(), stub_oauth_http(login_handler(models, policy)):
        await github_copilot_oauth.login(RecordingInteraction(prompt=blank_enterprise_prompt))

    assert len(policy_request_times) == 2
    # The retry honored the server's Retry-After of one second.
    assert policy_request_times[1] - policy_request_times[0] == 1000


@pytest.mark.tonio
async def test_continues_policy_updates_after_a_transport_failure():
    provider_models = github_copilot_provider().get_models()
    model_ids = [_require_model_id(provider_models, 0), _require_model_id(provider_models, 1)]
    policy_model_ids: list[str] = []

    def models():
        return json_response({"data": [account_model(model_id, "unconfigured") for model_id in model_ids]})

    def policy(model_id: str):
        policy_model_ids.append(model_id)
        if len(policy_model_ids) == 1:
            raise RuntimeError("fetch failed")
        return text_response("", 200)

    with virtual_clock(), stub_oauth_http(login_handler(models, policy)):
        await github_copilot_oauth.login(RecordingInteraction(prompt=blank_enterprise_prompt))

    assert policy_model_ids == model_ids


@pytest.mark.tonio
async def test_stops_policy_updates_and_persists_authentication_when_the_retry_delay_exceeds_the_login_budget():
    provider_models = github_copilot_provider().get_models()
    first_model_id = _require_model_id(provider_models, 0)
    second_model_id = _require_model_id(provider_models, 1)
    policy_model_ids: list[str] = []

    def models():
        return json_response(
            {"data": [account_model(model_id, "unconfigured") for model_id in (first_model_id, second_model_id)]}
        )

    def policy(model_id: str):
        policy_model_ids.append(model_id)
        return throttled_response("5")

    store = InMemoryCredentialStore()
    models_runtime = create_models(credentials=store)
    models_runtime.set_provider(github_copilot_provider())

    with virtual_clock(), stub_oauth_http(login_handler(models, policy)):
        credential = await models_runtime.login(
            "github-copilot", "oauth", RecordingInteraction(prompt=blank_enterprise_prompt)
        )

    assert credential.access == COPILOT_TOKEN
    assert policy_model_ids == [first_model_id]
    assert await store.read("github-copilot") == credential
