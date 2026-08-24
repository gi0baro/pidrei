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


@pytest.mark.tonio
async def test_filters_models_to_the_authenticated_account_picker_catalog():
    credential = await refresh_github_copilot_models_for_test(
        [
            {"id": "gpt-4.1", "model_picker_enabled": True, "capabilities": {"supports": {"tool_calls": True}}},
            {
                "id": "claude-opus-4.7",
                "model_picker_enabled": True,
                "policy": {"state": "disabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "gpt-5.4-nano",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
        ]
    )
    assert credential.extra["availableModelIds"] == ["gpt-4.1"]

    store = InMemoryCredentialStore()
    await store.modify("github-copilot", lambda _current: _store(credential))
    models = create_models(credentials=store)
    models.set_provider(github_copilot_provider())
    available = await models.get_available("github-copilot")

    assert [model.id for model in available] == ["gpt-4.1"]


@pytest.mark.tonio
async def test_falls_back_to_explicitly_enabled_policy_models_when_the_picker_catalog_is_empty():
    credential = await refresh_github_copilot_models_for_test(
        [
            {
                "id": "gpt-4.1",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "claude-opus-4.7",
                "model_picker_enabled": False,
                "policy": {"state": "disabled"},
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "gpt-5.4-nano",
                "model_picker_enabled": False,
                "capabilities": {"supports": {"tool_calls": True}},
            },
            {
                "id": "gpt-4o",
                "model_picker_enabled": False,
                "policy": {"state": "enabled"},
                "capabilities": {"supports": {"tool_calls": False}},
            },
        ]
    )

    assert credential.extra["availableModelIds"] == ["gpt-4.1"]

    store = InMemoryCredentialStore()
    await store.modify("github-copilot", lambda _current: _store(credential))
    models = create_models(credentials=store)
    models.set_provider(github_copilot_provider())
    available = await models.get_available("github-copilot")

    assert [model.id for model in available] == ["gpt-4.1"]


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


@pytest.mark.tonio
async def test_enables_model_policies_sequentially_during_login():
    import threading

    import tonio.colored as tonio

    counter_guard = threading.Lock()
    active_policy_requests = 0
    max_active_policy_requests = 0
    policy_request_count = 0

    async def policy_delay_response():
        nonlocal active_policy_requests, max_active_policy_requests, policy_request_count
        with counter_guard:
            policy_request_count += 1
            active_policy_requests += 1
            max_active_policy_requests = max(max_active_policy_requests, active_policy_requests)
        # pi delays each policy request by 10ms of fake time; a small real
        # sleep keeps any overlap observable had the updates stayed batched.
        await tonio.time.sleep(0.01)
        with counter_guard:
            active_policy_requests -= 1
        return text_response("", 200)

    def handler(request: OAuthRequest):
        if request.url.endswith("/login/device/code"):
            return json_response(device_code_body())
        if request.url.endswith("/login/oauth/access_token"):
            return json_response({"access_token": "ghu_refresh_token"})
        if "/copilot_internal/v2/token" in request.url:
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        if request.url.endswith("/models"):
            return json_response({"data": []})
        if "/models/" in request.url and request.url.endswith("/policy"):
            return policy_delay_response()
        raise AssertionError(f"Unexpected request URL: {request.url}")

    with virtual_clock(), stub_oauth_http(handler):
        await github_copilot_oauth.login(RecordingInteraction(prompt=blank_enterprise_prompt))

    assert policy_request_count > 1
    assert max_active_policy_requests == 1


@pytest.mark.tonio
async def test_retries_get_models_once_after_a_429_honoring_retry_after():
    models_request_count = 0
    retry_wait_ms: list[float] = []

    def handler(request: OAuthRequest):
        nonlocal models_request_count
        if request.url.endswith("/login/device/code"):
            return json_response(device_code_body())
        if request.url.endswith("/login/oauth/access_token"):
            return json_response({"access_token": "ghu_refresh_token"})
        if "/copilot_internal/v2/token" in request.url:
            return json_response({"token": COPILOT_TOKEN, "expires_at": 9999999999})
        if request.url.endswith("/models"):
            models_request_count += 1
            retry_wait_ms.append(clock.now_ms())
            if models_request_count == 1:
                return text_response("too many requests", 429, {"retry-after": "1"})
            return json_response({"data": [{"id": "gpt-5.4", "model_picker_enabled": True}]})
        if "/models/" in request.url and request.url.endswith("/policy"):
            return text_response("", 200)
        raise AssertionError(f"Unexpected request URL: {request.url}")

    interaction = RecordingInteraction(prompt=blank_enterprise_prompt)
    with virtual_clock(), stub_oauth_http(handler):
        credential = await github_copilot_oauth.login(interaction)

    assert models_request_count == 2
    assert credential.extra["availableModelIds"] == ["gpt-5.4"]
    # The retry honored the server's Retry-After of one second.
    assert retry_wait_ms[1] - retry_wait_ms[0] == 1000
