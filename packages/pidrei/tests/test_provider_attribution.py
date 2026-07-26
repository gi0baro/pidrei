"""Provider attribution headers.

Adapted from pi's sdk-openrouter-attribution.test.ts, which was never mirrored.
Two deliberate differences:

- pi drives every case through `createAgentSession` and reads the headers off a
  captured request. All the logic lives in `merge_provider_attribution_headers`
  and the session only plumbs it, so these exercise the function directly.
- The expected values are pidrei's, not pi's (Phase 7 step 1). The cases that
  pin *which* identity is sent are the point of the suite: a value silently
  reverting to "pi" or "https://pi.dev" is the regression worth catching.
"""

import os
from types import SimpleNamespace

import pytest

from pidrei.core.provider_attribution import (
    ATTRIBUTION_ENV,
    ATTRIBUTION_NAME,
    ATTRIBUTION_URL,
    merge_provider_attribution_headers,
)


def model(provider: str, base_url: str = "https://api.example.com/v1"):
    return SimpleNamespace(provider=provider, base_url=base_url)


def settings(enabled: bool = True):
    return SimpleNamespace(get_enable_provider_attribution=lambda: enabled)


@pytest.fixture(autouse=True)
def _clear_env_override(request):
    """The env override wins over the setting, so it must not leak in."""
    previous = os.environ.pop(ATTRIBUTION_ENV, None)
    if previous is not None:
        request.addfinalizer(lambda: os.environ.__setitem__(ATTRIBUTION_ENV, previous))


class TestIdentity:
    def test_attribution_values_are_pidrei_not_pi(self):
        headers = merge_provider_attribution_headers(model("openrouter"), settings(), None)

        assert headers == {
            "HTTP-Referer": ATTRIBUTION_URL,
            "X-OpenRouter-Title": ATTRIBUTION_NAME,
            "X-OpenRouter-Categories": "cli-agent",
        }
        # The whole point of step 1: nothing here credits pi.
        assert "pi.dev" not in str(headers)
        assert ATTRIBUTION_URL == "https://github.com/gi0baro/pidrei"
        assert ATTRIBUTION_NAME == "pidrei"


class TestOpenRouter:
    def test_adds_default_attribution_headers_for_openrouter_models(self):
        headers = merge_provider_attribution_headers(model("openrouter"), settings(), None)
        assert headers["HTTP-Referer"] == ATTRIBUTION_URL

    def test_adds_headers_for_custom_providers_routed_through_openrouter(self):
        headers = merge_provider_attribution_headers(
            model("my-proxy", "https://openrouter.ai/api/v1"), settings(), None
        )
        assert headers["X-OpenRouter-Title"] == ATTRIBUTION_NAME

    def test_preserves_legacy_base_url_substring_matching(self):
        # pi matches OpenRouter by substring, not by parsed host, so a gateway
        # that merely embeds the name still gets attributed.
        headers = merge_provider_attribution_headers(
            model("gateway", "https://proxy.internal/openrouter.ai/v1"), settings(), None
        )
        assert headers["X-OpenRouter-Categories"] == "cli-agent"

    def test_does_not_add_headers_when_disabled(self):
        assert merge_provider_attribution_headers(model("openrouter"), settings(False), None) is None

    def test_env_override_disables_attribution(self):
        os.environ[ATTRIBUTION_ENV] = "0"
        assert merge_provider_attribution_headers(model("openrouter"), settings(True), None) is None

    def test_env_override_enables_attribution(self):
        os.environ[ATTRIBUTION_ENV] = "1"
        headers = merge_provider_attribution_headers(model("openrouter"), settings(False), None)
        assert headers["HTTP-Referer"] == ATTRIBUTION_URL

    def test_lets_provider_and_request_headers_override_the_defaults(self):
        headers = merge_provider_attribution_headers(
            model("openrouter"),
            settings(),
            None,
            {"HTTP-Referer": "https://provider.example"},
            {"X-OpenRouter-Title": "request-wins"},
        )
        assert headers["HTTP-Referer"] == "https://provider.example"
        assert headers["X-OpenRouter-Title"] == "request-wins"
        assert headers["X-OpenRouter-Categories"] == "cli-agent"


class TestNvidiaNim:
    def test_adds_headers_for_direct_nvidia_nim_endpoints(self):
        headers = merge_provider_attribution_headers(
            model("custom", "https://integrate.api.nvidia.com/v1"), settings(), None
        )
        assert headers == {"X-BILLING-INVOKE-ORIGIN": ATTRIBUTION_NAME}

    def test_adds_headers_for_the_nvidia_provider(self):
        headers = merge_provider_attribution_headers(model("nvidia"), settings(), None)
        assert headers == {"X-BILLING-INVOKE-ORIGIN": ATTRIBUTION_NAME}

    def test_does_not_add_headers_when_disabled(self):
        model_ = model("nvidia")
        assert merge_provider_attribution_headers(model_, settings(False), None) is None

    def test_lets_headers_override_the_defaults(self):
        headers = merge_provider_attribution_headers(
            model("nvidia"), settings(), None, {"X-BILLING-INVOKE-ORIGIN": "custom"}
        )
        assert headers == {"X-BILLING-INVOKE-ORIGIN": "custom"}

    def test_openrouter_wins_for_nvidia_models_routed_through_openrouter(self):
        headers = merge_provider_attribution_headers(model("nvidia", "https://openrouter.ai/api/v1"), settings(), None)
        assert "X-BILLING-INVOKE-ORIGIN" not in headers
        assert headers["HTTP-Referer"] == ATTRIBUTION_URL

    def test_no_nvidia_headers_when_routed_through_another_gateway(self):
        headers = merge_provider_attribution_headers(
            model("vercel-ai-gateway", "https://ai-gateway.vercel.sh/v1"), settings(), None
        )
        assert headers is None


class TestCloudflare:
    @pytest.mark.parametrize(
        "provider,base_url",
        [
            ("cloudflare-workers-ai", "https://api.example.com/v1"),
            ("cloudflare-ai-gateway", "https://api.example.com/v1"),
            ("custom", "https://api.cloudflare.com/client/v4"),
            ("custom", "https://gateway.ai.cloudflare.com/v1"),
        ],
    )
    def test_sends_a_pidrei_user_agent(self, provider, base_url):
        headers = merge_provider_attribution_headers(model(provider, base_url), settings(), None)
        assert headers == {"User-Agent": "pidrei-coding-agent"}


class TestOpenCodeSession:
    def test_adds_session_headers(self):
        headers = merge_provider_attribution_headers(model("opencode"), settings(), "sess-1")
        assert headers == {"x-opencode-session": "sess-1", "x-opencode-client": ATTRIBUTION_NAME}

    def test_matches_the_opencode_host(self):
        headers = merge_provider_attribution_headers(model("custom", "https://opencode.ai/v1"), settings(), "sess-2")
        assert headers["x-opencode-session"] == "sess-2"

    def test_session_headers_are_not_gated_by_the_attribution_toggle(self):
        # pi gates only the *default* attribution headers; the session id is
        # routing information, not attribution.
        headers = merge_provider_attribution_headers(model("opencode"), settings(False), "sess-3")
        assert headers == {"x-opencode-session": "sess-3", "x-opencode-client": ATTRIBUTION_NAME}

    def test_no_session_headers_without_a_session_id(self):
        assert merge_provider_attribution_headers(model("opencode"), settings(False), None) is None

    def test_lets_configured_headers_override_the_defaults(self):
        headers = merge_provider_attribution_headers(
            model("opencode"), settings(), "sess-4", {"x-opencode-client": "custom"}
        )
        assert headers["x-opencode-client"] == "custom"
        assert headers["x-opencode-session"] == "sess-4"


def test_unattributed_providers_get_nothing():
    assert merge_provider_attribution_headers(model("anthropic"), settings(), None) is None
