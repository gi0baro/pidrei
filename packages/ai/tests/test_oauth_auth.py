"""Mirror of pi's oauth-auth.test.ts.

pi's first case asserts its type-only `src/oauth.ts` barrel exports no flow
implementations — a bundler guard. There is no barrel here, so the case is ported
as the invariant it actually protects: constructing every built-in provider must
not pull a flow module in. It runs in a subprocess because the rest of this suite
imports those modules directly.
"""

import subprocess
import sys
import time

import pytest

from pidrei_ai.auth.credential_store import InMemoryCredentialStore
from pidrei_ai.auth.oauth.anthropic import anthropic_oauth
from pidrei_ai.auth.oauth.github_copilot import github_copilot_oauth
from pidrei_ai.auth.oauth.openai_codex import openai_codex_oauth
from pidrei_ai.auth.oauth.openrouter import openrouter_oauth
from pidrei_ai.auth.oauth.xai import xai_oauth
from pidrei_ai.auth.types import OAuthCredential
from pidrei_ai.providers.anthropic import anthropic_provider
from pidrei_ai.providers.github_copilot import github_copilot_provider
from pidrei_ai.registry import create_models
from pidrei_ai.utils import clock

from .oauth_helpers import OAuthRequest, json_response, stub_oauth_http, virtual_clock


MAX_SAFE_INTEGER = 9007199254740991

_LAZY_CHAIN_PROBE = """
import sys

from pidrei_ai.providers.all import builtin_providers

providers = builtin_providers()
assert providers, "expected built-in providers"
assert any(provider.auth.oauth is not None for provider in providers), "expected an OAuth provider"

leaked = sorted(name for name in sys.modules if name.startswith("pidrei_ai.auth.oauth."))
allowed = {"pidrei_ai.auth.oauth.load"}
print(",".join(name for name in leaked if name not in allowed))
"""


def test_constructing_the_builtin_providers_does_not_import_a_flow_module():
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", _LAZY_CHAIN_PROBE],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "", f"provider construction imported: {result.stdout.strip()}"


async def _stored(credential: OAuthCredential) -> OAuthCredential:
    return credential


@pytest.mark.tonio
async def test_anthropic_to_auth_derives_the_api_key_from_the_access_token():
    auth = await anthropic_oauth.to_auth(OAuthCredential(access="token", refresh="r", expires=0))
    assert auth.api_key == "token"
    assert auth.headers is None and auth.base_url is None


@pytest.mark.tonio
async def test_openai_codex_to_auth_derives_the_api_key_from_the_access_token():
    auth = await openai_codex_oauth.to_auth(OAuthCredential(access="token", refresh="r", expires=0))
    assert auth.api_key == "token"


@pytest.mark.tonio
async def test_openrouter_derives_the_api_key_and_keeps_the_permanent_credential_on_refresh():
    credential = OAuthCredential(access="token", refresh="", expires=MAX_SAFE_INTEGER)
    assert (await openrouter_oauth.to_auth(credential)).api_key == "token"
    assert await openrouter_oauth.refresh(credential, None) is credential


@pytest.mark.tonio
async def test_xai_to_auth_derives_the_api_key_from_the_access_token():
    auth = await xai_oauth.to_auth(OAuthCredential(access="token", refresh="r", expires=0))
    assert auth.api_key == "token"


@pytest.mark.tonio
async def test_github_copilot_to_auth_derives_base_url_from_the_token_proxy_endpoint():
    access = "tid=abc;exp=123;proxy-ep=proxy.enterprise.example;rest"
    auth = await github_copilot_oauth.to_auth(OAuthCredential(access=access, refresh="r", expires=0))
    assert auth.api_key == access
    assert auth.base_url == "https://api.enterprise.example"


@pytest.mark.tonio
async def test_github_copilot_to_auth_falls_back_to_the_enterprise_domain_then_the_individual_endpoint():
    enterprise = await github_copilot_oauth.to_auth(
        OAuthCredential(
            access="no-proxy-ep",
            refresh="r",
            expires=0,
            extra={"enterpriseUrl": "https://company.ghe.com"},
        )
    )
    assert enterprise.base_url == "https://copilot-api.company.ghe.com"

    individual = await github_copilot_oauth.to_auth(OAuthCredential(access="no-proxy-ep", refresh="r", expires=0))
    assert individual.base_url == "https://api.individual.githubcopilot.com"


@pytest.mark.tonio
async def test_anthropic_refresh_exchanges_the_refresh_token_and_returns_a_typed_credential():
    def handler(_request: OAuthRequest):
        return json_response({"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    with virtual_clock(), stub_oauth_http(handler):
        refreshed = await anthropic_oauth.refresh(OAuthCredential(access="old", refresh="old-r", expires=0), None)
        assert refreshed.expires > clock.now_ms()

    assert refreshed.type == "oauth"
    assert refreshed.access == "new-access"
    assert refreshed.refresh == "new-refresh"


@pytest.mark.tonio
async def test_github_copilot_refresh_preserves_the_enterprise_domain():
    fetched_urls: list[str] = []

    def handler(request: OAuthRequest):
        fetched_urls.append(request.url)
        if request.url.endswith("/models"):
            return json_response({"data": []})
        return json_response({"token": "new-token", "expires_at": 9999999999})

    with virtual_clock(), stub_oauth_http(handler):
        refreshed = await github_copilot_oauth.refresh(
            OAuthCredential(access="old", refresh="gh-token", expires=0, extra={"enterpriseUrl": "company.ghe.com"}),
            None,
        )

    assert refreshed.access == "new-token"
    assert refreshed.extra["enterpriseUrl"] == "company.ghe.com"
    assert "api.company.ghe.com" in fetched_urls[0]


# --- OAuth through Models.get_auth (lazy load chain) --------------------------


@pytest.mark.tonio
async def test_resolves_stored_anthropic_oauth_credentials_via_the_lazy_flow_import():
    """No virtual clock: `auth/resolve.py` reads the real clock to decide whether a
    stored credential still needs refreshing, exactly as pi does."""
    credentials = InMemoryCredentialStore()
    await credentials.modify(
        "anthropic",
        lambda _current: _stored(
            OAuthCredential(access="oauth-access-token", refresh="r", expires=int(time.time() * 1000) + 60_000)
        ),
    )
    models = create_models(credentials=credentials)
    models.set_provider(anthropic_provider())

    model = models.get_models("anthropic")[0]
    result = await models.get_auth(model.provider)

    assert result is not None
    assert result.auth.api_key == "oauth-access-token"
    assert result.source == "OAuth"


@pytest.mark.tonio
async def test_resolves_stored_github_copilot_oauth_credentials_including_per_credential_base_url():
    access = "tid=abc;exp=123;proxy-ep=proxy.business.githubcopilot.com;rest"
    credentials = InMemoryCredentialStore()
    await credentials.modify(
        "github-copilot",
        lambda _current: _stored(OAuthCredential(access=access, refresh="r", expires=int(time.time() * 1000) + 60_000)),
    )
    models = create_models(credentials=credentials)
    models.set_provider(github_copilot_provider())

    model = models.get_models("github-copilot")[0]
    result = await models.get_auth(model.provider)

    assert result is not None
    assert result.auth.api_key == access
    assert result.auth.base_url == "https://api.business.githubcopilot.com"
