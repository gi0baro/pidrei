"""Mirror of pi's cloudflare-stream.test.ts, plus pidrei-only auth cases.

pi has no spec for `cloudflare-auth.ts`; the gateway's header suppression
(`Authorization: null`, `x-api-key: null`) is the kind of thing that fails
silently — the upstream provider's own auth header would ride along and the
gateway would reject or double-authenticate — so it is pinned here.
"""

import pytest

from pidrei_ai.providers.cloudflare_auth import cloudflare_ai_gateway_auth, cloudflare_workers_ai_auth
from pidrei_ai.providers.cloudflare_stream import cloudflare_streams
from pidrei_ai.types import Context, Model, ModelCost, StreamOptions
from pidrei_ai.utils.cancel import CancelToken


_never_aborted_cancel = CancelToken()
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


MODEL = Model(
    id="model",
    name="model",
    api="openai-completions",
    provider="cloudflare-ai-gateway",
    base_url="https://gateway.ai.cloudflare.com/v1/{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/openai",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=1000,
    max_tokens=100,
)

CONTEXT = Context(messages=[])


class _Capturing:
    def __init__(self, captured: list[str]):
        self._captured = captured

    def stream(self, request_model, _context, _options=None):
        self._captured.append(request_model.base_url)
        return AssistantMessageEventStream()

    def stream_simple(self, request_model, _context, _options=None):
        self._captured.append(request_model.base_url)
        return AssistantMessageEventStream()


def test_materializes_the_model_endpoint_before_dispatch():
    captured: list[str] = []
    streams = cloudflare_streams(_Capturing(captured))
    options = StreamOptions(env={"CLOUDFLARE_ACCOUNT_ID": "account", "CLOUDFLARE_GATEWAY_ID": "gateway"})

    streams.stream(MODEL, CONTEXT, options)
    streams.stream_simple(MODEL, CONTEXT, options)

    assert captured == [
        "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
        "https://gateway.ai.cloudflare.com/v1/account/gateway/openai",
    ]


def test_keeps_placeholders_when_the_provider_env_does_not_resolve_them():
    captured: list[str] = []
    streams = cloudflare_streams(_Capturing(captured))

    streams.stream_simple(MODEL, CONTEXT, StreamOptions())

    assert captured == [MODEL.base_url]


# --- pidrei-only: auth resolution ---------------------------------------------


class _Ctx:
    def __init__(self, env: dict[str, str]):
        self._env = env

    async def env(self, name: str) -> str | None:
        return self._env.get(name)

    async def file_exists(self, _path: str) -> bool:
        return False


@pytest.mark.tonio
async def test_workers_ai_resolves_the_key_and_account_from_the_environment():
    auth = cloudflare_workers_ai_auth()

    result = await auth.resolve(
        _Ctx({"CLOUDFLARE_API_KEY": "k", "CLOUDFLARE_ACCOUNT_ID": "acct"}), None, _never_aborted_cancel
    )

    assert result.auth.api_key == "k"
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct"}
    assert result.source == "CLOUDFLARE_API_KEY"


@pytest.mark.tonio
async def test_workers_ai_is_unconfigured_without_an_account_id():
    auth = cloudflare_workers_ai_auth()

    assert await auth.resolve(_Ctx({"CLOUDFLARE_API_KEY": "k"}), None, _never_aborted_cancel) is None


@pytest.mark.tonio
async def test_the_gateway_suppresses_the_upstream_providers_auth_headers():
    auth = cloudflare_ai_gateway_auth()

    result = await auth.resolve(
        _Ctx(
            {
                "CLOUDFLARE_API_KEY": "k",
                "CLOUDFLARE_ACCOUNT_ID": "acct",
                "CLOUDFLARE_GATEWAY_ID": "gw",
            }
        ),
        None,
        _never_aborted_cancel,
    )

    assert result.auth.api_key is None
    assert result.auth.headers == {
        "cf-aig-authorization": "Bearer k",
        "Authorization": None,
        "x-api-key": None,
    }
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_GATEWAY_ID": "gw"}


@pytest.mark.tonio
async def test_the_gateway_is_unconfigured_without_a_gateway_id():
    auth = cloudflare_ai_gateway_auth()

    assert (
        await auth.resolve(
            _Ctx({"CLOUDFLARE_API_KEY": "k", "CLOUDFLARE_ACCOUNT_ID": "acct"}), None, _never_aborted_cancel
        )
        is None
    )


@pytest.mark.tonio
async def test_a_credential_carrying_only_the_key_still_picks_up_ambient_ids():
    from pidrei_ai.auth.types import ApiKeyCredential

    auth = cloudflare_ai_gateway_auth()

    result = await auth.resolve(
        _Ctx({"CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_GATEWAY_ID": "gw"}),
        ApiKeyCredential(key="stored-key"),
        _never_aborted_cancel,
    )

    assert result.auth.headers["cf-aig-authorization"] == "Bearer stored-key"
    assert result.env == {"CLOUDFLARE_ACCOUNT_ID": "acct", "CLOUDFLARE_GATEWAY_ID": "gw"}
    assert result.source == "stored credential"
