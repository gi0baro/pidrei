"""Mirror of pi's identity.test.ts.

The expected values are pidrei's (`PIDREI_ATTRIBUTION`, the `pidrei` User-Agent,
NVIDIA's `PiDrei` billing origin); pi's `acme` cases keep pi's upper-case-first
rule. pi reads the version constant against package.json; here it is checked
against pidrei-ai's pyproject.
"""

import json
import platform
import sys
import tomllib
from pathlib import Path

import pytest

from pidrei_ai.api import openai_completions
from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.registry import create_models, create_provider
from pidrei_ai.types import (
    ClientIdentity,
    Context,
    Model,
    ModelCost,
    SimpleStreamOptions,
    StreamOptions,
    UserMessage,
)
from pidrei_ai.utils.identity import (
    PIDREI_ATTRIBUTION,
    format_user_agent,
    get_attribution_headers,
    get_identity_headers,
    merge_client_identity,
    with_client_identity,
)
from pidrei_ai.utils.transcript import normalize_context
from pidrei_ai.utils.version import PIDREI_AI_VERSION
from pidrei_http import http


ACME = {"name": "acme", "url": "https://acme.example"}

RUNTIME = f"{sys.platform}; python/{platform.python_version()}; {platform.machine()}"


class Target:
    def __init__(self, provider: str, base_url: str):
        self.provider = provider
        self.base_url = base_url


def completions_model(provider: str, base_url: str, headers: dict[str, str] | None = None) -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="openai-completions",
        provider=provider,
        base_url=base_url,
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=4096,
        headers=headers,
    )


class CapturingFetch:
    """Records the request headers and answers with a one-chunk completion."""

    def __init__(self):
        self.captured: http.Request | None = None

    async def __call__(self, request, *, env=None):
        self.captured = request
        chunks = [
            {"id": "chatcmpl-identity", "choices": [{"delta": {"content": "ok"}, "finish_reason": None, "index": 0}]},
            {
                "id": "chatcmpl-identity",
                "choices": [{"delta": {}, "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        ]
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
        return http.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode(), request=request)

    def headers(self) -> dict[str, str]:
        assert self.captured is not None, "no request was sent"
        return {name.lower(): value for name, value in self.captured.headers.items()}


CONTEXT = normalize_context(Context(messages=[UserMessage(content="hello", timestamp=1)]))


def test_keeps_the_version_constant_in_sync_with_the_pyproject():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert PIDREI_AI_VERSION == tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["version"]


def test_formats_user_agents():
    assert format_user_agent() == f"pidrei/{PIDREI_AI_VERSION} ({RUNTIME})"
    assert format_user_agent({"version": "9.9.9"}) == f"pidrei/9.9.9 ({RUNTIME})"
    assert format_user_agent({"user_agent": "short", "version": "9.9.9"}) == "pidrei"
    assert format_user_agent({"user_agent": {"custom": "acme/1.0"}}) == "acme/1.0"


def test_merges_identities_field_by_field():
    assert merge_client_identity(None, None) is None
    assert merge_client_identity({"attribution": PIDREI_ATTRIBUTION, "user_agent": "short"}, {"version": "1.0.0"}) == {
        "attribution": PIDREI_ATTRIBUTION,
        "user_agent": "short",
        "version": "1.0.0",
    }
    assert merge_client_identity({"attribution": PIDREI_ATTRIBUTION}, {"attribution": None}) == {"attribution": None}


def test_sends_no_attribution_unless_enabled():
    open_router = Target("openrouter", "https://openrouter.ai/api/v1")
    assert get_attribution_headers(open_router) == {}
    assert get_attribution_headers(open_router, {"attribution": None}) == {}
    assert get_identity_headers(open_router) == {"User-Agent": format_user_agent()}


def test_attributes_openrouter_requests():
    open_router = Target("openrouter", "https://openrouter.ai/api/v1")
    assert get_attribution_headers(open_router, {"attribution": PIDREI_ATTRIBUTION}) == {
        "HTTP-Referer": "https://github.com/gi0baro/pidrei",
        "X-OpenRouter-Title": "pidrei",
        "X-OpenRouter-Categories": "cli-agent",
    }
    assert get_attribution_headers(
        open_router, {"attribution": {**ACME, "categories": ["cli-agent", "cloud-agent"]}}
    ) == {
        "HTTP-Referer": "https://acme.example",
        "X-OpenRouter-Title": "acme",
        "X-OpenRouter-Categories": "cli-agent,cloud-agent",
    }
    assert get_attribution_headers(open_router, {"attribution": {**ACME, "categories": []}}) == {
        "HTTP-Referer": "https://acme.example",
        "X-OpenRouter-Title": "acme",
    }
    # Custom providers routed through OpenRouter, including legacy base URLs that are not valid URLs.
    headers = get_attribution_headers(Target("custom", "not-a-url-openrouter.ai"), {"attribution": PIDREI_ATTRIBUTION})
    assert headers["X-OpenRouter-Title"] == "pidrei"


def test_attributes_nvidia_nim_requests_with_a_capitalized_name_and_opencode_requests_with_the_name():
    assert get_attribution_headers(
        Target("nvidia", "https://example.test/v1"), {"attribution": PIDREI_ATTRIBUTION}
    ) == {"X-BILLING-INVOKE-ORIGIN": "PiDrei"}
    assert get_attribution_headers(
        Target("custom-nim", "https://integrate.api.nvidia.com/v1"), {"attribution": ACME}
    ) == {"X-BILLING-INVOKE-ORIGIN": "Acme"}
    assert get_attribution_headers(
        Target("opencode", "https://opencode.ai/zen/v1"), {"attribution": PIDREI_ATTRIBUTION}
    ) == {"x-opencode-client": "pidrei"}
    assert get_attribution_headers(Target("custom", "https://opencode.ai/zen/v1"), {"attribution": ACME}) == {
        "x-opencode-client": "acme"
    }


def test_does_not_attribute_nvidia_models_routed_through_gateways_to_nvidia():
    assert "X-BILLING-INVOKE-ORIGIN" not in get_attribution_headers(
        Target("openrouter", "https://openrouter.ai/api/v1"), {"attribution": PIDREI_ATTRIBUTION}
    )
    assert (
        get_attribution_headers(
            Target("vercel-ai-gateway", "https://ai-gateway.vercel.sh/v1"), {"attribution": PIDREI_ATTRIBUTION}
        )
        == {}
    )


# --- client identity in requests ----------------------------------------------


@pytest.mark.tonio
async def test_applies_the_identity_below_model_and_request_headers():
    capture = CapturingFetch()
    model = completions_model("openrouter", "https://openrouter.ai/api/v1", {"X-OpenRouter-Title": "model-title"})
    await openai_completions.stream(
        model,
        CONTEXT,
        StreamOptions(
            api_key="key",
            fetch=capture,
            identity={"attribution": PIDREI_ATTRIBUTION, "user_agent": "short"},
            headers={"HTTP-Referer": None},
        ),
    ).result()

    headers = capture.headers()
    assert headers["user-agent"] == "pidrei"
    assert headers["x-openrouter-title"] == "model-title"
    assert "http-referer" not in headers


@pytest.mark.tonio
async def test_keeps_impersonating_model_headers_over_the_identity_user_agent():
    capture = CapturingFetch()
    model = completions_model(
        "github-copilot", "https://api.githubcopilot.com", {"User-Agent": "GitHubCopilotChat/0.35.0"}
    )
    await openai_completions.stream(
        model,
        CONTEXT,
        StreamOptions(api_key="key", fetch=capture, identity={"user_agent": {"custom": "acme"}}),
    ).result()

    assert capture.headers()["user-agent"] == "GitHubCopilotChat/0.35.0"


def _api_key_auth() -> ProviderAuth:
    async def resolve(_ctx, _credential, _cancel):
        return AuthResult(auth=ModelAuth(api_key="key"))

    return ProviderAuth(api_key=ApiKeyAuth(name="Test", resolve=resolve))


def _completions_provider(id: str, model: Model):
    return create_provider(id=id, auth=_api_key_auth(), models=[model], api=openai_completions)


@pytest.mark.tonio
async def test_merges_request_identity_over_the_models_and_wrapper_identities():
    model = completions_model("openrouter", "https://openrouter.ai/api/v1")

    def attribution(name: str):
        return {"name": name, "url": "https://acme.example"}

    models = create_models(identity={"attribution": attribution("models"), "user_agent": "short"})
    models.set_provider(_completions_provider("openrouter", model))

    async def send(target, identity: ClientIdentity | None = None) -> dict[str, str]:
        capture = CapturingFetch()
        await target.stream_simple(model, CONTEXT, SimpleStreamOptions(fetch=capture, identity=identity)).result()
        return capture.headers()

    from_models = await send(models)
    assert from_models["user-agent"] == "pidrei"
    assert from_models["x-openrouter-title"] == "models"

    wrapped = with_client_identity(models, {"attribution": attribution("wrapper")})
    from_wrapper = await send(wrapped)
    assert from_wrapper["user-agent"] == "pidrei"
    assert from_wrapper["x-openrouter-title"] == "wrapper"
    assert wrapped.get_model("openrouter", "test-model") is model
    # A mutable Models stays mutable.
    wrapped.delete_provider("openrouter")
    assert models.get_provider("openrouter") is None
    models.set_provider(_completions_provider("openrouter", model))

    from_request = await send(wrapped, {"attribution": None})
    assert "x-openrouter-title" not in from_request
    assert from_request["user-agent"] == "pidrei"


@pytest.mark.tonio
async def test_sends_the_opencode_session_header_to_custom_providers_pointing_at_opencode():
    model = completions_model("custom-opencode", "https://opencode.ai/zen/v1")
    models = create_models()
    models.set_provider(_completions_provider("custom-opencode", model))

    async def send(headers: dict[str, str] | None = None) -> dict[str, str]:
        capture = CapturingFetch()
        await models.stream_simple(
            model, CONTEXT, SimpleStreamOptions(fetch=capture, session_id="session-1", headers=headers)
        ).result()
        return capture.headers()

    assert (await send())["x-opencode-session"] == "session-1"
    assert (await send({"X-OpenCode-Session": "configured"}))["x-opencode-session"] == "configured"
