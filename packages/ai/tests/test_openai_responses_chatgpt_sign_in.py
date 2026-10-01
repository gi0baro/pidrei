"""Mirror of pi's openai-responses-chatgpt-sign-in.test.ts.

pi captures the payload through `onPayload` with a `fetch` that fails; the
payload is `build_params`' output, so the mirror reads it there.
"""

from dataclasses import replace

import pytest

from pidrei_ai.api.openai_responses import OpenAIResponsesOptions, build_params
from pidrei_ai.types import Context, Model, ModelCost, OpenAIResponsesCompat, TextContent, UserMessage


MODEL = Model(
    id="gpt-5-mini",
    name="GPT-5 Mini",
    api="openai-responses",
    provider="openai",
    base_url="https://api.openai.com/v1",
    reasoning=True,
    input=["text"],
    cost=ModelCost(),
    context_window=400_000,
    max_tokens=128_000,
)

CONTEXT = Context(system_prompt="", messages=[UserMessage(content=[TextContent(text="hi")], timestamp=0)], tools=[])


def capture_payload(api_key: str, request_model: Model = MODEL) -> dict:
    return build_params(
        request_model,
        CONTEXT,
        OpenAIResponsesOptions(api_key=api_key, max_tokens=1000, temperature=0.5, cache_retention="long"),
    )


def test_omits_request_fields_that_token_sharing_rejects():
    payload = capture_payload("chatgpt-access-token")

    assert "max_output_tokens" not in payload
    assert "temperature" not in payload
    assert payload.get("prompt_cache_retention") is None


def test_omits_prompt_cache_options_on_models_with_explicit_prompt_cache_mode():
    explicit_cache_model = replace(MODEL, compat=OpenAIResponsesCompat(supports_explicit_prompt_cache_mode=True))

    sign_in_payload = capture_payload("chatgpt-access-token", explicit_cache_model)
    api_key_payload = capture_payload("sk-proj-test", explicit_cache_model)

    assert sign_in_payload.get("prompt_cache_options") is None
    assert api_key_payload["prompt_cache_options"] == {"ttl": "30m"}


@pytest.mark.parametrize(
    ("api_key", "request_model"),
    [
        pytest.param("sk-proj-test", MODEL, id="OpenAI API keys"),
        pytest.param(
            "gateway-key",
            replace(MODEL, base_url="https://gateway.example.com/v1"),
            id="other OpenAI-compatible endpoints",
        ),
    ],
)
def test_keeps_those_fields(api_key: str, request_model: Model):
    payload = capture_payload(api_key, request_model)

    assert payload["max_output_tokens"] == 1000
    assert payload["temperature"] == 0.5
    assert payload["prompt_cache_retention"] == "24h"
