"""Mirror of pi's sampling-options.test.ts.

pi's `stream()`/`streamSimple()` dispatch by `model.api`; here the adapter
modules are called directly, keyed by the same api names.
"""

import pytest

from pidrei_ai.api import (
    anthropic_messages,
    azure_openai_responses,
    openai_completions,
    openai_responses,
)
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, StreamOptions, UserMessage
from pidrei_ai.utils.transcript import normalize_context


_ADAPTERS = {
    "openai-completions": openai_completions,
    "openai-responses": openai_responses,
    "azure-openai-responses": azure_openai_responses,
    "anthropic-messages": anthropic_messages,
}


def make_context() -> Context:
    return normalize_context(Context(messages=[UserMessage(content="Hello", timestamp=1)]))


def make_model(api: str, sampling_params: dict | None = None) -> Model:
    return Model(
        id="custom-model",
        name="Custom Model",
        api=api,
        provider="custom-provider",
        base_url="http://127.0.0.1:9/v1",
        reasoning=False,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=16384,
        sampling_params=sampling_params,
    )


def capturing_on_payload(captured: dict):
    async def on_payload(payload, _model):
        captured["payload"] = payload
        raise RuntimeError("payload captured")

    return on_payload


async def capture_payload(model: Model, **option_kwargs) -> dict:
    captured: dict = {}
    await (
        _ADAPTERS[model.api]
        .stream(
            model,
            make_context(),
            StreamOptions(api_key="fake-key", on_payload=capturing_on_payload(captured), **option_kwargs),
        )
        .result()
    )

    assert "payload" in captured, "Expected payload to be captured before request failure"
    return captured["payload"]


@pytest.mark.tonio
async def test_merges_request_sampling_params_into_the_request_body():
    payload = await capture_payload(
        make_model("openai-completions"), sampling_params={"top_p": 0.95, "top_k": 0, "min_p": 0}
    )

    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 0
    assert payload["min_p"] == 0


@pytest.mark.tonio
async def test_omits_sampling_params_when_neither_options_nor_model_set_them():
    payload = await capture_payload(make_model("openai-completions"))

    assert "temperature" not in payload
    assert "top_p" not in payload


# Model defaults must apply to direct stream()/complete() calls, not only stream_simple() (#9506)
@pytest.mark.tonio
@pytest.mark.parametrize("api", ["openai-completions", "openai-responses", "azure-openai-responses"])
async def test_applies_model_level_sampling_params_with_request_keys_taking_precedence(api):
    payload = await capture_payload(
        make_model(api, {"top_p": 0.95, "min_p": 0.05}),
        sampling_params={"top_p": 0.5},
    )

    assert payload["top_p"] == 0.5
    assert payload["min_p"] == 0.05


@pytest.mark.tonio
async def test_passes_request_sampling_params_through_stream_simple():
    captured: dict = {}

    await openai_completions.stream_simple(
        make_model("openai-completions"),
        make_context(),
        SimpleStreamOptions(
            sampling_params={"top_p": 0.5}, api_key="fake-key", on_payload=capturing_on_payload(captured)
        ),
    ).result()

    assert captured["payload"]["top_p"] == 0.5


@pytest.mark.tonio
async def test_overrides_named_request_fields():
    payload = await capture_payload(make_model("openai-completions"), temperature=0, sampling_params={"temperature": 1})

    assert payload["temperature"] == 1


@pytest.mark.tonio
async def test_is_ignored_by_non_openai_compatible_apis():
    payload = await capture_payload(make_model("anthropic-messages"), sampling_params={"top_p": 0.9, "top_k": 40})

    assert "top_p" not in payload
    assert "top_k" not in payload
