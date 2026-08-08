"""Mirror of pi's sampling-options.test.ts."""

import pytest

from pidrei_ai.api.anthropic_messages import stream_simple as stream_simple_anthropic
from pidrei_ai.api.openai_completions import stream_simple as stream_simple_completions
from pidrei_ai.types import Context, Model, ModelCost, SimpleStreamOptions, UserMessage


def make_context() -> Context:
    return Context(messages=[UserMessage(content="Hello", timestamp=1)])


def make_completions_model(**overrides) -> Model:
    defaults: dict = {
        "id": "custom-model",
        "name": "Custom Model",
        "api": "openai-completions",
        "provider": "custom-provider",
        "base_url": "http://127.0.0.1:9/v1",
        "reasoning": False,
        "input": ["text"],
        "cost": ModelCost(),
        "context_window": 128000,
        "max_tokens": 16384,
    }
    defaults.update(overrides)
    return Model(**defaults)


def make_anthropic_model() -> Model:
    return Model(
        id="vendor--claude",
        name="Vendor Proxy Claude",
        api="anthropic-messages",
        provider="vendor-proxy",
        base_url="http://127.0.0.1:9",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=200000,
        max_tokens=32000,
    )


async def capture_payload(model: Model, **option_kwargs) -> dict:
    captured: dict = {}

    async def on_payload(payload, _model):
        captured["payload"] = payload
        raise RuntimeError("payload captured")

    run = stream_simple_anthropic if model.api == "anthropic-messages" else stream_simple_completions
    await run(
        model,
        make_context(),
        SimpleStreamOptions(api_key="fake-key", on_payload=on_payload, **option_kwargs),
    ).result()

    assert "payload" in captured, "Expected payload to be captured before request failure"
    return captured["payload"]


@pytest.mark.tonio
async def test_merges_stream_option_sampling_params_into_the_request_body():
    payload = await capture_payload(make_completions_model(), sampling_params={"top_p": 0.95, "top_k": 0, "min_p": 0})

    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 0
    assert payload["min_p"] == 0


@pytest.mark.tonio
async def test_omits_sampling_params_when_neither_options_nor_model_set_them():
    payload = await capture_payload(make_completions_model())

    assert "temperature" not in payload
    assert "top_p" not in payload


@pytest.mark.tonio
async def test_applies_model_level_sampling_params():
    payload = await capture_payload(make_completions_model(sampling_params={"temperature": 1, "top_p": 0.95}))

    assert payload["temperature"] == 1
    assert payload["top_p"] == 0.95


@pytest.mark.tonio
async def test_merges_stream_option_keys_over_model_level_keys():
    payload = await capture_payload(
        make_completions_model(sampling_params={"top_p": 0.95, "min_p": 0.05}),
        sampling_params={"top_p": 0.5},
    )

    assert payload["top_p"] == 0.5
    assert payload["min_p"] == 0.05


@pytest.mark.tonio
async def test_overrides_named_request_fields():
    payload = await capture_payload(make_completions_model(), temperature=0, sampling_params={"temperature": 1})

    assert payload["temperature"] == 1


@pytest.mark.tonio
async def test_is_ignored_by_non_openai_compatible_apis():
    payload = await capture_payload(make_anthropic_model(), sampling_params={"top_p": 0.9, "top_k": 40})

    assert "top_p" not in payload
    assert "top_k" not in payload
