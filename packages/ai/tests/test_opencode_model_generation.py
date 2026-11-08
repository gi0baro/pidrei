"""Mirror of pi's opencode-model-generation.test.ts.

pi runs the generator end to end against mocked upstream catalogs; here the
catalog build runs in-process against the same mocked sources
(`catalog_generation_utils`). pi's failing strict run (exit 1 with the message
on stderr) is the build raising that message.
"""

import pytest

from tests.catalog_generation_utils import generate_chat_models


def _source_model(model_id: str, api: dict) -> dict:
    return {
        "id": model_id,
        "type": "chat",
        "name": model_id,
        "modalities": {"input": ["text"], "output": ["text"]},
        "capabilities": {"tools": {"supported": True}, "reasoning": {"supported": False}},
        "api": api,
    }


def _generate_opencode_models(models: list[dict], *, strict: bool = False) -> dict[str, dict]:
    return generate_chat_models(
        {"opencode": {"models": {model["id"]: model for model in models}}}, "opencode", strict=strict
    )


def test_uses_the_protocol_and_base_url_models_dev_lists_for_each_model(capsys):
    base = "https://example.test/zen"
    generated = _generate_opencode_models(
        [
            _source_model("chat", {"chat-completions": {"base_url": f"{base}/openai/v1"}}),
            _source_model("responses", {"responses": {"base_url": f"{base}/openai/v1"}}),
            _source_model("messages", {"messages": {"base_url": f"{base}/anthropic/v1"}}),
            _source_model("gemini", {"generate-content": {"base_url": f"{base}/google/v1beta"}}),
            # Native protocols win over Chat Completions regardless of listing order.
            _source_model(
                "both",
                {
                    "chat-completions": {"base_url": f"{base}/openai/v1"},
                    "messages": {"base_url": f"{base}/anthropic/v1"},
                },
            ),
            _source_model("live", {"bidi-generate-content": {"base_url": f"{base}/google/v1beta"}}),
        ]
    )

    assert "models.dev lists no supported API for opencode/live (bidi-generate-content)" in capsys.readouterr().err
    endpoints = {model["id"]: {"api": model["api"], "baseUrl": model["baseUrl"]} for model in generated.values()}
    assert endpoints == {
        "chat": {"api": "openai-completions", "baseUrl": f"{base}/openai/v1"},
        "responses": {"api": "openai-responses", "baseUrl": f"{base}/openai/v1"},
        # The Anthropic SDK appends /v1/messages itself.
        "messages": {"api": "anthropic-messages", "baseUrl": f"{base}/anthropic"},
        "gemini": {"api": "google-generative-ai", "baseUrl": f"{base}/google/v1beta"},
        "both": {"api": "anthropic-messages", "baseUrl": f"{base}/anthropic"},
    }


def test_fails_strict_generation_for_a_model_without_a_supported_protocol():
    with pytest.raises(RuntimeError) as excinfo:
        _generate_opencode_models(
            [_source_model("live", {"bidi-generate-content": {"base_url": "https://example.test/google/v1beta"}})],
            strict=True,
        )

    assert "models.dev lists no supported API for opencode/live (bidi-generate-content)" in str(excinfo.value)
