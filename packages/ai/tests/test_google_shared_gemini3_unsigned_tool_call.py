"""Mirror of pi's google-shared-gemini3-unsigned-tool-call.test.ts.

The last two cases are pidrei-only: pi never replays an *invalid* signature, nor
pins where the cross-model stripping actually happens. Both came out of mutation
testing this slice.
"""

from dataclasses import replace

import pytest

from pidrei_ai.api.google_shared import convert_messages, requires_tool_call_id
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def make_gemini3_model(api: str, provider: str, id: str = "gemini-3-pro-preview") -> Model:
    return Model(
        id=id,
        name="Gemini 3 Pro Preview",
        api=api,
        provider=provider,
        base_url="https://example.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=8192,
    )


def make_context(model: Model, thought_signature: str | None = None) -> Context:
    now = 1
    return Context(
        messages=[
            UserMessage(content="Hi", timestamp=now),
            AssistantMessage(
                content=[
                    ToolCall(
                        id="call_1",
                        name="bash",
                        arguments={"command": "echo hi"},
                        **({"thought_signature": thought_signature} if thought_signature else {}),
                    ),
                    ToolCall(id="call_2", name="bash", arguments={"command": "ls -la"}),
                ],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=now,
            ),
            ToolResultMessage(
                tool_call_id="call_1",
                tool_name="bash",
                content=[TextContent(text="hi")],
                is_error=False,
                timestamp=now,
            ),
            ToolResultMessage(
                tool_call_id="call_2",
                tool_name="bash",
                content=[TextContent(text="files")],
                is_error=False,
                timestamp=now,
            ),
        ]
    )


@pytest.mark.parametrize(
    "model",
    [
        make_gemini3_model("google-generative-ai", "google"),
        make_gemini3_model("google-generative-ai", "google", "gemini-3.6-flash"),
        make_gemini3_model("google-vertex", "google-vertex"),
    ],
    ids=lambda model: f"{model.id}-{model.api}",
)
def test_preserves_tool_call_ids_via_history(model):
    contents = convert_messages(model, make_context(model))

    function_call_ids = [
        part["functionCall"]["id"]
        for content in contents
        for part in content.get("parts", [])
        if part.get("functionCall", {}).get("id")
    ]
    function_response_ids = [
        part["functionResponse"]["id"]
        for content in contents
        for part in content.get("parts", [])
        if part.get("functionResponse", {}).get("id")
    ]

    assert function_call_ids == ["call_1", "call_2"]
    assert function_response_ids == ["call_1", "call_2"]


def test_does_not_add_skip_thought_signature_validator_for_unsigned_google_gen_ai_tool_calls():
    model = make_gemini3_model("google-generative-ai", "google")
    contents = convert_messages(model, make_context(replace(model, id="other-model")))

    model_turn = next((c for c in contents if c["role"] == "model"), None)
    assert model_turn

    function_call_parts = [p for p in model_turn["parts"] if p.get("functionCall") is not None]
    assert len(function_call_parts) == 2
    assert function_call_parts[0].get("thoughtSignature") is None
    assert function_call_parts[1].get("thoughtSignature") is None
    assert "skip_thought_signature_validator" not in repr(model_turn)

    text_parts = [p for p in model_turn["parts"] if p.get("text") is not None]
    historical_text = [p for p in text_parts if "Historical context" in p["text"]]
    assert len(historical_text) == 0


def test_does_not_add_skip_thought_signature_validator_for_unsigned_vertex_tool_calls():
    model = make_gemini3_model("google-vertex", "google-vertex")
    contents = convert_messages(model, make_context(model))
    model_turn = next((c for c in contents if c["role"] == "model"), None)
    function_call_parts = [p for p in model_turn["parts"] if p.get("functionCall") is not None]

    assert len(function_call_parts) == 2
    assert function_call_parts[0].get("thoughtSignature") is None
    assert function_call_parts[1].get("thoughtSignature") is None
    assert "skip_thought_signature_validator" not in repr(model_turn)


def test_preserves_valid_thought_signature_when_present_for_the_same_provider_and_model():
    model = make_gemini3_model("google-generative-ai", "google")
    valid_sig = "AAAAAAAAAAAAAAAAAAAAAA=="
    contents = convert_messages(model, make_context(model, valid_sig))
    model_turn = next((c for c in contents if c["role"] == "model"), None)
    function_call_parts = [p for p in model_turn["parts"] if p.get("functionCall") is not None]

    assert len(function_call_parts) == 2
    assert function_call_parts[0]["thoughtSignature"] == valid_sig
    assert function_call_parts[1].get("thoughtSignature") is None


def test_does_not_add_a_thought_signature_for_non_gemini_3_models():
    model = make_gemini3_model("google-generative-ai", "google", "gemini-2.5-flash")
    contents = convert_messages(model, make_context(replace(model, id="other-model")))
    model_turn = next((c for c in contents if c["role"] == "model"), None)
    function_call_parts = [p for p in model_turn["parts"] if p.get("functionCall") is not None]
    function_response_parts = [
        part for content in contents for part in content.get("parts", []) if part.get("functionResponse") is not None
    ]

    assert len(function_call_parts) == 2
    assert all(part["functionCall"].get("id") is None for part in function_call_parts)
    assert all(part.get("thoughtSignature") is None for part in function_call_parts)
    assert len(function_response_parts) == 2
    assert all(part["functionResponse"].get("id") is None for part in function_response_parts)


@pytest.mark.parametrize(
    ("expected", "model_id"),
    [
        (False, "gemini-2.5-flash"),
        (True, "gemini-3.6-flash"),
        (True, "claude-sonnet-4-5"),
        (True, "gpt-oss-120b"),
    ],
)
def test_requires_tool_call_id(expected, model_id):
    assert requires_tool_call_id(model_id) is expected


# --- pidrei-only ---------------------------------------------------------------


def signed_context(model_id: str, api: str, signature: str) -> Context:
    """One assistant turn whose every block carries `signature`."""
    return Context(
        messages=[
            UserMessage(content="Hi", timestamp=1),
            AssistantMessage(
                content=[
                    TextContent(text="answer", text_signature=signature),
                    ThinkingContent(thinking="pondering", thinking_signature=signature),
                    ToolCall(id="call_1", name="bash", arguments={}, thought_signature=signature),
                ],
                api=api,
                provider="google",
                model=model_id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=1,
            ),
        ]
    )


def signatures_in(contents: list[dict]) -> list[str | None]:
    model_turn = next(c for c in contents if c["role"] == "model")
    return [part.get("thoughtSignature") for part in model_turn["parts"]]


def test_a_valid_signature_survives_on_every_block_type_for_the_same_model():
    model = make_gemini3_model("google-generative-ai", "google")
    valid = "AAAAAAAAAAAAAAAAAAAAAA=="

    contents = convert_messages(model, signed_context(model.id, model.api, valid))

    assert signatures_in(contents) == [valid, valid, valid]


def test_a_signature_with_non_base64_characters_is_dropped_even_for_the_same_model():
    # Google types thought signatures as TYPE_BYTES, so a non-base64 value would
    # be rejected by the API rather than ignored. Length is deliberately a
    # multiple of four here, so only the charset check can reject it.
    model = make_gemini3_model("google-generative-ai", "google")

    contents = convert_messages(model, signed_context(model.id, model.api, "not-base64!!"))

    assert signatures_in(contents) == [None, None, None]


def test_a_signature_whose_length_is_not_a_multiple_of_four_is_dropped():
    model = make_gemini3_model("google-generative-ai", "google")

    contents = convert_messages(model, signed_context(model.id, model.api, "AAAAA"))

    assert signatures_in(contents) == [None, None, None]


def test_cross_model_signatures_are_gone_before_conversion_even_sees_them():
    """Pins the layering, not just the outcome.

    `convert_messages` re-checks provider+model before keeping a signature, but
    `transform_messages` has already stripped it (thinking and text blocks are
    rebuilt as plain text, tool calls have `thought_signature` cleared), so that
    re-check is defence in depth and cannot be reached with a signature present.
    """
    model = make_gemini3_model("google-generative-ai", "google")
    valid = "AAAAAAAAAAAAAAAAAAAAAA=="

    by_id = convert_messages(model, signed_context("other-model", model.api, valid))
    # Same provider and id, different api: `convert_messages` alone would call
    # this the same model — `transform_messages` also compares the api, and wins.
    by_api = convert_messages(model, signed_context(model.id, "openai-completions", valid))

    assert signatures_in(by_id) == [None, None, None]
    assert signatures_in(by_api) == [None, None, None]
