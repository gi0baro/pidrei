"""Mirror of pi's google-shared-signed-empty-blocks.test.ts.

Gemini can attach `thoughtSignature` to a response part whose visible text is
empty (e.g. a thought burst preceding a function call) and requires the
signature echoed back on the next request. Dropping such blocks while
rebuilding history silently breaks the reasoning chain: the model then
intermittently ends a mid-task turn with a thought-only STOP and no tool call.
These tests pin the rule: an empty text/thinking block is skipped only when it
is UNSIGNED.
"""

from pidrei_ai.api.google_shared import convert_messages
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    ModelCost,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    UserMessage,
)


VALID_SIG = "AAAAAAAAAAAAAAAAAAAAAA=="


def make_model(id: str = "gemini-3-pro-preview") -> Model:
    return Model(
        id=id,
        name=id,
        api="google-generative-ai",
        provider="google",
        base_url="https://example.com",
        reasoning=True,
        input=["text"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=8192,
    )


def make_context(api: str, provider: str, model_id: str, content: list) -> Context:
    now = 1
    return Context(
        messages=[
            UserMessage(content="Hi", timestamp=now),
            AssistantMessage(
                content=content,
                api=api,
                provider=provider,
                model=model_id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=now,
            ),
        ]
    )


def model_turn_of(contents: list[dict]) -> dict | None:
    return next((c for c in contents if c["role"] == "model"), None)


def test_keeps_a_signed_empty_thinking_block_so_its_signature_is_echoed_back():
    model = make_model()
    contents = convert_messages(
        model,
        make_context(
            model.api,
            model.provider,
            model.id,
            [
                ThinkingContent(thinking="", thinking_signature=VALID_SIG),
                ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
            ],
        ),
    )
    model_turn = model_turn_of(contents)
    signed = [p for p in model_turn["parts"] if p.get("thoughtSignature") == VALID_SIG]
    assert len(signed) == 1
    assert signed[0].get("thought") is True


def test_keeps_a_signed_empty_text_block_the_same_way():
    model = make_model()
    contents = convert_messages(
        model,
        make_context(
            model.api,
            model.provider,
            model.id,
            [
                TextContent(text="", text_signature=VALID_SIG),
                ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
            ],
        ),
    )
    model_turn = model_turn_of(contents)
    signed = [p for p in model_turn["parts"] if p.get("thoughtSignature") == VALID_SIG]
    assert len(signed) == 1


def test_still_drops_unsigned_empty_blocks():
    model = make_model()
    contents = convert_messages(
        model,
        make_context(
            model.api,
            model.provider,
            model.id,
            [
                ThinkingContent(thinking=""),
                TextContent(text="   "),
                ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
            ],
        ),
    )
    model_turn = model_turn_of(contents)
    assert len(model_turn["parts"]) == 1
    assert model_turn["parts"][0].get("functionCall")


def test_still_drops_signed_empty_blocks_from_a_different_provider_model():
    model = make_model()
    contents = convert_messages(
        model,
        make_context(
            model.api,
            model.provider,
            "other-model",
            [
                ThinkingContent(thinking="", thinking_signature=VALID_SIG),
                TextContent(text="", text_signature=VALID_SIG),
                ToolCall(id="call_1", name="bash", arguments={"command": "ls"}),
            ],
        ),
    )
    model_turn = model_turn_of(contents)
    assert len(model_turn["parts"]) == 1
    assert model_turn["parts"][0].get("functionCall")
    assert VALID_SIG not in repr(model_turn)
