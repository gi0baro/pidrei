"""Mirror of pi's google-shared-image-tool-result-routing.test.ts."""

from pidrei_ai.api.google_shared import convert_messages
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    ModelCost,
    TextContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)


def make_model(api: str, provider: str, id: str) -> Model:
    return Model(
        id=id,
        name=id,
        api=api,
        provider=provider,
        base_url="https://example.com",
        reasoning=True,
        input=["text", "image"],
        cost=ModelCost(),
        context_window=128000,
        max_tokens=8192,
    )


def make_context(model: Model) -> Context:
    now = 1
    return Context(
        messages=[
            UserMessage(content="read the files", timestamp=now),
            AssistantMessage(
                content=[
                    ToolCall(id="call_a", name="read", arguments={"path": "a.txt"}),
                    ToolCall(id="call_img", name="read", arguments={"path": "image.png"}),
                    ToolCall(id="call_b", name="read", arguments={"path": "b.txt"}),
                ],
                api=model.api,
                provider=model.provider,
                model=model.id,
                usage=Usage(),
                stop_reason="toolUse",
                timestamp=now,
            ),
            ToolResultMessage(
                tool_call_id="call_a",
                tool_name="read",
                content=[TextContent(text="alpha text")],
                is_error=False,
                timestamp=now,
            ),
            ToolResultMessage(
                tool_call_id="call_img",
                tool_name="read",
                content=[ImageContent(data="abc", mime_type="image/png")],
                is_error=False,
                timestamp=now,
            ),
            ToolResultMessage(
                tool_call_id="call_b",
                tool_name="read",
                content=[TextContent(text="beta text")],
                is_error=False,
                timestamp=now,
            ),
        ]
    )


def test_keeps_separate_synthetic_image_turn_for_gemini_2x_google_api_models():
    model = make_model("google-generative-ai", "google", "gemini-2.5-flash")
    contents = convert_messages(model, make_context(model))

    assert len(contents) == 5
    assert all(part.get("functionResponse") for part in contents[2]["parts"]) is True
    assert contents[3]["parts"][0]["text"] == "Tool result image:"
    assert contents[3]["parts"][1]["inlineData"]
    assert contents[4]["parts"][0]["functionResponse"]


def test_nests_image_tool_results_for_gemini_3_google_api_models():
    model = make_model("google-generative-ai", "google", "gemini-3-pro-preview")
    contents = convert_messages(model, make_context(model))

    assert len(contents) == 3
    tool_result_turn = contents[2]
    assert len(tool_result_turn["parts"]) == 3
    image_response = tool_result_turn["parts"][1]["functionResponse"]
    assert image_response
    assert len(image_response["parts"]) == 1
    assert image_response["parts"][0]["inlineData"]
