"""Mirror of pi's google-shared-image-tool-result-routing.test.ts."""

import dataclasses

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


# https://github.com/earendil-works/pi/issues/8720
def test_treats_whitespace_only_tool_result_text_as_no_text():
    model = make_model("google-generative-ai", "google", "gemini-3-pro-preview")
    context = make_context(model)
    # pi mutates the tool results' content in place; messages are frozen here
    # (recipe freeze-at-seam), so the context is rebuilt with replacements.
    messages = []
    for message in context.messages:
        if isinstance(message, ToolResultMessage) and message.tool_call_id == "call_a":
            message = dataclasses.replace(message, content=[TextContent(text="\r\n")])
        elif isinstance(message, ToolResultMessage) and message.tool_call_id == "call_img":
            message = dataclasses.replace(message, content=[TextContent(text=" \n"), *message.content])
        messages.append(message)
    contents = convert_messages(model, dataclasses.replace(context, messages=messages))

    responses = [part["functionResponse"]["response"] for part in contents[2]["parts"]]
    assert responses[0] == {"output": ""}
    assert responses[1] == {"output": "(see attached image)"}
    assert responses[2] == {"output": "beta text"}
