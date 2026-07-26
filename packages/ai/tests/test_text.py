from pidrei_ai.types import ImageContent, TextContent, ThinkingContent, ToolCall
from pidrei_ai.utils.text import content_text


def test_plain_string_passthrough():
    assert content_text("hello") == "hello"


def test_joins_text_blocks_and_skips_others():
    content = [
        TextContent(text="one"),
        ThinkingContent(thinking="hidden"),
        ImageContent(data="...", mime_type="image/png"),
        ToolCall(id="t", name="tool", arguments={}),
        TextContent(text="two"),
    ]
    assert content_text(content) == "one\ntwo"
    assert content_text(content, separator=" ") == "one two"
