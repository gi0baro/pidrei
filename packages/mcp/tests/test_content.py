"""Mirror of pi mcp test/content.test.ts."""

from pidrei_mcp import to_llm_content


def test_passes_text_and_images_through_and_replaces_other_blocks_with_text():
    blocks = [
        {"type": "text", "text": "hello", "annotations": {"priority": 1}},
        {"type": "image", "data": "aW1n", "mimeType": "image/png", "_meta": {"x": 1}},
        {"type": "audio", "data": "YXVk", "mimeType": "audio/wav"},
        {"type": "resource_link", "uri": "file:///a.txt", "name": "a.txt"},
        {"type": "resource", "resource": {"uri": "file:///b.txt", "text": "inline"}},
        {"type": "resource", "resource": {"uri": "file:///c.png", "mimeType": "image/png", "blob": "Yw=="}},
        {"type": "resource", "resource": {"uri": "file:///d.bin", "blob": "ZA=="}},
    ]
    assert to_llm_content({"content": blocks}) == [
        {"type": "text", "text": "hello"},
        {"type": "image", "data": "aW1n", "mimeType": "image/png"},
        {"type": "text", "text": "[audio audio/wav omitted]"},
        {"type": "text", "text": "a.txt: file:///a.txt"},
        {"type": "text", "text": "inline"},
        {"type": "image", "data": "Yw==", "mimeType": "image/png"},
        {"type": "text", "text": "[binary resource file:///d.bin (unknown type) omitted]"},
    ]


def test_falls_back_to_the_structured_content_as_json_when_there_are_no_blocks():
    assert to_llm_content({"content": [], "structuredContent": {"n": 1}}) == [
        {"type": "text", "text": '{\n  "n": 1\n}'}
    ]
    assert to_llm_content({"content": [{"type": "text", "text": "n=1"}], "structuredContent": {"n": 1}}) == [
        {"type": "text", "text": "n=1"}
    ]
