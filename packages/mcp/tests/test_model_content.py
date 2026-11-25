"""Mirror of pi mcp test/model-content.test.ts."""

import base64

import pytest

from pidrei_mcp import ContentBlock, McpModelContentOptions, to_mcp_model_content


_LINK: ContentBlock = {
    "type": "resource_link",
    "uri": "docs://guide",
    "name": "guide",
    "title": "The Guide",
    "mimeType": "text/markdown",
    "size": 2048,
    "description": "How to use it",
}
_PDF: ContentBlock = {
    "type": "resource",
    "resource": {"uri": "file:///r/report.pdf", "mimeType": "application/pdf", "blob": "JVBERg=="},
}


@pytest.mark.tonio
async def test_names_the_read_tool_in_resource_links_when_there_is_one():
    assert await to_mcp_model_content(
        "docs", {"content": [_LINK]}, McpModelContentOptions(read_resource_tool="read_mcp_resource")
    ) == [
        {
            "type": "text",
            "text": '[Resource docs://guide "The Guide" (text/markdown, 2.0KB): How to use it. Read it with read_mcp_resource (server "docs")]',
        }
    ]
    assert await to_mcp_model_content(
        "docs", {"content": [{"type": "resource_link", "uri": "file:///a", "name": "a"}]}
    ) == [{"type": "text", "text": '[Resource file:///a "a"]'}]


@pytest.mark.tonio
async def test_saves_binary_resources_shows_text_blobs_as_text_and_keeps_images():
    saved: list[tuple[str | bytes, str]] = []

    async def save_output(data: str | bytes, extension: str) -> str:
        saved.append((data, extension))
        return f"/tmp/saved{extension}"

    json_blob = base64.b64encode(b'{"a":1}').decode()
    content: list[ContentBlock] = [
        _PDF,
        {
            "type": "resource",
            "resource": {"uri": "docs://data", "mimeType": "application/json; charset=utf-8", "blob": json_blob},
        },
        {"type": "resource", "resource": {"uri": "docs://logo", "mimeType": "image/png", "blob": "AAAA"}},
        {"type": "resource", "resource": {"uri": "docs://raw", "blob": "AAAA"}},
    ]
    assert await to_mcp_model_content(
        "docs", {"content": content}, McpModelContentOptions(save_output=save_output)
    ) == [
        {
            "type": "text",
            "text": "[Binary resource file:///r/report.pdf (application/pdf, 4B) saved to /tmp/saved.pdf]",
        },
        {"type": "text", "text": '{"a":1}'},
        {"type": "image", "data": "AAAA", "mimeType": "image/png"},
        {"type": "text", "text": "[Binary resource docs://raw (unknown type, 3B) saved to /tmp/saved.bin]"},
    ]
    # The blocks are converted concurrently: compare the saves as a set of calls, not in call order.
    assert sorted(saved) == sorted([(b"%PDF", ".pdf"), (base64.b64decode("AAAA"), ".bin")])


@pytest.mark.tonio
async def test_leaves_binary_resources_out_without_a_saver_and_says_why_when_saving_fails():
    assert await to_mcp_model_content("docs", {"content": [_PDF]}) == [
        {"type": "text", "text": "[binary resource file:///r/report.pdf (application/pdf) omitted]"}
    ]

    async def failing(_data: str | bytes, _extension: str) -> str:
        raise Exception("disk full")

    assert await to_mcp_model_content("docs", {"content": [_PDF]}, McpModelContentOptions(save_output=failing)) == [
        {
            "type": "text",
            "text": "[Binary resource file:///r/report.pdf (application/pdf, 4B) could not be saved: disk full]",
        }
    ]


@pytest.mark.tonio
async def test_shows_structured_content_as_json_when_there_are_no_content_blocks():
    assert await to_mcp_model_content("docs", {"content": [], "structuredContent": [{"id": 1}]}) == [
        {"type": "text", "text": '[\n  {\n    "id": 1\n  }\n]'}
    ]
