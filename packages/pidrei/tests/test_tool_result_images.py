"""Mirror of pi coding-agent test/tool-result-images.test.ts.

pi hand-encodes its test PNGs with `node:zlib`; pidrei already depends on
Pillow for `process_image`, so the fixtures are built with it. `toBe(content)`
identity assertions become `is content`, which carries the same contract: an
unchanged result must be the caller's own list.
"""

import base64
import io
import struct

import pytest
from PIL import Image

from pidrei.utils.tool_result_images import normalize_tool_result_images
from pidrei_ai.types import ImageContent, TextContent


TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="


def create_png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def read_png_dimensions(base64_data: str) -> tuple[int, int]:
    buffer = base64.b64decode(base64_data)
    return struct.unpack(">I", buffer[16:20])[0], struct.unpack(">I", buffer[20:24])[0]


def create_tiny_bmp_1x1_red_24bpp() -> bytes:
    buffer = bytearray(58)
    buffer[0:2] = b"BM"
    struct.pack_into("<I", buffer, 2, len(buffer))
    struct.pack_into("<I", buffer, 10, 54)
    struct.pack_into("<I", buffer, 14, 40)
    struct.pack_into("<i", buffer, 18, 1)
    struct.pack_into("<i", buffer, 22, 1)
    struct.pack_into("<H", buffer, 26, 1)
    struct.pack_into("<H", buffer, 28, 24)
    struct.pack_into("<I", buffer, 30, 0)
    struct.pack_into("<I", buffer, 34, 4)
    buffer[56] = 0xFF
    return bytes(buffer)


def image_block(data: bytes, mime_type: str) -> ImageContent:
    return ImageContent(data=base64.b64encode(data).decode("ascii"), mime_type=mime_type)


@pytest.mark.tonio
async def test_returns_the_original_list_when_there_are_no_image_blocks():
    content = [TextContent(text="no images here")]

    assert await normalize_tool_result_images(content) is content


@pytest.mark.tonio
async def test_returns_the_original_list_when_images_are_already_within_limits():
    content = [
        TextContent(text="screenshot"),
        ImageContent(data=TINY_PNG_BASE64, mime_type="image/png"),
    ]

    assert await normalize_tool_result_images(content) is content


@pytest.mark.tonio
async def test_resizes_oversized_images_and_reports_the_original_dimensions():
    content = [image_block(create_png(2400, 4800), "image/png")]

    normalized = await normalize_tool_result_images(content)

    assert normalized is not content
    assert len(normalized) == 2
    assert normalized[0].type == "image"
    width, height = read_png_dimensions(normalized[0].data)
    assert width <= 2000
    assert height <= 2000
    assert normalized[1].type == "text"
    assert "original 2400x4800" in normalized[1].text


@pytest.mark.tonio
async def test_leaves_oversized_images_alone_when_auto_resize_is_disabled():
    content = [image_block(create_png(2400, 4800), "image/png")]

    assert await normalize_tool_result_images(content, auto_resize_images=False) is content


@pytest.mark.tonio
async def test_converts_unsupported_image_formats_even_when_auto_resize_is_disabled():
    content = [image_block(create_tiny_bmp_1x1_red_24bpp(), "image/bmp")]

    normalized = await normalize_tool_result_images(content, auto_resize_images=False)

    assert normalized is not content
    assert normalized[0].type == "image"
    assert normalized[0].mime_type == "image/png"
    assert normalized[1] == TextContent(text="[Image converted from image/bmp to image/png.]")


@pytest.mark.tonio
async def test_keeps_undecodable_images_instead_of_dropping_tool_output():
    content = [ImageContent(data="bm90LWFuLWltYWdl", mime_type="image/png")]

    assert await normalize_tool_result_images(content) is content


@pytest.mark.tonio
async def test_preserves_surrounding_text_blocks_and_their_order():
    content = [
        TextContent(text="before"),
        image_block(create_png(2400, 100), "image/png"),
        TextContent(text="after"),
    ]

    normalized = await normalize_tool_result_images(content)

    assert [block.type for block in normalized] == ["text", "image", "text", "text"]
    assert normalized[0] == TextContent(text="before")
    assert normalized[3] == TextContent(text="after")
