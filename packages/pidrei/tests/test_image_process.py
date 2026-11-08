"""Partial mirror of pi coding-agent test/image-process.test.ts: the 0.87.0
GIF-signature cases, and the 1.0.2 transcoder case and 1.5.0 PNG-or-JPEG case
of image-processing.test.ts (through `convert_image_to_png_base64`, pi's
`loadPngTranscoder`, and `resize_image`). BMP->PNG conversion is covered by
test_tool_result_images.py and test_tools.py.
"""

import base64
import io
import math

import pytest
from PIL import Image

from pidrei.utils.image_process import convert_image_to_png_base64, resize_image
from pidrei.utils.mime import detect_supported_image_mime_type
from pidrei_ai.types import ModelImageResizeOptions


@pytest.mark.parametrize("signature", ["GIF87a", "GIF89a"])
def test_detects_the_complete_gif_signature(signature):
    assert detect_supported_image_mime_type(signature.encode("ascii")) == "image/gif"


# Issue #10292: the TUI uses this converter to show non-PNG images on Kitty-protocol terminals.

_TINY_JPEG_2X1 = (
    "/9j/4AAQSkZJRgABAgAAAQABAAD/wAARCAABAAIDAREAAhEBAxEB/9sAQwADAgIDAgIDAwMDBAMDBAUIBQUEBAUKBwcGCAwKDAwLCgsLDQ4SEA0O"
    "EQ4LCxAWEBETFBUVFQwPFxgWFBgSFBUU/9sAQwEDBAQFBAUJBQUJFA0LDRQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQU"
    "FBQUFBQUFBQU/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKB"
    "kaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZ"
    "mqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQF"
    "BgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5"
    "OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX"
    "2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD4H8Q/8h/Uv+vmX/0M1/o1wJ/ySWU/9g1D/wBNRMOM/wDkp8z/AOv9b/05I//Z"
)


def _app1_segment(payload: bytes) -> bytes:
    return b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload


def _jpeg_with_xmp_before_orientation() -> str:
    jpeg = base64.b64decode(_TINY_JPEG_2X1)
    xmp = _app1_segment(b'http://ns.adobe.com/xap/1.0/\0<x:xmpmeta xmlns:x="adobe:ns:meta/"/>')
    orientation6 = _app1_segment(b"Exif\0\0" + bytes.fromhex("49492a0008000000010012010300010000000600000000000000"))
    return base64.b64encode(jpeg[:2] + xmp + orientation6 + jpeg[2:]).decode()


@pytest.mark.tonio
async def test_converts_to_oriented_png_data():
    png = base64.b64decode(await convert_image_to_png_base64(_jpeg_with_xmp_before_orientation(), "image/jpeg"))
    assert [int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")] == [1, 2]
    assert await convert_image_to_png_base64(base64.b64encode(b"not an image").decode(), "image/jpeg") is None


def _photo_like_jpeg(width: int, height: int) -> bytes:
    """Photo-like JPEG: smooth gradients with light deterministic noise, so PNG compresses it poorly."""
    pixels = bytearray(width * height * 3)
    seed = 1
    for y in range(height):
        for x in range(width):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            noise = seed % 6
            i = (y * width + x) * 3
            pixels[i] = int(x / width * 200 + noise)
            pixels[i + 1] = int(y / height * 200 + noise)
            pixels[i + 2] = int(128 + math.sin(x / 30) * 60 + noise)
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), bytes(pixels)).save(buffer, "JPEG", quality=85)
    return buffer.getvalue()


# Regression test for #11551: PNG was returned whenever it fit, even when JPEG was much smaller.
def test_picks_jpeg_over_a_larger_png_when_resizing_a_photo():
    # pi's resizeImageInProcess is resize_image here (Pillow runs in the caller's blocking job).
    input_bytes = _photo_like_jpeg(1200, 800)
    result = resize_image(input_bytes, "image/jpeg", ModelImageResizeOptions(max_width=1000, max_height=1000))

    assert result is not None
    assert result.was_resized is True
    assert result.mime_type == "image/jpeg"
    assert result.width == 1000
    assert len(result.data) < math.ceil(len(input_bytes) / 3) * 4 * 2
