"""Partial mirror of pi coding-agent test/image-process.test.ts: the 0.87.0
GIF-signature cases, and the 1.0.2 transcoder case of image-processing.test.ts
(through `convert_image_to_png_base64`, pi's `loadPngTranscoder`). BMP->PNG
conversion is covered by test_tool_result_images.py and test_tools.py.
"""

import base64

import pytest

from pidrei.utils.image_process import convert_image_to_png_base64
from pidrei.utils.mime import detect_supported_image_mime_type


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
