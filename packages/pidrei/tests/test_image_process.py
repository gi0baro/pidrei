"""Partial mirror of pi coding-agent test/image-process.test.ts: the 0.87.0
GIF-signature cases. BMP->PNG conversion is covered by
test_tool_result_images.py and test_tools.py.
"""

import pytest

from pidrei.utils.mime import detect_supported_image_mime_type


@pytest.mark.parametrize("signature", ["GIF87a", "GIF89a"])
def test_detects_the_complete_gif_signature(signature):
    assert detect_supported_image_mime_type(signature.encode("ascii")) == "image/gif"
