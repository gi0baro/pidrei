"""Mirror of pi coding-agent test/clipboard-image-bmp-conversion.test.ts.

WSL2/WSLg clipboards often provide image/bmp instead of image/png. pi runs the
case for the command (linux) and native (win32) readers; only the command
reader exists here, swapped on the module like pi's `vi.mock`.
"""

import struct

import pytest

from pidrei.utils import clipboard_image
from pidrei.utils.clipboard_image import read_clipboard_image


def _tiny_bmp_1x1_red_24bpp() -> bytes:
    # Minimal 1x1 24bpp BMP (BGR + row padding to 4 bytes)
    # File size = 14 (BMP header) + 40 (DIB header) + 4 (pixel row) = 58
    file_header = b"BM" + struct.pack("<IHHI", 58, 0, 0, 54)
    info_header = struct.pack("<IiiHHIIiiII", 40, 1, 1, 1, 24, 0, 4, 0, 0, 0, 0)
    pixels = bytes([0x00, 0x00, 0xFF, 0x00])  # B, G, R + padding
    return file_header + info_header + pixels


@pytest.mark.tonio
async def test_linux_converts_command_bmp_to_png():
    async def run(command, args, **_options):
        if command == "wl-paste" and "--list-types" in args:
            return b"image/bmp\n"
        if command == "wl-paste" and "image/bmp" in args:
            return _tiny_bmp_1x1_red_24bpp()
        return None

    original = clipboard_image.run_clipboard_command
    clipboard_image.run_clipboard_command = run
    try:
        image = await read_clipboard_image({"env": {"WAYLAND_DISPLAY": "wayland-0"}, "platform": "linux"})
    finally:
        clipboard_image.run_clipboard_command = original

    assert image is not None
    assert image["mimeType"] == "image/png"
    assert list(image["bytes"][:4]) == [0x89, 0x50, 0x4E, 0x47]
