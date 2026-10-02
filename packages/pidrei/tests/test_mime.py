"""pidrei-only: image type sniffing from files (utils/mime.py).

pi reads the first 4100 bytes; an animated PNG's `acTL` chunk can sit past a
shorter sniff window, behind metadata chunks.
"""

from pidrei.utils.mime import detect_supported_image_mime_type_from_file_blocking


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    # The CRC is not checked by the sniffer.
    return len(data).to_bytes(4, "big") + chunk_type + data + b"\0\0\0\0"


def test_rejects_an_animated_png_whose_actl_chunk_follows_large_metadata(tmp_path):
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", bytes(13))
        + _chunk(b"tEXt", b"Comment\0" + b"x" * 1000)
        + _chunk(b"acTL", bytes(8))
        + _chunk(b"IDAT", bytes(16))
    )
    path = tmp_path / "animated.png"
    path.write_bytes(png)

    assert detect_supported_image_mime_type_from_file_blocking(str(path)) is None
