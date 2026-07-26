"""Image signature detection and base64 helpers (port of pi `tools/image.ts`)."""

import base64


_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def detect_supported_image_mime_type(buffer: bytes) -> str | None:
    if buffer.startswith(b"\xff\xd8\xff"):
        return None if len(buffer) > 3 and buffer[3] == 0xF7 else "image/jpeg"
    if buffer.startswith(_PNG_SIGNATURE):
        return "image/png" if _is_png(buffer) and not _is_animated_png(buffer) else None
    if buffer.startswith(b"GIF"):
        return "image/gif"
    if buffer.startswith(b"RIFF") and buffer[8:12] == b"WEBP":
        return "image/webp"
    if buffer.startswith(b"BM") and _is_bmp(buffer):
        return "image/bmp"
    return None


def encode_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _is_png(buffer: bytes) -> bool:
    return len(buffer) >= 16 and _read_uint32_be(buffer, len(_PNG_SIGNATURE)) == 13 and buffer[12:16] == b"IHDR"


def _is_animated_png(buffer: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(buffer):
        chunk_length = _read_uint32_be(buffer, offset)
        chunk_type_offset = offset + 4
        if buffer[chunk_type_offset : chunk_type_offset + 4] == b"acTL":
            return True
        if buffer[chunk_type_offset : chunk_type_offset + 4] == b"IDAT":
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(buffer):
            return False
        offset = next_offset
    return False


def _is_bmp(buffer: bytes) -> bool:
    if len(buffer) < 26:
        return False
    declared_file_size = _read_uint32_le(buffer, 2)
    pixel_data_offset = _read_uint32_le(buffer, 10)
    dib_header_size = _read_uint32_le(buffer, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False

    if dib_header_size == 12:
        color_planes = _read_uint16_le(buffer, 22)
        bits_per_pixel = _read_uint16_le(buffer, 24)
    elif 40 <= dib_header_size <= 124:
        if len(buffer) < 30:
            return False
        color_planes = _read_uint16_le(buffer, 26)
        bits_per_pixel = _read_uint16_le(buffer, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in (1, 4, 8, 16, 24, 32)


def _byte_at(buffer: bytes, offset: int) -> int:
    return buffer[offset] if offset < len(buffer) else 0


def _read_uint16_le(buffer: bytes, offset: int) -> int:
    return _byte_at(buffer, offset) + (_byte_at(buffer, offset + 1) << 8)


def _read_uint32_be(buffer: bytes, offset: int) -> int:
    return (
        _byte_at(buffer, offset) * 0x1000000
        + (_byte_at(buffer, offset + 1) << 16)
        + (_byte_at(buffer, offset + 2) << 8)
        + _byte_at(buffer, offset + 3)
    )


def _read_uint32_le(buffer: bytes, offset: int) -> int:
    return (
        _byte_at(buffer, offset)
        + (_byte_at(buffer, offset + 1) << 8)
        + (_byte_at(buffer, offset + 2) << 16)
        + _byte_at(buffer, offset + 3) * 0x1000000
    )
