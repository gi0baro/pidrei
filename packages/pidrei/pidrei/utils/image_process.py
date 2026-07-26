"""Mirror of pi coding-agent src/utils/image-process.ts + image-resize-core.ts.

pi uses Photon (Rust/WASM); pidrei uses Pillow. The resize strategy is the
same: cap dimensions, try PNG and JPEG at descending quality, then shrink by
25% steps until the base64 payload fits.
"""

import base64
import io
from dataclasses import dataclass

from PIL import Image, ImageOps


# 4.5MB of base64 payload. Provides headroom below Anthropic's 5MB limit.
DEFAULT_RESIZE_MAX_BYTES = int(4.5 * 1024 * 1024)


@dataclass(slots=True, kw_only=True)
class ImageResizeOptions:
    max_width: int = 2000
    max_height: int = 2000
    max_bytes: int = DEFAULT_RESIZE_MAX_BYTES
    jpeg_quality: int = 80


@dataclass(slots=True)
class ResizedImage:
    data: str  # base64
    mime_type: str
    original_width: int
    original_height: int
    width: int
    height: int
    was_resized: bool


@dataclass(slots=True)
class ProcessImageResult:
    ok: bool
    data: str | None = None
    mime_type: str | None = None
    hints: list[str] | None = None
    message: str | None = None


def _base_mime_type(mime_type: str) -> str:
    return mime_type.split(";")[0].strip().lower()


def _normalize_supported_image_mime_type(mime_type: str) -> str | None:
    base = _base_mime_type(mime_type)
    if base == "image/png":
        return "image/png"
    if base in ("image/jpeg", "image/jpg"):
        return "image/jpeg"
    if base == "image/gif":
        return "image/gif"
    if base == "image/webp":
        return "image/webp"
    return None


def convert_image_bytes_to_png(data: bytes) -> bytes | None:
    try:
        with Image.open(io.BytesIO(data)) as image:
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
            return buffer.getvalue()
    except Exception:
        return None


def convert_to_png(base64_data: str, mime_type: str) -> dict | None:
    """Convert image to PNG for terminal display (kitty requires PNG).

    Mirror of pi utils/image-convert.ts convertToPng; Pillow replaces Photon.
    Returns ``{"data", "mimeType"}`` or None if conversion failed.
    """
    if mime_type == "image/png":
        return {"data": base64_data, "mimeType": mime_type}

    try:
        data = base64.b64decode(base64_data)
    except Exception:
        return None
    png_bytes = convert_image_bytes_to_png(data)
    if png_bytes is None:
        return None

    return {"data": base64.b64encode(png_bytes).decode("ascii"), "mimeType": "image/png"}


def _encode(image: Image.Image, format: str, quality: int | None = None) -> tuple[str, int, str]:
    buffer = io.BytesIO()
    if format == "PNG":
        image.save(buffer, "PNG")
        mime_type = "image/png"
    else:
        converted = image.convert("RGB") if image.mode not in ("RGB", "L") else image
        converted.save(buffer, "JPEG", quality=quality if quality is not None else 80)
        mime_type = "image/jpeg"
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return data, len(data), mime_type


def resize_image(
    input_bytes: bytes,
    mime_type: str,
    options: ImageResizeOptions | None = None,
) -> ResizedImage | None:
    """Resize an image to fit max dimensions and encoded size; None when it
    cannot fit below max_bytes."""
    opts = options if options is not None else ImageResizeOptions()
    input_base64_size = ((len(input_bytes) + 2) // 3) * 4

    try:
        with Image.open(io.BytesIO(input_bytes)) as raw_image:
            image = ImageOps.exif_transpose(raw_image)
            original_width, original_height = image.size
            format = mime_type.split("/")[1] if "/" in mime_type else "png"

            # Check if already within all limits (dimensions AND encoded size)
            if (
                original_width <= opts.max_width
                and original_height <= opts.max_height
                and input_base64_size < opts.max_bytes
            ):
                return ResizedImage(
                    data=base64.b64encode(input_bytes).decode("ascii"),
                    mime_type=mime_type or f"image/{format}",
                    original_width=original_width,
                    original_height=original_height,
                    width=original_width,
                    height=original_height,
                    was_resized=False,
                )

            # Calculate initial dimensions respecting max limits
            target_width = original_width
            target_height = original_height

            if target_width > opts.max_width:
                target_height = round(target_height * opts.max_width / target_width)
                target_width = opts.max_width
            if target_height > opts.max_height:
                target_width = round(target_width * opts.max_height / target_height)
                target_height = opts.max_height

            quality_steps = list(dict.fromkeys([opts.jpeg_quality, 85, 70, 55, 40]))
            current_width = target_width
            current_height = target_height

            while True:
                resized = image.resize((max(1, current_width), max(1, current_height)), Image.LANCZOS)
                candidates = [_encode(resized, "PNG")]
                for quality in quality_steps:
                    candidates.append(_encode(resized, "JPEG", quality))
                for data, encoded_size, candidate_mime in candidates:
                    if encoded_size < opts.max_bytes:
                        return ResizedImage(
                            data=data,
                            mime_type=candidate_mime,
                            original_width=original_width,
                            original_height=original_height,
                            width=current_width,
                            height=current_height,
                            was_resized=True,
                        )

                if current_width == 1 and current_height == 1:
                    break

                next_width = 1 if current_width == 1 else max(1, int(current_width * 0.75))
                next_height = 1 if current_height == 1 else max(1, int(current_height * 0.75))
                if next_width == current_width and next_height == current_height:
                    break

                current_width = next_width
                current_height = next_height

            return None
    except Exception:
        return None


def format_dimension_note(result: ResizedImage) -> str | None:
    if not result.was_resized:
        return None

    scale = result.original_width / result.width
    return (
        f"[Image: original {result.original_width}x{result.original_height}, "
        f"displayed at {result.width}x{result.height}. "
        f"Multiply coordinates by {scale:.2f} to map to original image.]"
    )


def _conversion_hint(from_mime: str | None, to_mime: str) -> str | None:
    if not from_mime or from_mime == to_mime:
        return None
    return f"[Image converted from {from_mime} to {to_mime}.]"


def process_image(
    data: bytes,
    mime_type: str,
    *,
    auto_resize_images: bool = True,
    resize_options: ImageResizeOptions | None = None,
) -> ProcessImageResult:
    normalized_mime = _normalize_supported_image_mime_type(mime_type)
    converted_from: str | None = None
    bytes_out = data
    if normalized_mime is None:
        png_bytes = convert_image_bytes_to_png(data)
        if png_bytes is None:
            return ProcessImageResult(
                ok=False, message="[Image omitted: could not be converted to a supported inline image format.]"
            )
        bytes_out = png_bytes
        normalized_mime = "image/png"
        converted_from = _base_mime_type(mime_type)

    if auto_resize_images:
        resized = resize_image(bytes_out, normalized_mime, resize_options)
        if resized is None:
            return ProcessImageResult(
                ok=False, message="[Image omitted: could not be resized below the inline image size limit.]"
            )

        hints: list[str] = []
        converted_hint = _conversion_hint(converted_from, resized.mime_type)
        if converted_hint:
            hints.append(converted_hint)
        dimension_note = format_dimension_note(resized)
        if dimension_note:
            hints.append(dimension_note)

        return ProcessImageResult(ok=True, data=resized.data, mime_type=resized.mime_type, hints=hints)

    hints = []
    converted_hint = _conversion_hint(converted_from, normalized_mime)
    if converted_hint:
        hints.append(converted_hint)

    return ProcessImageResult(
        ok=True, data=base64.b64encode(bytes_out).decode("ascii"), mime_type=normalized_mime, hints=hints
    )
