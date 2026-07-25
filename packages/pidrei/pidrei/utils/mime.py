"""Mirror of pi coding-agent src/utils/mime.ts (detection shared with the
agent-package port)."""

from pidrei_agent.harness.tools.image import detect_supported_image_mime_type


IMAGE_TYPE_SNIFF_BYTES = 256


def detect_supported_image_mime_type_from_file(file_path: str) -> str | None:
    with open(file_path, "rb") as f:
        buffer = f.read(IMAGE_TYPE_SNIFF_BYTES)
    return detect_supported_image_mime_type(buffer)


__all__ = ["IMAGE_TYPE_SNIFF_BYTES", "detect_supported_image_mime_type", "detect_supported_image_mime_type_from_file"]
