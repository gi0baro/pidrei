"""Mirror of pi coding-agent src/utils/tool-result-images.ts."""

import base64

import tonio.colored as tonio

from pidrei_ai.types import ImageContent, TextContent

from .image_process import process_image


__all__ = ["normalize_tool_result_images"]


async def normalize_tool_result_images(content: list, *, auto_resize_images: bool = True) -> list:
    """Normalize image blocks returned by tool results.

    The `read` tool and `@file` CLI attachments run their images through
    `process_image`, but tools that produce images themselves (extensions, MCP
    bridges, screenshot tools) hand back arbitrary base64 payloads that go
    straight into session history and every subsequent provider request.
    Oversized images make the provider reject the whole conversation, not just
    the offending turn, so normalize them once as they enter history.

    Returns the original list when nothing changed so callers can skip
    rewriting the result.
    """
    if not any(block.type == "image" for block in content):
        return content

    normalized: list = []
    changed = False

    for block in content:
        if block.type != "image":
            normalized.append(block)
            continue

        # Pillow work is CPU-bound and off the runtime, like every other
        # process_image call site.
        processed = await tonio.spawn_blocking(
            process_image,
            base64.b64decode(block.data),
            block.mime_type,
            auto_resize_images=auto_resize_images,
        )
        if not processed.ok:
            # Unlike `read`, keep the original block. The tool already produced
            # this image and the failure may just be an unavailable image
            # backend, so passing it through preserves the behavior tools have
            # today instead of silently deleting their output.
            normalized.append(block)
            continue

        hints = processed.hints or []
        if processed.data == block.data and processed.mime_type == block.mime_type and not hints:
            normalized.append(block)
            continue

        normalized.append(ImageContent(data=processed.data, mime_type=processed.mime_type))
        if hints:
            normalized.append(TextContent(text="\n".join(hints)))
        changed = True

    return normalized if changed else content
