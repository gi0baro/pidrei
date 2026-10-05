"""Image component (port of pi tui ``components/image.ts``).

``theme`` is a ``{"fallbackColor": fn}`` record; ``options`` mirrors pi's
``ImageOptions``: ``{"maxWidthCells", "maxHeightCells", "filename", "imageId"}``.

Kitty-protocol terminals accept PNG only. pi converts other images
synchronously while rendering, through a module-level transcoder. Here the
image takes its TUI first and converts through the TUI's `image_conversions`
(see `ImageConversions`), off the render path: without its PNG, `render()`
returns the text fallback and, once, starts a detached coroutine that awaits
the converter, stores the result through `tui.apply` and requests a frame.
An image without its PNG looks it up in the TUI's cache on every render, so an
owner that rebuilds its images picks the PNG up without converting again. A
failed conversion keeps the fallback until the TUI gets another converter.
`render()` runs under the UI state lock, as every component's does.
"""

import math
from typing import Any

import tonio.colored as tonio

from ..terminal_image import (
    allocate_image_id,
    get_capabilities,
    get_cell_dimensions,
    get_image_dimensions,
    get_png_dimensions,
    image_fallback,
    render_image,
)
from ..tui import IMAGE_CONVERSION_PENDING
from ..utils import truncate_to_width


__all__ = ["Image"]

_MISSING: Any = object()


class Image:
    def __init__(
        self,
        tui: Any,
        base64_data: str,
        mime_type: str,
        theme: dict,
        options: dict | None = None,
        dimensions: dict | None = None,
    ) -> None:
        self._tui = tui
        self._base64_data = base64_data
        self._mime_type = mime_type
        self._theme = theme
        self._options = options if options is not None else {}
        self._dimensions = (
            dimensions or get_image_dimensions(base64_data, mime_type) or {"widthPx": 800, "heightPx": 600}
        )
        self._image_id: int | None = self._options.get("imageId")
        # Converted PNG data for Kitty. Failures are not stored, so another converter can retry.
        self._png_data: str | None = None
        # The converter of the conversion this image started and is waiting for.
        self._converting_with: Any = None

        self._cached_lines: list[str] | None = None
        self._cached_width: int | None = None

    def get_image_id(self) -> int | None:
        """Get the Kitty image ID used by this image (if any)."""
        return self._image_id

    def invalidate(self) -> None:
        self._cached_lines = None
        self._cached_width = None

    def _kitty_png(self) -> str | None:
        """PNG data to send to a Kitty-protocol terminal, or None to show the
        fallback; starts the conversion when nobody has."""
        if self._png_data is not None:
            return self._png_data
        conversions = self._tui.image_conversions
        converter = conversions.converter
        if converter is None:
            return None
        cached = conversions.lookup(self._base64_data, _MISSING)
        if isinstance(cached, str):
            self._png_data = cached
            return cached
        if cached is _MISSING and self._converting_with is not converter:
            conversions.store(self._base64_data, IMAGE_CONVERSION_PENDING)
            self._converting_with = converter
            tonio.spawn.without_tracking(self._convert(conversions, converter))
        return None

    async def _convert(self, conversions: Any, converter: Any) -> None:
        source = self._base64_data
        failure: Exception | None = None
        try:
            png = await converter(source, self._mime_type)
        except Exception as error:
            png, failure = None, error

        def store() -> None:
            if self._converting_with is converter:
                self._converting_with = None
            # A converter set meanwhile dropped this one's results.
            if conversions.converter is not converter:
                return
            conversions.store(source, png)
            if png is not None:
                self._png_data = png
                self.invalidate()

        self._tui.apply(store)
        if failure is not None:
            self._tui.report_error(failure)
        self._tui.request_render()

    def render(self, width: int) -> list[str]:
        caps = get_capabilities()
        convert = caps["images"] == "kitty" and self._mime_type != "image/png"
        # A PNG that arrived since the last render replaces the cached fallback.
        if convert and self._png_data is None and self._kitty_png() is not None:
            self.invalidate()

        if self._cached_lines is not None and self._cached_width == width:
            return self._cached_lines

        max_width_option = self._options.get("maxWidthCells")
        max_width = max(1, min(width - 2, max_width_option if max_width_option is not None else 60))
        cell_dimensions = get_cell_dimensions()
        default_max_height = max(1, math.ceil((max_width * cell_dimensions["widthPx"]) / cell_dimensions["heightPx"]))
        max_height_option = self._options.get("maxHeightCells")
        max_height = max_height_option if max_height_option is not None else default_max_height

        data: str | None = self._base64_data
        dimensions = self._dimensions
        if convert:
            data = self._kitty_png()
            # Conversion may apply EXIF rotation, so prefer the PNG's own dimensions.
            if data is not None:
                dimensions = get_png_dimensions(data) or dimensions

        if caps["images"] and data is not None:
            if caps["images"] == "kitty" and self._image_id is None:
                self._image_id = allocate_image_id()
            result = render_image(
                data,
                dimensions,
                max_width_cells=max_width,
                max_height_cells=max_height,
                image_id=self._image_id,
                move_cursor=False,
            )

            if result is not None:
                # Store the image ID for later cleanup
                if result["imageId"]:
                    self._image_id = result["imageId"]

                if caps["images"] == "kitty":
                    # For Kitty: C=1 prevents cursor movement.
                    # Don't need the cursor movement.
                    lines = [result["sequence"]]

                    # Return `rows` lines so TUI accounts for image height.
                    for _ in range(result["rows"] - 1):
                        lines.append("")
                else:
                    # Return `rows` lines so TUI accounts for image height.
                    # First (rows-1) lines are empty and cleared before the image is drawn.
                    # Last line: move cursor back up, draw the image, then move back down
                    # so TUI cursor accounting stays inside the scroll area.
                    lines = ["" for _ in range(result["rows"] - 1)]
                    row_offset = result["rows"] - 1
                    move_up = f"\x1b[{row_offset}A" if row_offset > 0 else ""
                    lines.append(move_up + result["sequence"])
            else:
                fallback = image_fallback(self._mime_type, self._dimensions, self._options.get("filename"))
                lines = [truncate_to_width(self._theme["fallbackColor"](fallback), width)]
        else:
            fallback = image_fallback(self._mime_type, self._dimensions, self._options.get("filename"))
            lines = [truncate_to_width(self._theme["fallbackColor"](fallback), width)]

        self._cached_lines = lines
        self._cached_width = width

        return lines
