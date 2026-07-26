"""Image component (port of pi tui ``components/image.ts``).

``theme`` is a ``{"fallbackColor": fn}`` record; ``options`` mirrors pi's
``ImageOptions``: ``{"maxWidthCells", "maxHeightCells", "filename", "imageId"}``.
"""

import math

from ..terminal_image import (
    allocate_image_id,
    get_capabilities,
    get_cell_dimensions,
    get_image_dimensions,
    image_fallback,
    render_image,
)


__all__ = ["Image"]


class Image:
    def __init__(
        self,
        base64_data: str,
        mime_type: str,
        theme: dict,
        options: dict | None = None,
        dimensions: dict | None = None,
    ) -> None:
        self._base64_data = base64_data
        self._mime_type = mime_type
        self._theme = theme
        self._options = options if options is not None else {}
        self._dimensions = (
            dimensions or get_image_dimensions(base64_data, mime_type) or {"widthPx": 800, "heightPx": 600}
        )
        self._image_id: int | None = self._options.get("imageId")

        self._cached_lines: list[str] | None = None
        self._cached_width: int | None = None

    def get_image_id(self) -> int | None:
        """Get the Kitty image ID used by this image (if any)."""
        return self._image_id

    def invalidate(self) -> None:
        self._cached_lines = None
        self._cached_width = None

    def render(self, width: int) -> list[str]:
        if self._cached_lines is not None and self._cached_width == width:
            return self._cached_lines

        max_width_option = self._options.get("maxWidthCells")
        max_width = max(1, min(width - 2, max_width_option if max_width_option is not None else 60))
        cell_dimensions = get_cell_dimensions()
        default_max_height = max(1, math.ceil((max_width * cell_dimensions["widthPx"]) / cell_dimensions["heightPx"]))
        max_height_option = self._options.get("maxHeightCells")
        max_height = max_height_option if max_height_option is not None else default_max_height

        caps = get_capabilities()

        if caps["images"]:
            if caps["images"] == "kitty" and self._image_id is None:
                self._image_id = allocate_image_id()
            result = render_image(
                self._base64_data,
                self._dimensions,
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
                lines = [self._theme["fallbackColor"](fallback)]
        else:
            fallback = image_fallback(self._mime_type, self._dimensions, self._options.get("filename"))
            lines = [self._theme["fallbackColor"](fallback)]

        self._cached_lines = lines
        self._cached_width = width

        return lines
