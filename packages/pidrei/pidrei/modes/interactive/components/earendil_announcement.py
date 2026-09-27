"""Mirror of pi coding-agent src/modes/interactive/components/earendil-announcement.ts."""

import base64

from tonio.colored import fs

from pidrei_tui import Container, Image, Spacer, Text

from ....config import get_bundled_interactive_asset_path
from ..theme import theme
from .dynamic_border import DynamicBorder


BLOG_URL = "https://mariozechner.at/posts/2026-04-08-ive-sold-out/"
IMAGE_FILENAME = "clankolas.png"

_cached_image_base64: str | None = None
_attempted_image_load = False


async def load_earendil_image_base64() -> str | None:
    """The announcement image for `EarendilAnnouncementComponent`, read once."""
    global _cached_image_base64, _attempted_image_load
    if _attempted_image_load:
        return _cached_image_base64

    try:
        data = await fs.Path(get_bundled_interactive_asset_path(IMAGE_FILENAME)).read_bytes()
        image_base64 = base64.b64encode(data).decode("ascii")
    except OSError:
        image_base64 = None
    # Published together after the read: a concurrent caller that finds the
    # load not yet attempted reads the file too, and never a partial result.
    _cached_image_base64 = image_base64
    _attempted_image_load = True
    return image_base64


class EarendilAnnouncementComponent(Container):
    def __init__(self, image_base64: str | None) -> None:
        """`image_base64` comes from `load_earendil_image_base64()`."""
        super().__init__()

        self.add_child(DynamicBorder(lambda text: theme.fg("accent", text)))
        self.add_child(Text(theme.bold(theme.fg("accent", "pi has joined Earendil")), 1, 0))
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("muted", "Read the blog post:"), 1, 0))
        self.add_child(Text(theme.fg("mdLink", BLOG_URL), 1, 0))
        self.add_child(Spacer(1))

        if image_base64:
            self.add_child(
                Image(
                    image_base64,
                    "image/png",
                    {"fallbackColor": lambda text: theme.fg("muted", text)},
                    {"maxWidthCells": 56, "filename": IMAGE_FILENAME},
                )
            )
            self.add_child(Spacer(1))

        self.add_child(DynamicBorder(lambda text: theme.fg("accent", text)))
