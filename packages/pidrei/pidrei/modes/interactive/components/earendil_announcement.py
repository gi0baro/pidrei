"""Mirror of pi coding-agent src/modes/interactive/components/earendil-announcement.ts."""

import base64

from pidrei_tui import Container, Image, Spacer, Text

from ....config import get_bundled_interactive_asset_path
from ..theme import theme
from .dynamic_border import DynamicBorder


BLOG_URL = "https://mariozechner.at/posts/2026-04-08-ive-sold-out/"
IMAGE_FILENAME = "clankolas.png"

_cached_image_base64: str | None = None
_attempted_image_load = False


def _load_image_base64() -> str | None:
    global _cached_image_base64, _attempted_image_load
    if _attempted_image_load:
        return _cached_image_base64

    _attempted_image_load = True
    try:
        with open(get_bundled_interactive_asset_path(IMAGE_FILENAME), "rb") as f:
            _cached_image_base64 = base64.b64encode(f.read()).decode("ascii")
    except OSError:
        _cached_image_base64 = None
    return _cached_image_base64


class EarendilAnnouncementComponent(Container):
    def __init__(self) -> None:
        super().__init__()

        self.add_child(DynamicBorder(lambda text: theme.fg("accent", text)))
        self.add_child(Text(theme.bold(theme.fg("accent", "pi has joined Earendil")), 1, 0))
        self.add_child(Spacer(1))
        self.add_child(Text(theme.fg("muted", "Read the blog post:"), 1, 0))
        self.add_child(Text(theme.fg("mdLink", BLOG_URL), 1, 0))
        self.add_child(Spacer(1))

        image_base64 = _load_image_base64()
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
