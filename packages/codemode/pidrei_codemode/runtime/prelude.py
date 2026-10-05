"""Mirror of pi codemode src/runtime/prelude-source.ts.

pi evaluates a 383-line JavaScript prelude inside the VM that builds `tools`,
the output helpers and the store on top of one bridge function. On Monty the
script reaches the host through host functions and host objects directly, so
almost all of that is host-side Python here: `output_text` (`text()`),
`image_output` (`image()`) and `ScriptStore` (`store()`/`load()`), with pi's
error strings. What runs inside the session is `PRELUDE`, fed before each
script: the helpers that need no host round trip.
"""

import json
import re
import threading
from typing import Any

from ..types import CodemodeImageItem, CodemodeStoreWrites


MAX_STORE_VALUE_CHARS = 256 * 1024
MAX_STORE_TOTAL_CHARS = 1024 * 1024
# Output one script may produce with `text()`, `image()`, and `print()`:
# characters of text and base64 image data, and items. The host keeps all
# output until the script ends, so without a limit a script that prints in a
# loop grows the host's memory until it crashes. The item limit covers loops
# that output empty strings; it counts `text()` and `image()` calls, since
# `print()` output reaches the host in buffered chunks and counts by its
# characters only.
MAX_OUTPUT_CHARS = 16 * 1024 * 1024
MAX_OUTPUT_ITEMS = 100_000
OUTPUT_LIMIT_MESSAGE = (
    f"script output exceeded the limit of {MAX_OUTPUT_CHARS} characters or {MAX_OUTPUT_ITEMS} text() and image() "
    "calls. Print a summary instead, or write large data to a file with a tool."
)

# Defined in the session before each script (`ALL_TOOLS` is an input of the
# same feed). Session globals persist across feeds, so the script sees them.
PRELUDE = """import asyncio

async def _settle(call):
    try:
        return {'status': 'fulfilled', 'value': await call}
    except Exception as e:
        return {'status': 'rejected', 'reason': str(e)}

async def all_settled(*calls):
    return await asyncio.gather(*[_settle(c) for c in calls])

def has_tool(name):
    return any(t['name'] == name for t in ALL_TOOLS)
"""

STORE_HINT = (
    "store() is for small state such as IDs or summaries. Show images with image(), keep large data in "
    "variables, or write it to a file with a tool."
)

IMAGE_HELPER_EXPECTS = "image expects a non-empty image URL string, an object with image_url, or a raw MCP image block"


def to_json(value: Any) -> str:
    """`JSON.stringify` for script values: compact, and only what JSON can hold."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def json_round_trip(value: Any) -> Any:
    """A copy of `value` as JSON would carry it (tuples become lists), or
    `TypeError` when JSON cannot hold it."""
    try:
        return json.loads(to_json(value))
    except ValueError as error:
        raise TypeError(str(error)) from None


def output_text(value: Any) -> str:
    """`text(value)`: strings and other scalars as their string form,
    everything else as compact JSON."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return str(value)
    try:
        return to_json(value)
    except (TypeError, ValueError) as error:
        raise TypeError(str(error)) from None


def _image_url(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        raise TypeError(IMAGE_HELPER_EXPECTS)
    if "image_url" in value and value["image_url"] is not None:
        if not isinstance(value["image_url"], str):
            raise TypeError(IMAGE_HELPER_EXPECTS)
        return value["image_url"]
    kind = value.get("type")
    if not isinstance(kind, str):
        raise TypeError(IMAGE_HELPER_EXPECTS)
    if kind != "image":
        raise TypeError(f'image only accepts MCP image blocks, got "{kind}"')
    data = value.get("data")
    if not isinstance(data, str) or data == "":
        raise TypeError("image expected MCP image data")
    if data.lower().startswith("data:"):
        return data
    return f"data:;base64,{data}"


# Base64 of the signatures of the formats providers accept inline (PNG, JPEG
# except JPEG-LS, GIF, "RIFF....WEBP"). Signatures start at byte 0, so their
# encodings are prefixes.
_IMAGE_SIGNATURES = (
    ("image/png", re.compile(r"iVBORw0KGg")),
    ("image/jpeg", re.compile(r"/9j/(?!9)")),
    ("image/gif", re.compile(r"R0lGOD[dl]h")),
    ("image/webp", re.compile(r"UklG.{8}RUJQ")),
)
_BASE64 = re.compile(r"[A-Za-z0-9+/]+={0,2}")
_WHITESPACE = re.compile(r"\s+")


def image_output(value: Any) -> CodemodeImageItem:
    """`image(value)`: a base64 `data:` URL, an `{'image_url': ...}` dict, or an
    MCP image block. Validates the base64 and takes the type from the data."""
    url = _image_url(value)
    if url == "":
        raise TypeError(IMAGE_HELPER_EXPECTS)
    colon = url.find(":")
    scheme = "" if colon == -1 else url[:colon].lower()
    if scheme in ("http", "https"):
        raise TypeError("remote image URLs are not supported in tool outputs. Pass a base64 data URI instead")
    comma = url.find(",")
    header = [] if comma == -1 else url[colon + 1 : comma].split(";")
    if scheme != "data" or comma == -1 or all(part.lower() != "base64" for part in header[1:]):
        raise TypeError("invalid image output. Pass a base64 data URI instead")
    # Providers reject the whole request on a bad image, and a persisted image
    # block would be resent on every later turn. Line breaks from wrapped
    # base64 are dropped. The declared type is ignored in favor of the
    # detected one, as providers also reject mismatches.
    data = _WHITESPACE.sub("", url[comma + 1 :])
    if len(data) % 4 != 0 or not _BASE64.fullmatch(data):
        raise TypeError("invalid image output. The image data is not valid base64 (truncated or corrupted?)")
    head = data[:16]
    for mime_type, pattern in _IMAGE_SIGNATURES:
        if pattern.match(head):
            return CodemodeImageItem(data=data, mime_type=mime_type)
    raise TypeError("invalid image output. The image data is not a PNG, JPEG, GIF, or WebP image")


class ScriptStore:
    """`store(key, value)` and `load(key)` over a snapshot of JSON values.

    Sizes count key and JSON characters. Calls come from the script one at a
    time; the guard is for the reads of `writes()` by the execution's owner.
    """

    def __init__(self, snapshot: dict[str, Any] | None) -> None:
        self._guard = threading.Lock()
        # key -> JSON text.
        self._stored: dict[str, str] = {}
        # key -> JSON text, or None for a deleted key.
        self._writes: dict[str, str | None] = {}
        for key, value in (snapshot or {}).items():
            try:
                self._stored[key] = to_json(value)
            except TypeError, ValueError:
                continue
        self._chars = sum(len(key) + len(text) for key, text in self._stored.items())

    @staticmethod
    def _check_key(name: str, key: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(f"{name}() key must be a string")

    def store(self, key: Any, value: Any) -> None:
        self._check_key("store", key)
        with self._guard:
            previous = len(key) + len(self._stored[key]) if key in self._stored else 0
            if value is None:
                self._stored.pop(key, None)
                self._chars -= previous
                self._writes[key] = None
                return
        try:
            text = to_json(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"store({to_json(key)}) value is not JSON-serializable: {error}") from None
        if len(text) > MAX_STORE_VALUE_CHARS:
            raise ValueError(
                f"store({to_json(key)}) value has {len(text)} characters of JSON, more than the limit of "
                f"{MAX_STORE_VALUE_CHARS}. {STORE_HINT}"
            )
        with self._guard:
            previous = len(key) + len(self._stored[key]) if key in self._stored else 0
            total = self._chars - previous + len(key) + len(text)
            if total > MAX_STORE_TOTAL_CHARS:
                raise ValueError(
                    f"store is full: stored values would exceed {MAX_STORE_TOTAL_CHARS} characters of JSON. "
                    f"Delete keys with store(key, None). {STORE_HINT}"
                )
            self._stored[key] = text
            self._chars = total
            self._writes[key] = text

    def load(self, key: Any) -> Any:
        self._check_key("load", key)
        with self._guard:
            text = self._stored.get(key)
        return None if text is None else json.loads(text)

    def writes(self) -> CodemodeStoreWrites:
        with self._guard:
            writes = list(self._writes.items())
        return CodemodeStoreWrites(
            set={key: json.loads(text) for key, text in writes if text is not None},
            delete=tuple(key for key, text in writes if text is None),
        )
