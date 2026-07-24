"""Tool path resolution helpers (port of pi `tools/path-utils.ts`)."""

import re
import unicodedata

from pidrei_ai.utils.cancel import CancelToken

from ..types import get_or_throw


_UNICODE_SPACES = re.compile("[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202f"


def _normalize_tool_path(path: str) -> str:
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized.removeprefix("@")


async def resolve_tool_path(env, path: str, cancel: CancelToken | None = None) -> str:
    return get_or_throw(await env.absolute_path(_normalize_tool_path(path), cancel))


async def resolve_read_tool_path(env, path: str, cancel: CancelToken | None = None) -> str:
    """Resolve a read path, probing common Unicode variants the model may have flattened."""
    resolved = await resolve_tool_path(env, path, cancel)
    variants = [
        resolved,
        re.sub(r" (AM|PM)\.", f"{_NARROW_NO_BREAK_SPACE}\\1.", resolved, flags=re.IGNORECASE),
        unicodedata.normalize("NFD", resolved),
        resolved.replace("'", "’"),
        unicodedata.normalize("NFD", resolved).replace("'", "’"),
    ]

    for variant in dict.fromkeys(variants):
        if get_or_throw(await env.exists(variant, cancel)):
            return variant
    return resolved
