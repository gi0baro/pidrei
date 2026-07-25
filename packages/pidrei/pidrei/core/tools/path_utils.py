"""Mirror of pi coding-agent src/core/tools/path-utils.ts."""

import os
import re
import unicodedata

from ...utils.paths import normalize_path, resolve_path


_NARROW_NO_BREAK_SPACE = "\u202f"
_AM_PM_RE = re.compile(r" (AM|PM)\.", re.IGNORECASE)


def _try_macos_screenshot_path(file_path: str) -> str:
    return _AM_PM_RE.sub(lambda match: f"{_NARROW_NO_BREAK_SPACE}{match.group(1)}.", file_path)


def _try_nfd_variant(file_path: str) -> str:
    # macOS stores filenames in NFD (decomposed) form, try converting user input to NFD
    return unicodedata.normalize("NFD", file_path)


def _try_curly_quote_variant(file_path: str) -> str:
    # macOS uses U+2019 (right single quotation mark) in screenshot names like "Capture d'écran"
    # Users typically type U+0027 (straight apostrophe)
    return file_path.replace("'", "\u2019")


def path_exists(file_path: str) -> bool:
    return os.path.exists(file_path)


def expand_path(file_path: str) -> str:
    return normalize_path(file_path, normalize_unicode_spaces=True, strip_at_prefix=True)


def resolve_to_cwd(file_path: str, cwd: str) -> str:
    """Resolve a path relative to the given cwd. Handles ~ expansion and absolute paths."""
    return resolve_path(file_path, cwd, normalize_unicode_spaces=True, strip_at_prefix=True)


def resolve_read_path(file_path: str, cwd: str) -> str:
    resolved = resolve_to_cwd(file_path, cwd)

    if path_exists(resolved):
        return resolved

    # Try macOS AM/PM variant (narrow no-break space before AM/PM)
    am_pm_variant = _try_macos_screenshot_path(resolved)
    if am_pm_variant != resolved and path_exists(am_pm_variant):
        return am_pm_variant

    # Try NFD variant (macOS stores filenames in NFD form)
    nfd_variant = _try_nfd_variant(resolved)
    if nfd_variant != resolved and path_exists(nfd_variant):
        return nfd_variant

    # Try curly quote variant (macOS uses U+2019 in screenshot names)
    curly_variant = _try_curly_quote_variant(resolved)
    if curly_variant != resolved and path_exists(curly_variant):
        return curly_variant

    # Try combined NFD + curly quote (for French macOS screenshots like "Capture d'écran")
    nfd_curly_variant = _try_curly_quote_variant(nfd_variant)
    if nfd_curly_variant != resolved and path_exists(nfd_curly_variant):
        return nfd_curly_variant

    return resolved
