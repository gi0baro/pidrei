"""Mirror of pi coding-agent src/utils/paths.ts (POSIX-only; win32 branches not ported)."""

import os
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path


_UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")

_FILE_URL_RE = re.compile(r"^file://")

_PERCENT_ESCAPE_RE = re.compile("[0-9A-Fa-f]{2}")


def get_file_revision(path: str) -> str | None:
    """stat-identity revision used to skip redundant locked reloads."""
    try:
        stats = os.stat(path)
        return f"{stats.st_dev}:{stats.st_ino}:{stats.st_size}:{stats.st_mtime_ns}:{stats.st_ctime_ns}"
    except OSError:
        return None


def canonicalize_path(path: str) -> str:
    """Resolve a path to its canonical (real) form, following symlinks.

    Falls back to the raw path if resolution fails (e.g. the target does
    not exist yet), so that callers never crash on missing filesystem
    entries.
    """
    try:
        return str(Path(path).resolve(strict=True))
    except OSError:
        return path


def is_local_path(value: str) -> bool:
    """Returns True if the value is NOT a package source (npm:, git:, etc.)
    or a remote URL protocol. Bare names, relative paths, and file: URLs
    are considered local.
    """
    trimmed = value.strip()
    # Known non-local prefixes. file: URLs are local paths and are intentionally resolved by resolve_path().
    return not trimmed.startswith(("npm:", "git:", "github:", "http:", "https:", "ssh:"))


def _decode_uri_component(value: str) -> str:
    decoded = bytearray()
    index = 0
    while index < len(value):
        char = value[index]
        if char == "%":
            hex_part = value[index + 1 : index + 3]
            if len(hex_part) != 2 or not _PERCENT_ESCAPE_RE.fullmatch(hex_part):
                raise ValueError(f"URI malformed: {value}")
            decoded.append(int(hex_part, 16))
            index += 3
        else:
            decoded.extend(char.encode("utf-8"))
            index += 1
    return decoded.decode("utf-8")


def file_url_to_path(url: str) -> str:
    """POSIX equivalent of Node's fileURLToPath: strict host and percent-escape validation."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "file":
        raise ValueError(f"Invalid file URL: {url}")
    if parsed.netloc not in ("", "localhost"):
        raise ValueError(f'File URL host must be "localhost" or empty: {url}')
    if re.search("%2f", parsed.path, re.IGNORECASE):
        raise ValueError(f"File URL path must not include encoded / characters: {url}")
    return _decode_uri_component(parsed.path)


def normalize_path(
    input: str,
    *,
    trim: bool = False,
    expand_tilde: bool = True,
    home_dir: str | None = None,
    strip_at_prefix: bool = False,
    normalize_unicode_spaces: bool = False,
) -> str:
    normalized = input.strip() if trim else input
    if normalize_unicode_spaces:
        normalized = _UNICODE_SPACES.sub(" ", normalized)
    if strip_at_prefix and normalized.startswith("@"):
        normalized = normalized[1:]

    if expand_tilde:
        home = home_dir if home_dir is not None else os.path.expanduser("~")
        if normalized == "~":
            return home
        if normalized.startswith("~/"):
            return os.path.join(home, normalized[2:])

    if _FILE_URL_RE.match(normalized):
        return file_url_to_path(normalized)

    return normalized


def resolve_path(
    input: str,
    base_dir: str | None = None,
    *,
    trim: bool = False,
    expand_tilde: bool = True,
    home_dir: str | None = None,
    strip_at_prefix: bool = False,
    normalize_unicode_spaces: bool = False,
) -> str:
    if base_dir is None:
        base_dir = os.getcwd()
    normalized = normalize_path(
        input,
        trim=trim,
        expand_tilde=expand_tilde,
        home_dir=home_dir,
        strip_at_prefix=strip_at_prefix,
        normalize_unicode_spaces=normalize_unicode_spaces,
    )
    normalized_base_dir = normalize_path(base_dir)
    if os.path.isabs(normalized):
        return os.path.abspath(normalized)
    return os.path.abspath(os.path.join(normalized_base_dir, normalized))


def get_cwd_relative_path(file_path: str, cwd: str) -> str | None:
    resolved_cwd = resolve_path(cwd)
    resolved_path = resolve_path(file_path, resolved_cwd)
    relative_path = os.path.relpath(resolved_path, resolved_cwd)
    is_inside_cwd = relative_path == "." or (
        relative_path != ".." and not relative_path.startswith(".." + os.sep) and not os.path.isabs(relative_path)
    )

    return relative_path if is_inside_cwd else None


def format_path_relative_to_cwd_or_absolute(file_path: str, cwd: str) -> str:
    absolute_path = resolve_path(file_path, cwd)
    relative = get_cwd_relative_path(absolute_path, cwd)
    return relative if relative is not None else absolute_path


def mark_path_ignored_by_cloud_sync(path: str) -> None:
    if sys.platform == "darwin":
        attrs = ["com.dropbox.ignored", "com.apple.fileprovider.ignore#P"]
    elif sys.platform.startswith("linux"):
        attrs = ["user.com.dropbox.ignored"]
    else:
        attrs = []

    for attr in attrs:
        if sys.platform == "darwin":
            argv = ["xattr", "-w", attr, "1", path]
        else:
            argv = ["setfattr", "-n", attr, "-v", "1", path]
        try:
            subprocess.run(  # noqa: S603
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
            )
        except OSError:
            pass
