"""Update check against pidrei's GitHub releases (diverges from pi's version-check.ts).

pi polls `pi.dev/api/latest-version`, which reports *pi's* version — useless to
our users, and a request to someone else's service on every start. Phase 7
step 1 (2026-07-26) repoints it at the GitHub releases API, which is already
the channel of record now that PyPI is out.

Release records are ``{"version", "url", "note"?}``. pi's ``packageName`` is
gone: it named the npm dist-tag package to reinstall, and nothing ever read it.
``url`` is new — the release's own page, which the update notification links to
instead of pi's hosted changelog.

PIDREI_OFFLINE and PIDREI_SKIP_VERSION_CHECK are unchanged.
"""

import json
import os
import re

from .management_http import fetch_with_retry
from .user_agent import get_pidrei_user_agent


#: GitHub's "latest release" endpoint; excludes drafts and prereleases for us.
_LATEST_RELEASE_URL = "https://api.github.com/repos/gi0baro/pidrei/releases/latest"
RELEASES_URL = "https://github.com/gi0baro/pidrei/releases"
_DEFAULT_VERSION_CHECK_TIMEOUT_MS = 10000

# pi matches semver exactly. pidrei's scheme is pi's version plus our own
# segment (`0.82.0.N`, PEP 440), and dev builds carry `.devN`, so a
# three-segment-only pattern would fail to parse *both* sides of every
# comparison and silently fall back to string inequality — i.e. report an
# update whenever the strings differ at all.
_VERSION_RE = re.compile(
    r"^(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?"  # 3 or 4 numeric segments
    r"(?:[-.]?(?:(dev|a|b|rc|alpha|beta)\.?(\d*)|-([0-9A-Za-z.-]+)))?"  # pre-release, PEP 440 or semver
    r"(?:\+[0-9A-Za-z.-]+)?$"  # build metadata, ignored
)

#: Ordering of pre-release kinds; a release (no kind) sorts after all of them.
_PRERELEASE_RANK = {"dev": 0, "a": 1, "alpha": 1, "b": 2, "beta": 2, "rc": 3}


def _parse_semver(version: str):
    match = _VERSION_RE.match(version.strip())
    if not match:
        return None
    major, minor, patch, revision, kind, kind_number, semver_pre = match.groups()

    prerelease_key: tuple
    if kind is not None:
        prerelease_key = (0, ((0, _PRERELEASE_RANK[kind], ""), (0, int(kind_number or 0), "")))
    elif semver_pre is not None:
        parts = []
        for part in semver_pre.split("."):
            if part.isdigit():
                parts.append((0, int(part), ""))
            else:
                parts.append((1, 0, part))
        prerelease_key = (0, tuple(parts))
    else:
        # Releases sort after any prerelease of the same version
        prerelease_key = (1,)

    return (int(major), int(minor), int(patch), int(revision or 0), prerelease_key)


def compare_package_versions(left_version: str, right_version: str) -> int | None:
    left = _parse_semver(left_version)
    right = _parse_semver(right_version)
    if left is None or right is None:
        return None
    if left < right:
        return -1
    if left > right:
        return 1
    return 0


def is_newer_package_version(candidate_version: str, current_version: str) -> bool:
    comparison = compare_package_versions(candidate_version, current_version)
    if comparison is not None:
        return comparison > 0
    return candidate_version.strip() != current_version.strip()


async def get_latest_release(current_version: str, options: dict | None = None) -> dict | None:
    if os.environ.get("PIDREI_OFFLINE"):
        return None
    options = options or {}

    timeout_ms = options.get("timeoutMs", _DEFAULT_VERSION_CHECK_TIMEOUT_MS)
    response = await fetch_with_retry(
        _LATEST_RELEASE_URL,
        headers={
            "User-Agent": get_pidrei_user_agent(current_version),
            "accept": "application/vnd.github+json",
            "x-github-api-version": "2022-11-28",
        },
        max_retries=2 if options.get("retry") else 0,
        timeout_ms=timeout_ms,
    )
    if response.status_code < 200 or response.status_code >= 300:
        return None

    body = await response.read()
    data = json.loads(body.decode("utf-8", "replace") if isinstance(body, bytes) else body)
    # Tags are expected to be the bare version; `v` prefixes are tolerated
    # because a tap or a hand-cut tag may add one.
    tag = data.get("tag_name")
    if not isinstance(tag, str) or not tag.strip():
        return None
    version = tag.strip().removeprefix("v")
    if not version:
        return None

    url = data.get("html_url")
    url = url.strip() if isinstance(url, str) and url.strip() else RELEASES_URL
    note = data.get("body")
    note = note.strip() if isinstance(note, str) and note.strip() else None

    release = {"version": version, "url": url}
    if note:
        release["note"] = note
    return release


async def get_latest_version(current_version: str, options: dict | None = None) -> str | None:
    release = await get_latest_release(current_version, options)
    return release["version"] if release else None


async def check_for_new_version(current_version: str) -> dict | None:
    if os.environ.get("PIDREI_SKIP_VERSION_CHECK"):
        return None

    try:
        latest_release = await get_latest_release(current_version)
        if latest_release and is_newer_package_version(latest_release["version"], current_version):
            return latest_release
        return None
    except Exception:
        return None
