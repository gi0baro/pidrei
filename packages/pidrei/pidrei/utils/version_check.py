"""Mirror of pi coding-agent src/utils/version-check.ts.

Release records are ``{"version", "packageName"?, "note"?}``. The version
check still asks pi.dev (pidrei tracks upstream pi releases); PIDREI_OFFLINE
and PIDREI_SKIP_VERSION_CHECK are the PI_* environment equivalents.
"""

import json
import os
import re

from .user_agent import get_pidrei_user_agent


_LATEST_VERSION_URL = "https://pi.dev/api/latest-version"
_DEFAULT_VERSION_CHECK_TIMEOUT_MS = 10000

_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$")


def _parse_semver(version: str):
    match = _SEMVER_RE.match(version.strip())
    if not match:
        return None
    prerelease = match.group(4)
    prerelease_key: tuple
    if prerelease is None:
        # Releases sort after any prerelease of the same version
        prerelease_key = (1,)
    else:
        parts = []
        for part in prerelease.split("."):
            if part.isdigit():
                parts.append((0, int(part), ""))
            else:
                parts.append((1, 0, part))
        prerelease_key = (0, tuple(parts))
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)), prerelease_key)


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

    from pidrei_ai.utils.http import request_timeout, shared_client

    timeout_ms = options.get("timeoutMs", _DEFAULT_VERSION_CHECK_TIMEOUT_MS)
    response = await shared_client().get(
        _LATEST_VERSION_URL,
        headers={
            "User-Agent": get_pidrei_user_agent(current_version),
            "accept": "application/json",
        },
        timeout=request_timeout(timeout_ms),
    )
    if response.status_code < 200 or response.status_code >= 300:
        return None

    body = await response.read()
    data = json.loads(body.decode("utf-8", "replace") if isinstance(body, bytes) else body)
    version = data.get("version")
    if not isinstance(version, str) or not version.strip():
        return None
    package_name = data.get("packageName")
    package_name = package_name.strip() if isinstance(package_name, str) and package_name.strip() else None
    note = data.get("note")
    note = note.strip() if isinstance(note, str) and note.strip() else None
    release = {"version": version.strip(), "packageName": package_name}
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
