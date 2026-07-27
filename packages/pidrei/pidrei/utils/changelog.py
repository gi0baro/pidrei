"""Mirror of pi coding-agent src/utils/changelog.ts.

Entries are ``{"major", "minor", "patch", "build", "content"}`` records.

Three deliberate divergences, all following from whose changelog this reads:

- **The repo and base path are ours.** These constants turn a package-relative
  link in the changelog into a tag-pinned source link, so they have to name the
  repository and directory the changelog itself lives in. The doubled segment
  in ``packages/pidrei/pidrei`` is not a typo: the outer directory is the
  distribution, the inner one the Python package, and CHANGELOG.md ships inside
  the package next to README.md and docs/.
- **Versions carry a fourth segment.** pidrei versions are ``<pi version>.<our
  build>``, and the build number is the one that moves between pi releases —
  parsing three segments would make every pidrei-only release compare equal to
  the last one and silently skip the "What's New" panel. A header with only
  three segments still parses, as build ``0``.
- **pi's legacy-repository rewrite is gone.** It canonicalized
  ``badlogic|earendil-works/pi-mono`` URLs into pi's current repo — history we
  do not have. Left in place it would rewrite a genuine pi link into a pidrei
  URL that does not exist.
"""

import posixpath
import re
import sys
import urllib.parse

from tonio.colored import fs


GITHUB_REPO = "gi0baro/pidrei"
_CHANGELOG_LINK_BASE_PATH = "packages/pidrei/pidrei"
_URL_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_INLINE_MARKDOWN_LINK_RE = re.compile(r"(!?\[[^\]\n]+\]\()([^\s)]+)((?:\s+[^)]*)?\))")
_VERSION_HEADER_RE = re.compile(r"##\s+\[?(\d+)\.(\d+)\.(\d+)(?:\.(\d+))?\]?")


def _entry_version(entry: dict) -> str:
    return f"{entry['major']}.{entry['minor']}.{entry['patch']}.{entry['build']}"


def _normalize_tag(version) -> str:
    version_string = version if isinstance(version, str) else _entry_version(version)
    return version_string if version_string.startswith("v") else f"v{version_string}"


def _split_local_target(target: str) -> dict:
    hash_index = target.find("#")
    before_hash = target if hash_index == -1 else target[:hash_index]
    fragment = "" if hash_index == -1 else target[hash_index:]
    query_index = before_hash.find("?")

    if query_index == -1:
        return {"fragment": fragment, "pathPart": before_hash, "query": ""}

    return {
        "fragment": fragment,
        "pathPart": before_hash[:query_index],
        "query": before_hash[query_index:],
    }


def _resolve_repository_path(target_path: str) -> str | None:
    normalized_target = target_path.replace("\\", "/")
    if normalized_target.startswith("/"):
        joined = posixpath.normpath(normalized_target.lstrip("/"))
    else:
        joined = posixpath.normpath(posixpath.join(_CHANGELOG_LINK_BASE_PATH, normalized_target))

    if joined == "." or joined.startswith("../") or joined == "..":
        return None

    # Node's path.posix.normalize keeps a trailing separator, posixpath.normpath
    # drops it, and the slash reaches the emitted URL — pi's own test pins
    # `examples/extensions/` with it. Restore it.
    if normalized_target.endswith("/") and not joined.endswith("/"):
        joined += "/"

    return joined


def _is_directory_target(original_path: str, repository_path: str) -> bool:
    if original_path.endswith("/"):
        return True

    basename = posixpath.basename(repository_path)
    return "." not in basename


def _normalize_changelog_link_target(target: str, tag: str) -> str:
    canonical_target = target
    repo_url = f"https://github.com/{GITHUB_REPO}"

    for route in ("blob", "tree"):
        for branch in ("main", "master"):
            floating_ref_prefix = f"{repo_url}/{route}/{branch}/"
            if canonical_target.startswith(floating_ref_prefix):
                canonical_target = f"{repo_url}/{route}/{tag}/{canonical_target[len(floating_ref_prefix) :]}"

    if canonical_target.startswith(("#", "//")) or _URL_SCHEME_RE.match(canonical_target):
        return canonical_target

    parts = _split_local_target(canonical_target)
    if not parts["pathPart"]:
        return canonical_target

    repository_path = _resolve_repository_path(parts["pathPart"])
    if repository_path is None:
        return canonical_target

    route = "tree" if _is_directory_target(parts["pathPart"], repository_path) else "blob"
    encoded = urllib.parse.quote(repository_path, safe="/:@!$&'()*+,;=-._~?#[]")
    return f"https://github.com/{GITHUB_REPO}/{route}/{tag}/{encoded}{parts['query']}{parts['fragment']}"


def normalize_changelog_links(markdown: str, version) -> str:
    tag = _normalize_tag(version)

    def replace(match: re.Match) -> str:
        return f"{match.group(1)}{_normalize_changelog_link_target(match.group(2), tag)}{match.group(3)}"

    return _INLINE_MARKDOWN_LINK_RE.sub(replace, markdown)


async def parse_changelog(changelog_path: str) -> list:
    """Parse changelog entries from CHANGELOG.md.

    Scans for ## lines and collects content until the next ## or EOF.
    """
    if not await fs.Path(changelog_path).exists():
        return []

    try:
        content = await fs.Path(changelog_path).read_text(encoding="utf-8")
        entries: list = []

        current_lines: list = []
        current_version: dict | None = None

        for line in content.split("\n"):
            # Check if this is a version header (## [x.y.z] ...)
            if line.startswith("## "):
                # Save previous entry if exists
                if current_version is not None and current_lines:
                    entries.append({**current_version, "content": "\n".join(current_lines).strip()})

                # Try to parse version from this line
                version_match = _VERSION_HEADER_RE.search(line)
                if version_match:
                    current_version = {
                        "major": int(version_match.group(1)),
                        "minor": int(version_match.group(2)),
                        "patch": int(version_match.group(3)),
                        "build": int(version_match.group(4) or 0),
                    }
                    current_lines = [line]
                else:
                    # Reset if we can't parse version
                    current_version = None
                    current_lines = []
            elif current_version is not None:
                # Collect lines for current version
                current_lines.append(line)

        # Save last entry
        if current_version is not None and current_lines:
            entries.append({**current_version, "content": "\n".join(current_lines).strip()})

        return entries
    except Exception as error:
        print(f"Warning: Could not parse changelog: {error}", file=sys.stderr)
        return []


def compare_versions(v1: dict, v2: dict) -> int:
    """Compare versions: -1 if v1 < v2, 0 if equal, 1 if v1 > v2 (as sign)."""
    if v1["major"] != v2["major"]:
        return v1["major"] - v2["major"]
    if v1["minor"] != v2["minor"]:
        return v1["minor"] - v2["minor"]
    if v1["patch"] != v2["patch"]:
        return v1["patch"] - v2["patch"]
    return v1["build"] - v2["build"]


def get_new_entries(entries: list, last_version: str) -> list:
    """Get entries newer than last_version."""

    def part(index: int) -> int:
        pieces = last_version.split(".")
        try:
            return int(pieces[index])
        except IndexError, ValueError:
            return 0

    last = {"major": part(0), "minor": part(1), "patch": part(2), "build": part(3), "content": ""}

    return [entry for entry in entries if compare_versions(entry, last) > 0]
