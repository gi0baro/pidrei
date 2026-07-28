"""Print the GitHub release notes for a version, from the shipped changelog.

    uv run python scripts/release_notes.py <version-or-tag>

The body is the `CHANGELOG.md` entry for the version, parsed with the same
`utils/changelog.py` the running agent uses for its What's New panel, with
relative links normalized to absolute GitHub URLs pinned to the release tag —
relative links do not survive the releases page. The `## [version]` header
line is dropped: the release title already names the version. A missing entry
is an error here too; `release_check.py` gates it earlier, this is the belt
to those braces.
"""

import os
import sys

import tonio.colored as tonio

from pidrei.utils.changelog import normalize_changelog_links, parse_changelog


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHANGELOG = os.path.join(ROOT, "packages", "pidrei", "pidrei", "CHANGELOG.md")


async def _notes(version: str) -> str | None:
    try:
        numbers = [int(part) for part in version.split(".")]
    except ValueError:
        return None
    if len(numbers) != 4:
        return None
    for entry in await parse_changelog(CHANGELOG):
        if [entry["major"], entry["minor"], entry["patch"], entry["build"]] == numbers:
            split = entry["content"].split("\n", 1)
            body = split[1].strip() if len(split) > 1 else ""
            return normalize_changelog_links(body, entry)
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    version = sys.argv[1].removeprefix("v")
    notes = tonio.run(_notes(version))
    if not notes:
        print(f"CHANGELOG.md has no usable entry for {version!r}", file=sys.stderr)
        return 1
    print(notes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
