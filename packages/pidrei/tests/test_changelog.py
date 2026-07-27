"""Changelog parsing and link rewriting.

pi's changelog.test.ts covers only `normalizeChangelogLinks`; those two cases
are mirrored here with our repository and base path substituted, since the
module exists to rewrite links in *our* changelog.

The rest is pidrei-only and covers the fourth version segment. pidrei versions
are `<pi version>.<our build>`, so the segment that moves between pi releases is
the one pi's parser does not read — a three-segment parse would make every
pidrei-only release compare equal to the previous one, and the "What's New"
panel would silently never appear.
"""

import os

import pytest

from pidrei.config import get_changelog_path
from pidrei.utils.changelog import (
    compare_versions,
    get_new_entries,
    normalize_changelog_links,
    parse_changelog,
)


ENTRY = {"major": 0, "minor": 82, "patch": 0, "build": 0, "content": ""}
BLOB = "https://github.com/gi0baro/pidrei/blob/v0.82.0.0"
TREE = "https://github.com/gi0baro/pidrei/tree/v0.82.0.0"


def write_changelog(tmp_dir, text: str) -> str:
    path = os.path.join(tmp_dir, "CHANGELOG.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


# ---------------------------------------------------------------------------
# normalize_changelog_links — pi's cases, our repo
# ---------------------------------------------------------------------------


def test_rewrites_package_relative_links_to_tag_pinned_source_links():
    markdown = (
        "[Project Trust](README.md#project-trust)\n"
        "[Extensions](docs/extensions.md#project_trust)\n"
        "[Examples](examples/extensions/)\n"
        "[Root README](../../../README.md#status)"
    )

    assert normalize_changelog_links(markdown, ENTRY) == (
        f"[Project Trust]({BLOB}/packages/pidrei/pidrei/README.md#project-trust)\n"
        f"[Extensions]({BLOB}/packages/pidrei/pidrei/docs/extensions.md#project_trust)\n"
        f"[Examples]({TREE}/packages/pidrei/pidrei/examples/extensions/)\n"
        f"[Root README]({BLOB}/README.md#status)"
    )


def test_leaves_external_links_and_anchors_alone():
    markdown = "[External](https://example.com/docs)\n[Local anchor](#settings)\n[Protocol relative](//example.com/x)"

    assert normalize_changelog_links(markdown, "0.82.0.0") == markdown


def test_pi_links_survive_untouched():
    """The divergence from pi's module: it canonicalized `*/pi-mono` URLs into
    its own repo. Inherited unchanged, that rule would rewrite a real pi link
    into a pidrei URL that does not exist."""
    markdown = (
        "[Pi 0.82.0](https://github.com/earendil-works/pi/releases/tag/v0.82.0)\n"
        "[#5167](https://github.com/earendil-works/pi-mono/pull/5167)\n"
        "[Agent README](https://github.com/badlogic/pi-mono/blob/main/packages/agent/README.md)"
    )

    assert normalize_changelog_links(markdown, ENTRY) == markdown


def test_our_own_floating_branch_links_are_pinned_to_the_tag():
    markdown = "[Script](https://github.com/gi0baro/pidrei/blob/main/scripts/release_check.py)"

    assert normalize_changelog_links(markdown, ENTRY) == f"[Script]({BLOB}/scripts/release_check.py)"


# ---------------------------------------------------------------------------
# parse_changelog
# ---------------------------------------------------------------------------


@pytest.mark.tonio
async def test_parses_four_segment_headers(tmp_dir):
    path = write_changelog(
        tmp_dir,
        "# Changelog\n\n## [0.82.0.0] - 2026-07-27\n\nFirst.\n\n## [0.82.0.1] - 2026-08-01\n\nSecond.\n",
    )

    entries = await parse_changelog(path)

    assert [(e["major"], e["minor"], e["patch"], e["build"]) for e in entries] == [
        (0, 82, 0, 0),
        (0, 82, 0, 1),
    ]
    assert entries[1]["content"] == "## [0.82.0.1] - 2026-08-01\n\nSecond."


@pytest.mark.tonio
async def test_a_three_segment_header_is_build_zero(tmp_dir):
    path = write_changelog(tmp_dir, "## [0.82.0] - 2026-07-27\n\nBody.\n")

    assert (await parse_changelog(path))[0]["build"] == 0


@pytest.mark.tonio
async def test_headers_without_a_version_are_skipped(tmp_dir):
    """`## [Unreleased]` is the convention and must not become an entry."""
    path = write_changelog(
        tmp_dir,
        "# Changelog\n\n## [Unreleased]\n\n## [0.82.0.0] - 2026-07-27\n\nBody.\n",
    )

    entries = await parse_changelog(path)

    assert len(entries) == 1
    assert entries[0]["content"] == "## [0.82.0.0] - 2026-07-27\n\nBody."


@pytest.mark.tonio
async def test_a_missing_changelog_is_not_an_error(tmp_dir):
    assert await parse_changelog(os.path.join(tmp_dir, "nope.md")) == []


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def test_build_segment_breaks_the_tie():
    assert compare_versions({**ENTRY, "build": 1}, ENTRY) > 0
    assert compare_versions(ENTRY, {**ENTRY, "build": 1}) < 0
    assert compare_versions(ENTRY, dict(ENTRY)) == 0


@pytest.mark.tonio
async def test_new_entries_include_a_build_only_bump(tmp_dir):
    """The reason the fourth segment is parsed at all: between pi releases this
    is the only thing that changes, and it decides whether the user is shown
    what they just updated to."""
    path = write_changelog(
        tmp_dir,
        "## [0.82.0.0] - 2026-07-27\n\nFirst.\n\n## [0.82.0.1] - 2026-08-01\n\nSecond.\n",
    )
    entries = await parse_changelog(path)

    new = get_new_entries(entries, "0.82.0.0")

    assert len(new) == 1
    assert new[0]["build"] == 1
    assert get_new_entries(entries, "0.82.0.1") == []


@pytest.mark.tonio
async def test_a_last_version_without_a_build_segment_reads_as_zero(tmp_dir):
    """Settings written before the fourth segment was parsed, or by hand."""
    path = write_changelog(tmp_dir, "## [0.82.0.1] - 2026-08-01\n\nBody.\n")

    assert len(get_new_entries(await parse_changelog(path), "0.82.0")) == 1


# ---------------------------------------------------------------------------
# the shipped file
# ---------------------------------------------------------------------------


@pytest.mark.tonio
async def test_the_shipped_changelog_parses():
    """It is shipped inside the package and read at startup; a header shape the
    parser does not recognise fails silently and shows nothing.

    That the entry matches the version being released is a release gate
    (`scripts/release_check.py`), not a test — the tree carries the next
    version, so between the bump and writing its entry this would fail on work
    in progress.
    """
    entries = await parse_changelog(get_changelog_path())

    assert entries, "no parseable entries in the shipped CHANGELOG.md"
    assert all(entry["content"].startswith("## ") for entry in entries)
