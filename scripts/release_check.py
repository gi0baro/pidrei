"""Pre-release gates. `make release-check`, and the first step of release.yml.

Each check here exists because something already went wrong, or would have:

- **tag vs tree version** — the tree carries the next release version bare (no
  `.dev0`), which trades "remember to strip the marker" for "the machine checks
  the tag". Tags are `vX.Y.Z.N`; the `v` is stripped before comparing.
- **the five versions agree** — a skewed set ships a `pidrei` that depends on a
  `pidrei-ai==<other version>` and cannot resolve.
- **LICENSE copies** — PEP 639 forbids `..` in `license-files` and a symlink
  breaks the sdist unpack, so the file is duplicated into all five packages and
  nothing but this keeps them honest.
- **no stale `.dev`/`.post` marker** — a pre-release version reaching a tag
  means the tag and the artifact names disagree.
- **the changelog documents the version** — without a `## [<version>]` header
  the release ships, installs and runs, and the only symptom is that nobody
  ever sees a "What's New" panel for it. Silent, so it gets a gate.

The gates that need a built artifact (install outside the workspace, all five
wheels present) run in `release.yml` after the build, not here.
"""

import os
import re
import shutil
import subprocess
import sys
import tomllib


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ("ai", "agent", "server", "tui", "pidrei")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)


def package_version(name: str) -> str:
    with open(os.path.join(ROOT, "packages", name, "pyproject.toml"), "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def check_versions_agree() -> str:
    versions = {name: package_version(name) for name in PACKAGES}
    if len(set(versions.values())) != 1:
        fail(f"package versions disagree: {versions}")
    return versions["pidrei"]


def check_version_is_a_release(version: str) -> None:
    if not VERSION_RE.match(version):
        fail(
            f"version {version!r} is not a four-segment release version. "
            "A '.dev'/'.rc'/'.post' marker must not reach a tag."
        )


def check_intra_repo_pins(version: str) -> None:
    for name in PACKAGES:
        with open(os.path.join(ROOT, "packages", name, "pyproject.toml"), "rb") as handle:
            deps = tomllib.load(handle)["project"].get("dependencies", [])
        for dep in deps:
            if dep.startswith("pidrei") and not dep.endswith(f"=={version}"):
                fail(f"packages/{name}: {dep!r} is not pinned to {version}")


def check_license_copies() -> None:
    root_license = read(os.path.join(ROOT, "LICENSE"))
    for name in PACKAGES:
        path = os.path.join(ROOT, "packages", name, "LICENSE")
        if not os.path.exists(path):
            fail(f"packages/{name}/LICENSE is missing")
        elif read(path) != root_license:
            fail(f"packages/{name}/LICENSE differs from the root LICENSE")


def check_changelog_documents_version(version: str) -> None:
    """The version being released must have its own changelog entry.

    Matched the way `utils/changelog.py` parses it, so passing here means the
    running agent will actually find the entry.
    """
    path = os.path.join(ROOT, "packages", "pidrei", "pidrei", "CHANGELOG.md")
    if not os.path.exists(path):
        fail("packages/pidrei/pidrei/CHANGELOG.md is missing")
        return
    # The lookahead stops `0.82.0` from matching a `## [0.82.0.0]` header.
    header = re.compile(rf"^##\s+\[?{re.escape(version)}\]?(?![\d.])", re.MULTILINE)
    if not header.search(read(path)):
        fail(f"CHANGELOG.md has no '## [{version}]' entry for the version being released")


def check_tag_matches(version: str, tag: str | None) -> None:
    """Tags are `vX.Y.Z.N`; compare against the tree version without the `v`."""
    if not tag:
        return
    if not tag.startswith("v"):
        fail(f"tag {tag!r} does not start with 'v'")
        return
    if tag[1:] != version:
        fail(f"tag {tag!r} does not match the tree version {version!r} (expected v{version})")


def check_upstream_ref_agrees() -> None:
    package_ref = re.search(
        r'UPSTREAM_REF\s*=\s*"([0-9a-f]{40})"',
        read(os.path.join(ROOT, "packages", "pidrei", "pidrei", "upstream.py")),
    )
    if package_ref is None:
        fail("could not read UPSTREAM_REF from pidrei/upstream.py")
        return
    root_ref = read(os.path.join(ROOT, ".last_upstream_ref")).strip()
    if root_ref != package_ref.group(1):
        fail(f".last_upstream_ref ({root_ref}) does not match upstream.py ({package_ref.group(1)})")


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RELEASE_TAG")
    if tag is None:
        # Local run: use the tag pointing at HEAD, if there is one.
        result = subprocess.run(  # noqa: S603
            [shutil.which("git") or "git", "describe", "--tags", "--exact-match"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        tag = result.stdout.strip() if result.returncode == 0 else None

    version = check_versions_agree()
    check_version_is_a_release(version)
    check_intra_repo_pins(version)
    check_license_copies()
    check_upstream_ref_agrees()
    check_changelog_documents_version(version)
    check_tag_matches(version, tag)

    print(f"version {version}, tag {tag or '(none — not checked)'}")
    for failure in failures:
        print(f"  FAIL {failure}")
    print(f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
