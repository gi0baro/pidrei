"""The version scheme is self-enforcing.

pidrei is versioned `<pi version>.<our build>` — `0.82.0.0` is the first pidrei
build tracking pi 0.82.0. That only means anything if three facts stay in sync,
and all three live in different files that nothing otherwise ties together:

- five `pyproject.toml` versions (a release that skews them ships a `pidrei`
  depending on a differently-versioned `pidrei-ai`),
- `pidrei/upstream.py`'s `UPSTREAM_VERSION`, which the first three segments must
  equal — so the pi ref and the version cannot be bumped independently,
- the repo-root `.last_upstream_ref`, which tooling reads without importing
  pidrei and which would otherwise drift silently.

pidrei-only: pi has one version in one file and no upstream to track, so there
is nothing here to mirror.
"""

import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib

import pytest

from pidrei.config import VERSION
from pidrei.upstream import UPSTREAM_REF, UPSTREAM_VERSION, short_ref


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PACKAGES = ("ai", "agent", "server", "tui", "pidrei")

#: `0.82.0.0`, or with a PEP 440 pre-release suffix. Four release segments.
VERSION_RE = re.compile(r"^(\d+\.\d+\.\d+)\.(\d+)((?:a|b|rc|\.post|\.dev)\d+)?$")


def package_version(name: str) -> str:
    with open(os.path.join(REPO_ROOT, "packages", name, "pyproject.toml"), "rb") as handle:
        return tomllib.load(handle)["project"]["version"]


def test_installed_version_matches_the_scheme():
    assert VERSION_RE.match(VERSION), f"{VERSION!r} is not <pi version>.<build>[pre]"


def test_all_five_packages_share_one_version():
    versions = {name: package_version(name) for name in PACKAGES}
    assert len(set(versions.values())) == 1, f"packages disagree: {versions}"


def test_the_first_three_segments_are_the_ported_pi_version():
    """The point of the scheme: the number names the pi release it tracks."""
    match = VERSION_RE.match(package_version("pidrei"))
    assert match is not None
    assert match.group(1) == UPSTREAM_VERSION


def test_intra_repo_dependencies_pin_the_shared_version():
    """The five ship as a set. `pidrei-ai` unpinned would let a user mix wheels
    from two releases and get a silently mismatched install; pinned, it fails
    loudly instead. The cost is that a version bump must touch the pins too —
    which is what this asserts."""
    version = package_version("pidrei")
    for name in PACKAGES:
        with open(os.path.join(REPO_ROOT, "packages", name, "pyproject.toml"), "rb") as handle:
            deps = tomllib.load(handle)["project"].get("dependencies", [])
        for dep in deps:
            if dep.startswith("pidrei"):
                assert dep.endswith(f"=={version}"), f"packages/{name}: {dep!r} is not pinned to {version}"


def test_third_party_dependencies_have_upper_bounds():
    """A release installed months later must not silently pull a new major.
    Four deps had already drifted across majors past their floors before this
    was enforced (pathspec 0.12 -> 1.1, cryptography 46 -> 49, pillow 11 -> 12)."""
    unbounded = []
    for name in PACKAGES:
        with open(os.path.join(REPO_ROOT, "packages", name, "pyproject.toml"), "rb") as handle:
            deps = tomllib.load(handle)["project"].get("dependencies", [])
        for dep in deps:
            if dep.startswith("pidrei"):
                continue
            if "~=" not in dep and "<" not in dep:
                unbounded.append(f"packages/{name}: {dep}")
    assert not unbounded, "dependencies with no upper bound: " + ", ".join(unbounded)


def test_module_level_versions_are_not_stale_literals():
    """`pidrei_ai.__version__` was a hardcoded `0.1.0.dev0` that survived two
    version bumps, because nothing imported it and nothing checked it. Any
    `__version__` the packages expose must agree with the distribution."""
    import pidrei_ai

    assert pidrei_ai.__version__ == package_version("ai")


def test_no_package_hardcodes_a_version_literal():
    """The version lives in `pyproject.toml` and nowhere else in source.

    An assignment of a string literal to `__version__` is the shape that went
    stale; a derived one (`importlib.metadata`) is fine, so this looks for the
    assignment specifically rather than the identifier.
    """
    offenders = []
    for name in PACKAGES:
        root = os.path.join(REPO_ROOT, "packages", name)
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in ("tests", "__pycache__")]
            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                path = os.path.join(dirpath, filename)
                with open(path, encoding="utf-8") as handle:
                    tree = ast.parse(handle.read())
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                        continue
                    if not isinstance(node.value.value, str):
                        continue
                    if any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets):
                        offenders.append(f"{os.path.relpath(path, REPO_ROOT)}:{node.lineno}")
    assert not offenders, "hardcoded version literals: " + ", ".join(offenders)


def test_the_root_upstream_ref_file_agrees_with_the_package():
    """The file is not shipped, so only this keeps it honest."""
    with open(os.path.join(REPO_ROOT, ".last_upstream_ref"), encoding="utf-8") as handle:
        assert handle.read().strip() == UPSTREAM_REF


def test_upstream_ref_is_a_full_sha():
    assert re.fullmatch(r"[0-9a-f]{40}", UPSTREAM_REF)
    assert short_ref() == UPSTREAM_REF[:8]


def test_the_ported_pi_version_is_what_pi_called_itself_at_that_ref():
    """Guards the one mistake this scheme cannot survive: naming the wrong pi
    release.

    Needs a pi checkout, which is not vendored (open decision 12), so it is
    opt-in through `PIDREI_UPSTREAM_CHECKOUT` rather than defaulting to some
    developer's local path. `make upstream-diff` will want the same variable.
    """
    pi_repo = os.environ.get("PIDREI_UPSTREAM_CHECKOUT")
    if not pi_repo or not os.path.isdir(pi_repo):
        pytest.skip("set PIDREI_UPSTREAM_CHECKOUT to a pi checkout to run this")
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not on PATH")

    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [git, "show", f"{UPSTREAM_REF}:packages/coding-agent/package.json"],
        cwd=pi_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("pinned ref not present in the local pi checkout")

    assert json.loads(result.stdout)["version"] == UPSTREAM_VERSION


def test_version_flag_prints_only_the_version():
    """Whatever parses `--version` gets a number and nothing else."""
    result = subprocess.run(
        [sys.executable, "-m", "pidrei", "--version"],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == VERSION
