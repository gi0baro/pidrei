"""Upstream sync report: what changed in pi since the last ported commit.

`make upstream-diff` walks `$(cat .last_upstream_ref)..HEAD` in the pi checkout
(`PI_ROOT`) and prints a commit-by-commit porting checklist, oldest first —
upstream commits are the porting unit, and new pi tests are the spec by
construction.

Every changed file is classified:

- **src / test / doc** — maps to a pidrei file through PREFIX_MAP (kebab-case
  → snake_case, `.ts` → `.py`; pi's nested test dirs flatten into our flat
  `tests/` layout). Ports whose pidrei name diverges from the mechanical
  mapping live in RENAMES, one hand-verified entry per divergence.
- **dropped** — documented divergences (the radius provider, the llama.cpp
  extension, pi's evals and storage packages). Surfaced as a one-liner, no
  port needed.
- **noise** — pi-internal machinery: lockfiles, changelogs, CI, vitest
  configs, examples. Top-level `package.json` changes in ported packages are
  additionally summarized at the end: new runtime deps need a manual look.
- **UNMAPPED** — anything else. Loud, and the exit code is 2: either a new
  upstream file class (extend the tables) or a new pi package (decide port vs
  drop, and document the decision here).

A mapped target that does not exist is `[NEW]` for files pi added; for files
pi *modified* it usually means pidrei renamed the module — verify and extend
RENAMES instead of porting to the mechanical name.

`--bump <sha>` records progress once everything up to `<sha>` is ported:
verifies the sha sits on the ported-ref → HEAD line, then writes both
`.last_upstream_ref` and `upstream.py`'s UPSTREAM_REF. When the sha crosses a
pi release tag it warns that UPSTREAM_VERSION and the five package versions
must move together (release_check gates that).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_FILE = os.path.join(ROOT, ".last_upstream_ref")
UPSTREAM_PY = os.path.join(ROOT, "packages", "pidrei", "pidrei", "upstream.py")

#: pi path prefix -> (pidrei path prefix, kind). Kind drives the name
#: transform: src/doc keep pi's directory shape, test flattens.
PREFIX_MAP = (
    ("packages/ai/src/", "packages/ai/pidrei_ai/", "src"),
    ("packages/ai/scripts/", "packages/ai/scripts/", "src"),
    ("packages/ai/test/", "packages/ai/tests/", "test"),
    ("packages/agent/src/", "packages/agent/pidrei_agent/", "src"),
    ("packages/agent/test/", "packages/agent/tests/", "test"),
    ("packages/coding-agent/src/", "packages/pidrei/pidrei/", "src"),
    ("packages/coding-agent/test/", "packages/pidrei/tests/", "test"),
    ("packages/coding-agent/docs/", "packages/pidrei/pidrei/docs/", "doc"),
    ("packages/tui/src/", "packages/tui/pidrei_tui/", "src"),
    ("packages/tui/test/", "packages/tui/tests/", "test"),
    ("packages/server/src/", "packages/server/pidrei_server/", "src"),
    ("packages/server/test/", "packages/server/tests/", "test"),
)

#: pi paths pidrei deliberately does not port, with the recorded reason.
DROPPED_PREFIXES = (
    ("packages/evals/", "pi-internal eval harness, not ported"),
    ("packages/storage/", "storage backend, not ported"),
    (
        "packages/coding-agent/src/extensions/llama/",
        "llama.cpp extension not ported (see pidrei/extensions/__init__.py)",
    ),
    (
        "packages/coding-agent/test/llama-extension.test.ts",
        "llama.cpp extension not ported (see pidrei/extensions/__init__.py)",
    ),
    (
        "packages/ai/test/xhigh.test.ts",
        "live-API test (skipIf !OPENAI_API_KEY), not ported",
    ),
    (
        "packages/ai/test/openai-responses-reasoning-replay-e2e.test.ts",
        "live-API test, not ported (offline mirror: test_azure_openai_responses_reasoning_replay.py)",
    ),
)
#: The radius provider (pi's own gateway) is the documented provider drop;
#: its files carry "radius" in the basename wherever they sit.
DROPPED_BASENAME_RE = re.compile(r"radius")
DROPPED_BASENAME_REASON = "radius provider dropped (pi-specific gateway; FEASIBILITY.md)"

#: pi path -> pidrei path where the port's name diverges from the mechanical
#: mapping. Hand-verified; extend when the report flags a modified file whose
#: mechanical target is missing.
RENAMES = {
    "packages/coding-agent/src/core/remote-catalog-provider.ts": "packages/pidrei/pidrei/core/remote_catalog.py",
}

#: pi test files whose pidrei coverage is not a 1:1 mirror. Phase-1 `ai` tests
#: are organized by pidrei module, so several pi files map many-to-many; the
#: note names where the changed cases go. Entries marked PARITY GAP have a
#: ported production module but no mirrored tests — backfill when a commit
#: touches them.
TEST_HOMES = {
    "packages/ai/test/env-api-keys.test.ts": "covered by packages/ai/tests/test_providers.py (+ test_registry.py; see its docstring)",
    "packages/ai/test/supports-xhigh.test.ts": "covered by packages/ai/tests/test_registry.py + test_models_generated.py (get_supported_thinking_levels)",
    "packages/ai/test/models-runtime.test.ts": "covered by packages/ai/tests/test_registry.py (models.ts ported as registry.py)",
    "packages/ai/test/error-body.test.ts": "PARITY GAP: error_body.py ported, its 12 cases never mirrored — backfill here",
    "packages/ai/test/provider-error-body-regression.test.ts": "PARITY GAP: 4 cases never mirrored — backfill",
}

NOISE_BASENAMES = {
    "package.json",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "CHANGELOG.md",
    "tsconfig.json",
    "vitest.config.ts",
    "vitest.base.ts",
    ".npmignore",
}
NOISE_PREFIXES = (
    ".github/",
    "scripts/",
    "packages/coding-agent/examples/",
    "packages/coding-agent/install-lock/",
)

#: Top-level manifests of ported packages: still noise for porting purposes,
#: but a new runtime dependency there needs a human decision, so they get
#: their own summary section.
DEPS_REVIEW_PATHS = {f"packages/{name}/package.json" for name in ("ai", "agent", "coding-agent", "tui", "server")}


GIT = shutil.which("git") or "git"


def git(pi_root: str, *args: str) -> str:
    result = subprocess.run(  # noqa: S603
        [GIT, "-C", pi_root, *args], check=True, capture_output=True, text=True
    )
    return result.stdout


def is_ancestor(pi_root: str, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(  # noqa: S603
        [GIT, "-C", pi_root, "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def read(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def snake(segment: str) -> str:
    return segment.replace("-", "_")


def map_path(pi_path: str) -> tuple[str, str] | None:
    """Return (kind, pidrei_path) for a portable pi file, else None."""
    if pi_path in RENAMES:
        return "src", RENAMES[pi_path]
    for prefix, target, kind in PREFIX_MAP:
        if not pi_path.startswith(prefix):
            continue
        rest = pi_path[len(prefix) :]
        if kind == "doc":
            return kind, target + rest
        if kind == "test":
            # pidrei test trees are flat; pi nests (test/suite/, regressions/).
            base = rest.rsplit("/", 1)[-1]
            if base.endswith(".test.ts"):
                return kind, target + "test_" + snake(base[: -len(".test.ts")]) + ".py"
            if base.endswith(".ts"):
                return kind, target + snake(base[: -len(".ts")]) + ".py"
            return kind, target + base
        parts = [snake(part) for part in rest.split("/")]
        if parts[-1].endswith(".ts"):
            parts[-1] = parts[-1][: -len(".ts")] + ".py"
        return kind, target + "/".join(parts)
    return None


def classify(pi_path: str) -> tuple[str, object]:
    """Return (category, detail): portable -> (kind, target), else a reason."""
    for prefix, reason in DROPPED_PREFIXES:
        if pi_path.startswith(prefix):
            return "dropped", reason
    basename = pi_path.rsplit("/", 1)[-1]
    if DROPPED_BASENAME_RE.search(basename):
        return "dropped", DROPPED_BASENAME_REASON
    if basename in NOISE_BASENAMES or pi_path.startswith(NOISE_PREFIXES):
        return "noise", None
    if "/" not in pi_path and pi_path.endswith(".md"):
        return "noise", None
    mapped = map_path(pi_path)
    if mapped is not None:
        return "portable", mapped
    return "unmapped", None


def commit_files(pi_root: str, sha: str) -> list[tuple[str, str]]:
    """[(status, path)] for a commit; renames/copies report the new path."""
    out = git(pi_root, "diff-tree", "-r", "-m", "--first-parent", "--no-commit-id", "--name-status", sha)
    files = []
    for line in out.splitlines():
        fields = line.split("\t")
        status = fields[0][0]
        files.append((status, fields[-1]))
    return files


def marker(status: str, target: str) -> str:
    exists = os.path.exists(os.path.join(ROOT, target))
    if status == "D":
        return "  [DELETE]" if exists else "  [already absent]"
    if exists:
        return ""
    if status == "A":
        return "  [NEW]"
    return "  [MISSING — pidrei rename? verify and extend RENAMES]"


def report(pi_root: str) -> int:
    ported_ref = read(REF_FILE).strip()
    head = git(pi_root, "rev-parse", "HEAD").strip()
    describe = git(pi_root, "describe", "--tags", head).strip()
    print(f"pi checkout : {pi_root} @ {head[:8]} ({describe})")
    print(f"ported ref  : {ported_ref[:8]} ({git(pi_root, 'describe', '--tags', ported_ref).strip()})")

    log = git(pi_root, "log", "--reverse", "--format=%H\t%s", f"{ported_ref}..HEAD")
    commits = [line.split("\t", 1) for line in log.splitlines()]
    print(f"{len(commits)} upstream commits\n")

    to_port = 0
    files_to_port = 0
    new_files = 0
    missing_on_modify: list[str] = []
    unmapped: list[str] = []
    deps_review: dict[str, list[str]] = {}

    for index, (sha, subject) in enumerate(commits, start=1):
        portable: list[tuple[str, str, str, str]] = []
        dropped_reasons: list[str] = []
        noise = 0
        for status, path in commit_files(pi_root, sha):
            category, detail = classify(path)
            if category == "portable":
                kind, target = detail
                portable.append((status, kind, path, target))
            elif category == "dropped":
                if detail not in dropped_reasons:
                    dropped_reasons.append(detail)
            elif category == "noise":
                noise += 1
                if path in DEPS_REVIEW_PATHS:
                    deps_review.setdefault(path, []).append(sha[:8])
            else:
                unmapped.append(f"{path}  ({sha[:8]} {subject})")

        head_line = f"[{index:>2}/{len(commits)}] {sha[:8]}  {subject}"
        if not portable:
            why = "; ".join(dropped_reasons) if dropped_reasons else "noise only"
            print(f"{head_line} — nothing to port ({why})")
            continue

        to_port += 1
        print(head_line)
        for status, kind, path, target in portable:
            files_to_port += 1
            print(f"    {status} {kind:<4} {path}")
            note = TEST_HOMES.get(path)
            if note is not None:
                print(f"             → {note}")
                continue
            mark = marker(status, target)
            print(f"             → {target}{mark}")
            if mark == "  [NEW]":
                new_files += 1
            elif mark.startswith("  [MISSING"):
                missing_on_modify.append(f"{path} → {target}")
        for reason in dropped_reasons:
            print(f"      (also touches dropped surface: {reason})")

    print("\n== summary ==")
    print(
        f"{to_port} commits to port ({files_to_port} files, {new_files} new), "
        f"{len(commits) - to_port} with nothing to port"
    )
    if deps_review:
        print("package.json changed in ported packages — review for new runtime deps:")
        for path, shas in deps_review.items():
            print(f"  {path}  ({', '.join(shas)})")
    if missing_on_modify:
        print("modified upstream, but the mechanical pidrei target is missing — verify rename vs new file:")
        for entry in missing_on_modify:
            print(f"  {entry}")
    if unmapped:
        print("UNMAPPED — extend PREFIX_MAP / DROPPED / NOISE tables:")
        for entry in unmapped:
            print(f"  {entry}")
        return 2
    print("port in upstream order; after each commit lands:")
    print("  make upstream-bump REF=<sha>")
    return 0


def bump(pi_root: str, ref: str) -> int:
    sha = git(pi_root, "rev-parse", "--verify", f"{ref}^{{commit}}").strip()
    ported_ref = read(REF_FILE).strip()
    if not is_ancestor(pi_root, ported_ref, sha):
        print(f"refusing: {sha[:8]} is not a descendant of the ported ref {ported_ref[:8]}")
        return 1
    if not is_ancestor(pi_root, sha, "HEAD"):
        print(f"refusing: {sha[:8]} is not an ancestor of the pi checkout's HEAD")
        return 1

    with open(REF_FILE, "w", encoding="utf-8") as handle:
        handle.write(sha + "\n")
    source = read(UPSTREAM_PY)
    updated, count = re.subn(r'UPSTREAM_REF = "[0-9a-f]{40}"', f'UPSTREAM_REF = "{sha}"', source)
    if count != 1:
        print(f"could not find UPSTREAM_REF assignment in {UPSTREAM_PY}")
        return 1
    with open(UPSTREAM_PY, "w", encoding="utf-8") as handle:
        handle.write(updated)
    print(f"bumped .last_upstream_ref and upstream.py to {sha[:8]}")

    release = git(pi_root, "describe", "--tags", "--abbrev=0", sha).strip().lstrip("v")
    version = re.search(r'UPSTREAM_VERSION = "([^"]+)"', updated).group(1)
    if release != version:
        print(
            f"NOTE: pi released v{release} at or before this commit, but "
            f"UPSTREAM_VERSION is still {version} — bump it and the five "
            f"package versions together (release_check gates this)."
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi-root", default=os.environ.get("PI_ROOT"), help="path to the pi checkout (or set PI_ROOT)")
    parser.add_argument("--bump", metavar="SHA", help="record everything up to SHA as ported")
    args = parser.parse_args()
    if not args.pi_root:
        parser.error("pass --pi-root or set PI_ROOT")
    if not os.path.isdir(os.path.join(args.pi_root, ".git")):
        parser.error(f"{args.pi_root} is not a git checkout")
    if args.bump:
        return bump(args.pi_root, args.bump)
    return report(args.pi_root)


if __name__ == "__main__":
    sys.exit(main())
