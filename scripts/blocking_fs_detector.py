"""Runtime detector for filesystem calls made on a tonio runtime worker.

Why this exists
---------------
The "never block the runtime" rule (PLAN.md) is about a *transitive* property:
a sync helper doing I/O is fine until something async calls it. Static analysis
cannot see that here — two attempts at a call-graph check returned 614 and 1531
findings that were almost entirely name collisions, because resolving
`time`/`add`/`append`/`set`/`open` across packages needs types, not names.

So this checks it dynamically instead, the way the never-awaited-coroutine check
did: instrument the filesystem entry points, let the test suite drive coverage,
and report any call that happened on a runtime worker. No false positives by
construction; coverage is bounded by what the tests actually execute.

How a thread is classified
--------------------------
Runtime workers and blocking-pool threads are *both* named ``Dummy-N`` — that is
CPython's name for any thread it did not create — so the name alone cannot tell
them apart. The discriminator is a marker set by wrapping the pool entry points:

    main thread            -> allowed (pre-runtime, CLI, import time)
    marked in-pool         -> allowed (inside spawn_blocking / map_blocking)
    non-``Dummy-`` name    -> allowed (threading.Timer, fs_watch's poller, ...)
    ``Dummy-N``, unmarked  -> VIOLATION (a tonio runtime worker)

What is instrumented
--------------------
``sys.addaudithook`` covers ``open`` and the mutating/listing os calls, but the
**entire stat family is unaudited** — ``os.stat``, ``os.path.exists``,
``os.path.isdir/isfile``, ``Path.exists``, ``os.access`` raise no audit event.
That is the most common violation shape (every site fixed on 2026-07-27 was an
``os.path.exists``), so ``os.stat``/``lstat``/``access`` are additionally
patched. Patching ``os.stat`` alone is enough to catch ``os.path.exists``,
``isdir``, ``isfile``, ``Path.exists`` and ``Path.is_dir``, which all resolve it
at call time.

``tonio.colored.fs`` performs its I/O in Rust, so correct code is invisible here
and produces no findings.

Audit hooks cannot be uninstalled, so this is opt-in: set ``PIDREI_FS_DETECT=1``.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading


_local = threading.local()
_findings: dict[tuple[str, int], dict] = {}
_installed = False

# Frames in these trees are never the culprit we want to report.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PACKAGES = str(_REPO_ROOT / "packages")

# Audited events worth reporting. `open` covers reads and writes; the rest are
# the directory/mutation calls that do raise audit events.
_AUDITED = {
    "open",
    "os.listdir",
    "os.scandir",
    "os.mkdir",
    "os.rmdir",
    "os.remove",
    "os.rename",
    "os.link",
    "os.symlink",
    "os.truncate",
    "os.chmod",
    "os.utime",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "shutil.copytree",
    "shutil.move",
    "shutil.rmtree",
}


def _on_runtime_worker() -> bool:
    """True when the calling thread is a tonio runtime worker."""
    thread = threading.current_thread()
    if thread is threading.main_thread():
        return False
    if getattr(_local, "in_pool", 0):
        return False
    # CPython names threads it did not create `Dummy-N`; anything else was made
    # through `threading` and is an own-thread carve-out.
    return thread.name.startswith("Dummy-")


def _blame() -> tuple[str, int, str, bool] | None:
    """First frame inside `packages/`, preferring shipped code over tests.

    The fourth element says whether the **outermost** `packages/` frame is test
    code — i.e. whether a test drove this rather than production code. Asking
    instead whether *any* frame is shipped code is useless: a test calling a
    shipped sync helper directly always puts a shipped frame on the stack, so
    everything would look production-reached.
    """
    frame = sys._getframe(2)
    shipped = None
    test_frame = None
    outermost_is_test = False
    while frame is not None:
        filename = frame.f_code.co_filename
        if filename.startswith(_PACKAGES):
            is_test = "/tests/" in filename or "/conftest.py" in filename
            outermost_is_test = is_test
            if not is_test:
                if shipped is None:
                    shipped = (filename, frame.f_lineno, frame.f_code.co_name)
            elif test_frame is None:
                test_frame = (filename, frame.f_lineno, frame.f_code.co_name)
        frame = frame.f_back
    location = shipped or test_frame
    if location is None:
        return None
    return (*location, outermost_is_test)


def _record(what: str) -> None:
    if not _on_runtime_worker():
        return
    blamed = _blame()
    if blamed is None:
        return
    filename, lineno, func, only_tests = blamed
    key = (filename, lineno)
    entry = _findings.get(key)
    if entry is None:
        _findings[key] = {"calls": {what}, "func": func, "count": 1, "test_driven": only_tests}
    else:
        entry["calls"].add(what)
        entry["count"] += 1
        # Driven by production even once -> not a test-only artefact.
        entry["test_driven"] = entry["test_driven"] and only_tests


def _audit_hook(event: str, args: tuple) -> None:
    if event in _AUDITED:
        _record(event)


def _wrap_pool_entry(original):
    """Mark the pool thread for the duration of the offloaded callable."""

    def wrapper(fn, *args, **kwargs):
        def marked(*inner_args, **inner_kwargs):
            _local.in_pool = getattr(_local, "in_pool", 0) + 1
            try:
                return fn(*inner_args, **inner_kwargs)
            finally:
                _local.in_pool -= 1

        return original(marked, *args, **kwargs)

    return wrapper


def _patch_stat_family() -> None:
    """The stat family raises no audit events, so patch it directly."""
    for name in ("stat", "lstat", "access"):
        original = getattr(os, name)

        def make(original=original, name=name):
            def patched(*args, **kwargs):
                _record(f"os.{name}")
                return original(*args, **kwargs)

            return patched

        setattr(os, name, make())


def install() -> None:
    """Idempotent. Audit hooks cannot be removed, so this is one-way."""
    global _installed
    if _installed:
        return
    _installed = True

    import tonio.colored as tonio

    tonio.spawn_blocking = _wrap_pool_entry(tonio.spawn_blocking)
    tonio.map_blocking = _wrap_pool_entry(tonio.map_blocking)

    _patch_stat_family()
    sys.addaudithook(_audit_hook)


def findings() -> list[dict]:
    """Reported sites, worst offenders first."""
    rows = []
    for (filename, lineno), entry in _findings.items():
        rows.append(
            {
                "path": os.path.relpath(filename, _REPO_ROOT),
                "line": lineno,
                "func": entry["func"],
                "calls": sorted(entry["calls"]),
                "count": entry["count"],
                "test_driven": entry["test_driven"],
            }
        )
    rows.sort(key=lambda row: (-row["count"], row["path"], row["line"]))
    return rows


def format_report() -> str:
    rows = findings()
    if not rows:
        return "blocking-fs detector: no filesystem calls on a runtime worker"

    shipped = [row for row in rows if "/tests/" not in row["path"]]
    driven = [row for row in shipped if not row["test_driven"]]
    lines = [
        f"blocking-fs detector: {len(rows)} site(s) called the filesystem on a tonio runtime worker",
        (
            f"  shipped code: {len(shipped)} ({len(driven)} driven by production code, "
            f"{len(shipped) - len(driven)} only ever driven by a test)"
        ),
        f"  test code:    {len(rows) - len(shipped)}",
        "",
    ]
    for row in rows:
        if "/tests/" in row["path"]:
            continue
        tag = "" if not row["test_driven"] else "  (test-driven only)"
        calls = ", ".join(row["calls"])
        lines.append(f"  {row['path']}:{row['line']} in {row['func']}()  x{row['count']}  [{calls}]{tag}")
    return "\n".join(lines)
