"""Mirror of pi coding-agent src/core/timings.ts (PI_TIMING -> PIDREI_TIMING)."""

import os
import sys
import time as time_module
from typing import Any


_ENABLED = os.environ.get("PIDREI_TIMING") == "1"

_timing_namespaces: dict[str, dict[str, Any]] = {}


def reset_timings(namespace: str = "main") -> None:
    if not _ENABLED:
        return
    _timing_namespaces[namespace] = {"timings": [], "last_time": time_module.time() * 1000}


def time(label: str, namespace: str = "main") -> None:
    if not _ENABLED:
        return
    now = time_module.time() * 1000

    if namespace not in _timing_namespaces:
        reset_timings(namespace)

    timing_namespace = _timing_namespaces[namespace]
    timing_namespace["timings"].append({"label": label, "ms": now - timing_namespace["last_time"]})
    timing_namespace["last_time"] = now


def _print_timing_group(title: str, timings: list[dict[str, Any]]) -> None:
    printable = [timing for timing in timings if timing["ms"] >= 0]
    if not printable:
        return
    print(f"\n--- {title} ---", file=sys.stderr)
    for timing in printable:
        print(f"  {timing['label']}: {round(timing['ms'])}ms", file=sys.stderr)
    total = sum(timing["ms"] for timing in printable)
    print(f"  TOTAL: {round(total)}ms", file=sys.stderr)
    print("-" * (len(title) + 8) + "\n", file=sys.stderr)


def print_timings() -> None:
    if not _ENABLED:
        return
    for namespace, timing_namespace in _timing_namespaces.items():
        _print_timing_group(f"Startup Timings: {namespace}", timing_namespace["timings"])
