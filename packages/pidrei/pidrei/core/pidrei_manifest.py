"""Package manifest reader (port of pi coding-agent src/core/pi-manifest.ts).

pi's manifest is the ``pi`` key of a package's ``package.json``; pidrei's is the
``[tool.pidrei]`` table of its ``pyproject.toml``. Both declare the same four
resource lists.

A malformed manifest must never take the loader down: an unreadable file,
invalid TOML, a non-table ``[tool.pidrei]``, or a field that is not a list of
strings all degrade to "declares nothing" rather than raising (pi #7187).
"""

import tomllib
from typing import Any

from ..utils.text import strip_bom


__all__ = ["MANIFEST_TABLE", "RESOURCE_FIELDS", "read_pidrei_manifest"]

MANIFEST_TABLE = ("tool", "pidrei")
RESOURCE_FIELDS = ("extensions", "skills", "prompts", "themes")


def read_pidrei_manifest(pyproject_path: str) -> dict[str, Any] | None:
    """The `[tool.pidrei]` table (pi: the `pi` key in package.json)."""
    try:
        with open(pyproject_path, encoding="utf-8") as handle:
            # pi parses package.json text and strips the BOM there; `tomllib` takes bytes
            # and rejects a leading BOM outright, so the decode happens here instead.
            document: Any = tomllib.loads(strip_bom(handle.read()))
        for key in MANIFEST_TABLE:
            document = document.get(key) if isinstance(document, dict) else None
            if not isinstance(document, dict):
                return None
    except Exception:
        return None

    manifest: dict[str, Any] = {}
    for field in RESOURCE_FIELDS:
        entries = document.get(field)
        if isinstance(entries, list) and all(isinstance(entry, str) for entry in entries):
            manifest[field] = entries
    return manifest
