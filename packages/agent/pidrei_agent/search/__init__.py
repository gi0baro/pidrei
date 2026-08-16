"""Session search contracts (port of pi `agent/src/search/index.ts`).

The scanning fallback and the option/hit dataclasses live in `scanning.py`
(pi splits them across `index.ts`/`scanning.ts` with a type-only import cycle
Python cannot express); this module re-exports pi's public surface.
"""

from .scanning import (
    ScanningSessionSearchHit,
    SessionSearchCandidate,
    SessionSearchHit,
    SessionSearchOptions,
    create_scanning_session_search,
    scanning_entries,
)


__all__ = [
    "ScanningSessionSearchHit",
    "SessionSearchCandidate",
    "SessionSearchHit",
    "SessionSearchOptions",
    "create_scanning_session_search",
    "scanning_entries",
]
