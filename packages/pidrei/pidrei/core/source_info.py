"""Mirror of pi coding-agent src/core/source-info.ts."""

from dataclasses import dataclass


type SourceScope = str  # "user" | "project" | "temporary"
type SourceOrigin = str  # "package" | "top-level"


@dataclass(slots=True)
class PathMetadata:
    """Provenance of a resolved resource path (from pi's package-manager.ts)."""

    source: str
    scope: SourceScope
    origin: SourceOrigin = "top-level"
    base_dir: str | None = None


@dataclass(slots=True)
class SourceInfo:
    path: str
    source: str
    scope: SourceScope
    origin: SourceOrigin
    base_dir: str | None = None


def create_source_info(path: str, metadata: PathMetadata) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=metadata.source,
        scope=metadata.scope,
        origin=metadata.origin,
        base_dir=metadata.base_dir,
    )


def create_synthetic_source_info(
    path: str,
    *,
    source: str,
    scope: SourceScope | None = None,
    origin: SourceOrigin | None = None,
    base_dir: str | None = None,
) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=source,
        scope=scope if scope is not None else "temporary",
        origin=origin if origin is not None else "top-level",
        base_dir=base_dir,
    )
