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


# Prefix of built-in tool and extension paths, such as `builtin:read` or `builtin:mcp`.
BUILTIN_PATH_PREFIX = "builtin:"


def get_synthetic_path_source(path: str) -> str | None:
    """Source of a path that names no file: `builtin` for `builtin:<name>`, or the prefix of an
    angle-bracket path such as `inline` for `<inline:name>`. None for file paths."""
    if path.startswith(BUILTIN_PATH_PREFIX):
        return "builtin"
    if path.startswith("<") and path.endswith(">"):
        return path[1:-1].split(":")[0] or "temporary"
    return None


def is_synthetic_path(path: str) -> bool:
    return path.startswith((BUILTIN_PATH_PREFIX, "<"))


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
