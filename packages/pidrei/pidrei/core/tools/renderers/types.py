"""The renderer record a tool definition contributes to the TUI.

pi moved `ToolRenderers` to `core/extensions/types.ts` and re-exports it from
`renderers/index.ts` and `tool-execution.ts`; this module re-exports it for
the per-tool renderer modules, so they and the package `__init__` share it
without a cycle.

`detail_field` reads a field of a tool result's `details`, which is the tool's
dataclass while the result is live and pi's camelCase dict once the session
is read back from its file (`message_wire.to_wire_value`).
"""

from dataclasses import fields
from typing import Any

from ...extensions.types import ToolRenderers
from ..truncate import TruncationResult


__all__ = ["ToolRenderers", "detail_field", "detail_truncation"]


def detail_field(details: Any, name: str) -> Any:
    """`details.<name>` for a live dataclass, `details[<camelName>]` for a resumed dict, None when absent."""
    if details is None:
        return None
    if isinstance(details, dict):
        head, *rest = name.split("_")
        return details.get(head + "".join(part.capitalize() for part in rest))
    return getattr(details, name, None)


def detail_truncation(details: Any) -> TruncationResult | None:
    """`details.truncation` as a `TruncationResult`, rebuilt from a resumed dict
    (fields the wire dropped as None come back as None)."""
    truncation = detail_field(details, "truncation")
    if isinstance(truncation, dict):
        return TruncationResult(
            **{field.name: detail_field(truncation, field.name) for field in fields(TruncationResult)}
        )
    return truncation
