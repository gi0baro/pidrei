"""The renderer record a tool definition contributes to the TUI.

pi moved `ToolRenderers` to `core/extensions/types.ts` and re-exports it from
`renderers/index.ts` and `tool-execution.ts`; this module re-exports it for
the per-tool renderer modules, so they and the package `__init__` share it
without a cycle.
"""

from ...extensions.types import ToolRenderers


__all__ = ["ToolRenderers"]
