"""The renderer pair a tool definition contributes to the TUI.

pi types this twice — `Pick<ToolDefinition, "renderCall" | "renderResult">`
in `renderers/index.ts` and the `ToolRenderers` interface (plus `renderShell`)
in `tool-execution.ts`; both are structural, so one record serves here. It
lives in its own module so the per-tool renderer modules and the package
`__init__` share it without a cycle.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolRenderers:
    """What ToolExecutionComponent needs from a tool: how to draw it.

    A `ToolDefinition` satisfies the same three attributes, so a definition
    and a bare renderer pair are equally acceptable to the component.
    """

    render_call: Any = None
    render_result: Any = None
    render_shell: str | None = None
