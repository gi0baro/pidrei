"""Tool-definition subset of pi coding-agent src/core/extensions/types.ts.

Only the pieces the tool layer and AgentSession registry need are here; the
extension system itself (Extension, ExtensionAPI, the ~35-event hook bus,
runners, renderer contexts) lands in Phase 5. The TUI render hooks
(renderCall/renderResult/renderShell) are Phase 4 and are represented only by
the optional `render_shell` marker.
"""

from dataclasses import dataclass, field
from typing import Any


class ExtensionContext:
    """Placeholder for pi's ExtensionContext (Phase 5).

    Tools only duck-read optional attributes from it (session_manager, model,
    thinking_level, ...); a plain attribute bag keeps the seam alive.
    """

    def __init__(self, **attributes: Any):
        for name, value in attributes.items():
            setattr(self, name, value)

    def __getattr__(self, _name: str) -> Any:
        return None


@dataclass(slots=True, kw_only=True)
class ToolDefinition:
    """Definition-first tool record (pi's ToolDefinition<TParams, TDetails>)."""

    # Tool name (used in LLM tool calls)
    name: str
    # Human-readable label for UI
    label: str
    # Description for LLM
    description: str
    # Parameter schema (JSON Schema; pi: TypeBox)
    parameters: dict[str, Any]
    # Execute the tool: async (tool_call_id, params, cancel, on_update, ctx) -> AgentToolResult
    execute: Any
    # Optional one-line snippet for the Available tools section in the default system prompt.
    prompt_snippet: str | None = None
    # Optional guideline bullets appended to the default system prompt Guidelines section.
    prompt_guidelines: list[str] | None = None
    # Optional provider-side constrained sampling request for this tool.
    constrained_sampling: Any = None
    # TUI shell rendering marker ("default" | "self"); renderers themselves are Phase 4.
    render_shell: str | None = None
    # Optional compatibility shim to prepare raw tool call arguments before schema validation.
    prepare_arguments: Any = None
    # Per-tool execution mode override ("sequential" | "parallel").
    execution_mode: str | None = None
    # Extra metadata slot mirroring pi's open object shape.
    extra: dict[str, Any] = field(default_factory=dict)
