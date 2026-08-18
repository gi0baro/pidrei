"""Structured Output Tool

Demonstrates `terminate=True` so the agent can end on a tool call without
paying for an extra follow-up LLM turn.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/structured_output.py
"""

from pidrei.core.extensions.types import ToolDefinition
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent
from pidrei_tui import Text


def _get(record, key: str, snake: str):
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, snake, None)


def extension(pi):
    async def execute(_tool_call_id, params, _cancel=None, _on_update=None, _ctx=None):
        return AgentToolResult(
            content=[TextContent(text=f"Saved structured output: {params['headline']}")],
            details={
                "headline": params["headline"],
                "summary": params["summary"],
                "actionItems": params["actionItems"],
            },
            terminate=True,
        )

    def render_result(result, _options, theme, _context):
        details = _get(result, "details", "details")
        if not details:
            blocks = _get(result, "content", "content") or []
            first = blocks[0] if blocks else None
            if first is not None and _get(first, "type", "type") == "text":
                return Text(_get(first, "text", "text") or "", 0, 0)
            return Text("", 0, 0)

        lines = [
            theme.fg("toolTitle", theme.bold(details["headline"])),
            theme.fg("text", details["summary"]),
            "",
            *(theme.fg("muted", f"{index + 1}. {item}") for index, item in enumerate(details["actionItems"])),
        ]
        return Text("\n".join(lines), 0, 0)

    # pi wraps this in defineTool(), a typing helper; pidrei constructs the
    # ToolDefinition directly.
    pi.register_tool(
        ToolDefinition(
            name="structured_output",
            label="Structured Output",
            description=(
                "Return a final structured answer. Use this as your last action when the user asks for "
                "structured output or a machine-readable summary."
            ),
            prompt_snippet="Emit a final structured answer as a terminating tool result",
            prompt_guidelines=[
                (
                    "Use structured_output as your final action when the user asks for structured output, "
                    "JSON-like output, or a machine-readable summary."
                ),
                "After calling structured_output, do not emit another assistant response in the same turn.",
            ],
            parameters={
                "type": "object",
                "properties": {
                    "headline": {"type": "string", "description": "Short title for the result"},
                    "summary": {"type": "string", "description": "One-paragraph summary"},
                    "actionItems": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Concrete next steps or key bullets",
                    },
                },
                "required": ["headline", "summary", "actionItems"],
            },
            execute=execute,
            render_result=render_result,
        )
    )
