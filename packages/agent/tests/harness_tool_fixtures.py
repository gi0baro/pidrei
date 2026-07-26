"""Mirror of pi agent/test/utils/calculate.ts and get-current-time.ts (harness-tool shaped)."""

import time
from typing import Any

from pidrei_agent.harness.types import AgentHarnessTool
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import TextContent, Usage


CALCULATE_SCHEMA = {
    "type": "object",
    "properties": {"expression": {"type": "string", "description": "The mathematical expression to evaluate"}},
    "required": ["expression"],
}

GET_CURRENT_TIME_SCHEMA = {
    "type": "object",
    "properties": {
        "timezone": {"type": "string", "description": "Optional timezone (e.g., 'America/New_York', 'Europe/London')"}
    },
}


def calculate(expression: str) -> AgentToolResult[None]:
    result = eval(expression)  # noqa: S307 - test fixture mirroring pi's `new Function`
    return AgentToolResult(content=[TextContent(text=f"{expression} = {result}")], details=None)


class FnHarnessTool(AgentHarnessTool):
    """Configurable harness tool; extra attributes (e.g. `source`) may be attached."""

    def __init__(self, name: str, label: str, description: str, parameters: dict, execute, **extra: Any):
        self.name = name
        self.label = label
        self.description = description
        self.parameters = parameters
        self.execution_mode = None
        self.prepare_arguments = None
        self._execute = execute
        for key, value in extra.items():
            setattr(self, key, value)

    def clone(self, **overrides: Any) -> FnHarnessTool:
        merged = {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "parameters": self.parameters,
            "execute": self._execute,
        }
        merged.update(overrides)
        execute = merged.pop("execute")
        extra = {
            key: value for key, value in merged.items() if key not in ("name", "label", "description", "parameters")
        }
        return FnHarnessTool(
            merged["name"], merged["label"], merged["description"], merged["parameters"], execute, **extra
        )

    async def execute(self, tool_call_id, params, cancel, on_update, context):
        return await self._execute(tool_call_id, params, cancel, on_update, context)


async def _calculate_execute(_tool_call_id, params, _cancel, _on_update, _context):
    return calculate(params["expression"])


def make_calculate_tool() -> FnHarnessTool:
    return FnHarnessTool(
        "calculate", "Calculator", "Evaluate mathematical expressions", CALCULATE_SCHEMA, _calculate_execute
    )


calculate_tool = make_calculate_tool()


def create_calculate_tool_with_usage(usage: Usage) -> FnHarnessTool:
    async def execute(tool_call_id, params, cancel, on_update, context):
        result = calculate(params["expression"])
        result.usage = usage
        return result

    return FnHarnessTool("calculate", "Calculator", "Evaluate mathematical expressions", CALCULATE_SCHEMA, execute)


async def _get_current_time_execute(_tool_call_id, _params, _cancel, _on_update, _context):
    now = int(time.time() * 1000)
    return AgentToolResult(content=[TextContent(text=time.strftime("%A, %B %d, %Y"))], details={"utcTimestamp": now})


get_current_time_tool = FnHarnessTool(
    "get_current_time",
    "Current Time",
    "Get the current date and time",
    GET_CURRENT_TIME_SCHEMA,
    _get_current_time_execute,
)
