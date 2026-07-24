"""Port of pi's constrained-sampling helpers (packages/ai/src/api/constrained-sampling.ts).

Currently the JSON-schema strict resolver used by the anthropic adapter; the
grammar-tool helpers join with the OpenAI adapters (PLAN.md).
"""

from pidrei_ai.types import Tool


def resolve_json_schema_strict_sampling(tool: Tool, supports_strict_mode: bool) -> bool | None:
    config = tool.constrained_sampling
    if not config or config is True or config.type != "json_schema":
        return None

    if supports_strict_mode:
        return True
    if config.strict == "require":
        raise ValueError(
            f'Tool "{tool.name}" requires JSON-schema constrained sampling, but strict tools are unsupported.'
        )
    return None
