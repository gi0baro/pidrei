"""Port of pi's constrained-sampling helpers (packages/ai/src/api/constrained-sampling.ts)."""

import json
from dataclasses import dataclass

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


@dataclass(slots=True)
class GrammarConstrainedSampling:
    format: str  # "lark" | "regex"
    definition: str
    input_property: str


@dataclass(slots=True)
class GrammarToolInputJsonBuffer:
    input: str = ""
    started: bool = False
    closed: bool = False


def get_grammar_tool_input(tool_name: str, arguments: dict, input_property: str) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise TypeError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    close: bool,
) -> str | None:
    """Incrementally re-encode a grammar tool's raw input as a JSON object delta."""
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    if not next_input.startswith(buffer.input):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')

    input_delta = next_input[len(buffer.input) :]
    if not close and not input_delta:
        return None

    delta = ""
    if not buffer.started:
        delta += f'{{{json.dumps(input_property)}:"'
        buffer.started = True
    delta += json.dumps(input_delta, ensure_ascii=False)[1:-1]
    buffer.input = next_input

    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(tool: Tool) -> str:
    schema = tool.parameters or {}
    if schema.get("type") != "object":
        raise ValueError("grammar constrained sampling requires an object parameter schema")
    required = schema.get("required")
    if not isinstance(required, list) or len(required) != 1 or not isinstance(required[0], str):
        raise ValueError("grammar constrained sampling requires exactly one required string property")

    input_property = required[0]
    properties = schema.get("properties") or {}
    property_schema = properties.get(input_property)
    if not property_schema:
        raise ValueError(f"grammar constrained sampling requires a properties entry for {input_property}")
    if property_schema.get("type") != "string":
        raise ValueError(f"grammar constrained sampling property {input_property} must have type string")
    return input_property


def resolve_grammar_constrained_sampling(
    tool: Tool,
    supports_openai_grammar_tools: bool,
) -> GrammarConstrainedSampling | None:
    config = tool.constrained_sampling
    if not config or config is True or config.type != "grammar":
        return None

    if not supports_openai_grammar_tools:
        return None

    lark_definition = config.variants.get("openai_lark")
    regex_definition = config.variants.get("openai_regex")
    has_lark = isinstance(lark_definition, str) and lark_definition.strip() != ""
    has_regex = isinstance(regex_definition, str) and regex_definition.strip() != ""
    if not has_lark and not has_regex:
        raise ValueError(
            f'Tool "{tool.name}" cannot use grammar constrained sampling: no supported grammar variant was provided.'
        )

    try:
        return GrammarConstrainedSampling(
            format="lark" if has_lark else "regex",
            definition=lark_definition if has_lark else regex_definition,  # type: ignore[arg-type]
            input_property=_infer_grammar_input_property(tool),
        )
    except ValueError as error:
        raise ValueError(f'Tool "{tool.name}" cannot use grammar constrained sampling: {error}.')


def create_grammar_tool_input_properties(
    tools: list[Tool] | None,
    supports_openai_grammar_tools: bool,
) -> dict[str, str]:
    properties: dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar:
            properties[tool.name] = grammar.input_property
    return properties
