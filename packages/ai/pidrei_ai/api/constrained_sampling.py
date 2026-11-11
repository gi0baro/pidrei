"""Port of pi's constrained-sampling helpers (packages/ai/src/api/constrained-sampling.ts)."""

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pidrei_ai.types import Tool
from pidrei_ai.utils.streaming_json import ToolCallArgumentsStream


class UnsupportedStrictJsonSchemaError(Exception):
    pass


# Returns True when a provider's strict mode rejects this schema keyword with this value.
type UnsupportedStrictSchemaKeywordCheck = Callable[[str, Any], bool]


_UNSUPPORTED_STRICT_SCHEMA_KEYS = (
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
)


# pi checks `schema[key] !== undefined`; key presence stands in for that here.
def _is_json_schema_object(value: Any) -> bool:
    return isinstance(value, dict)


def _is_structured_schema(schema: Any) -> bool:
    if not _is_json_schema_object(schema):
        return False
    schema_type = schema.get("type")
    types = [schema_type] if isinstance(schema_type, str) else (schema_type if isinstance(schema_type, list) else [])
    return "object" in types or "array" in types or "properties" in schema or "items" in schema


def _schema_allows_null(schema: Any) -> bool:
    if not _is_json_schema_object(schema):
        return False
    schema_type = schema.get("type")
    if schema_type == "null" or (isinstance(schema_type, list) and "null" in schema_type):
        return True
    if "const" in schema and schema["const"] is None:
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    any_of = schema.get("anyOf")
    return isinstance(any_of, list) and any(_schema_allows_null(variant) for variant in any_of)


def _make_json_schema_node_strict(
    schema: Any, is_unsupported_keyword: UnsupportedStrictSchemaKeywordCheck | None = None
) -> None:
    if not _is_json_schema_object(schema):
        raise UnsupportedStrictJsonSchemaError("boolean schemas are unsupported")
    for key in _UNSUPPORTED_STRICT_SCHEMA_KEYS:
        if key in schema:
            raise UnsupportedStrictJsonSchemaError(f"{key} schemas are unsupported")
    if is_unsupported_keyword:
        for key, value in schema.items():
            if is_unsupported_keyword(key, value):
                rendered = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
                raise UnsupportedStrictJsonSchemaError(f"{key}: {rendered} is unsupported")

    if "anyOf" in schema:
        any_of = schema["anyOf"]
        if not isinstance(any_of, list) or len(any_of) == 0:
            raise UnsupportedStrictJsonSchemaError("anyOf must contain at least one schema")
        for variant in any_of:
            if _is_structured_schema(variant):
                raise UnsupportedStrictJsonSchemaError("object and array unions are unsupported")
            _make_json_schema_node_strict(variant, is_unsupported_keyword)

    if "items" in schema:
        if isinstance(schema["items"], list):
            raise UnsupportedStrictJsonSchemaError("tuple schemas are unsupported")
        _make_json_schema_node_strict(schema["items"], is_unsupported_keyword)

    is_object_schema = schema.get("type") == "object"
    if "properties" in schema and not is_object_schema:
        raise UnsupportedStrictJsonSchemaError("properties require type object")
    if not is_object_schema:
        return
    if "additionalProperties" in schema and schema["additionalProperties"] is not False:
        raise UnsupportedStrictJsonSchemaError("schema-valued or true additionalProperties is unsupported")
    if "properties" in schema and not _is_json_schema_object(schema["properties"]):
        raise UnsupportedStrictJsonSchemaError("object properties must be a schema map")
    if "required" in schema and (
        not isinstance(schema["required"], list) or any(not isinstance(key, str) for key in schema["required"])
    ):
        raise UnsupportedStrictJsonSchemaError("object required must be a string array")

    properties = schema.get("properties") or {}
    property_names = list(properties.keys())
    required = set(schema["required"]) if isinstance(schema.get("required"), list) else set()
    if any(key not in property_names for key in required):
        raise UnsupportedStrictJsonSchemaError("required contains an unknown property")
    for key, prop in list(properties.items()):
        _make_json_schema_node_strict(prop, is_unsupported_keyword)
        if key not in required and not _schema_allows_null(prop):
            properties[key] = {"anyOf": [prop, {"type": "null"}]}
    schema["required"] = property_names
    schema["additionalProperties"] = False


def make_strict_json_schema(
    schema: dict, is_unsupported_keyword: UnsupportedStrictSchemaKeywordCheck | None = None
) -> dict:
    """Convert a tool schema to the strict subset expected by provider constrained sampling."""
    cloned = copy.deepcopy(schema)
    if not _is_json_schema_object(cloned):
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    _make_json_schema_node_strict(cloned, is_unsupported_keyword)
    if cloned.get("type") != "object":
        raise UnsupportedStrictJsonSchemaError("root schema must have type object")
    return cloned


def prefers_json_schema_strict_sampling(tool: Tool) -> bool:
    """True for a tool that asks for strict JSON-schema sampling only when possible (`strict: "prefer"`)."""
    config = tool.constrained_sampling
    # `False` opts the tool out of constrained sampling: not a preferred-strict tool.
    return config is not None and config is not False and config.type == "json_schema" and config.strict == "prefer"


def get_json_schema_tool_parameters(tool: Tool, strict: bool | None) -> dict:
    return make_strict_json_schema(tool.parameters) if strict is True else tool.parameters


def resolve_json_schema_strict_sampling(
    tool: Tool,
    supports_strict_mode: bool,
    is_unsupported_keyword: UnsupportedStrictSchemaKeywordCheck | None = None,
) -> bool | None:
    """Decide whether a JSON-schema tool is sent in strict mode. `is_unsupported_keyword` lets a provider
    reject extra keywords its strict mode does not accept, so "prefer" tools fall back to non-strict.
    """
    config = tool.constrained_sampling
    if not config or config is True or config.type != "json_schema":
        return None

    if supports_strict_mode:
        try:
            make_strict_json_schema(tool.parameters, is_unsupported_keyword)
        except UnsupportedStrictJsonSchemaError as error:
            if config.strict != "require":
                return None
            raise ValueError(f'Tool "{tool.name}" requires JSON-schema constrained sampling, but {error}.') from error
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
    """A streaming grammar tool input, re-encoded as the JSON arguments text of its call.

    The input is kept in pieces, so appending costs the piece, never the input
    so far. `arguments` is what the call's partial arguments show,
    `{property: input}`: pi rebuilds it from the whole input on every delta,
    which is free with JavaScript's rope strings; here it is read back from
    the JSON pieces through a `ToolCallArgumentsStream`, so the preview of a
    large input lags a little behind instead of costing the whole input per
    delta (see `pidrei_ai.utils.streaming_json`). Once closed it is exact.
    """

    started: bool = False
    closed: bool = False
    arguments: dict[str, Any] = field(default_factory=dict)
    _parts: list[str] = field(default_factory=list)
    _json: ToolCallArgumentsStream = field(default_factory=ToolCallArgumentsStream)

    @property
    def input(self) -> str:
        """The raw input so far."""
        parts = self._parts
        if len(parts) > 1:
            parts[:] = ["".join(parts)]
        return parts[0] if parts else ""


def get_grammar_tool_input(tool_name: str, arguments: dict, input_property: str) -> str:
    value = arguments.get(input_property)
    if not isinstance(value, str):
        raise TypeError(f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.')
    return value


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer, input_property: str, input_delta: str
) -> str | None:
    """Append `input_delta` to a streaming grammar tool input and return the matching
    piece of its JSON arguments text, or `None` for an empty delta.

    Costs time linear in the delta, never in the input so far.
    """
    if buffer.closed:
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    if not input_delta:
        return None
    buffer._parts.append(input_delta)
    delta = _grammar_tool_input_json_start(buffer, input_property) + json.dumps(input_delta, ensure_ascii=False)[1:-1]
    buffer.arguments = buffer._json.append(delta)
    return delta


def close_grammar_tool_input_json(
    buffer: GrammarToolInputJsonBuffer, input_property: str, final_input: str | None
) -> str | None:
    """Close a streaming grammar tool input and return the rest of its JSON arguments text.

    `final_input` is the complete input when the provider sends one; it must
    continue the input streamed so far. Closing again with the same input
    returns `None`.
    """
    if buffer.closed:
        if final_input is None or final_input == buffer.input:
            return None
        raise ValueError(f'grammar tool input for property "{input_property}" changed after it was closed')
    current = buffer.input
    if final_input is not None and not final_input.startswith(current):
        raise ValueError(f'grammar tool input for property "{input_property}" changed non-monotonically')
    rest = "" if final_input is None else final_input[len(current) :]
    if final_input is not None:
        buffer._parts[:] = [final_input]
    buffer.closed = True
    buffer.arguments = {input_property: buffer.input}
    return f'{_grammar_tool_input_json_start(buffer, input_property)}{json.dumps(rest, ensure_ascii=False)[1:-1]}"}}'


def _grammar_tool_input_json_start(buffer: GrammarToolInputJsonBuffer, input_property: str) -> str:
    """The opening of the JSON arguments text, once."""
    if buffer.started:
        return ""
    buffer.started = True
    return f'{{{json.dumps(input_property)}:"'


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
