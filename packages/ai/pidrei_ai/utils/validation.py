"""Port of pi's tool-argument validation (packages/ai/src/utils/validation.ts).

pi layers TypeBox's `Value.Convert` plus a custom JSON-schema coercion for
plain (serialized) schemas; pidrei tool schemas are always plain JSON Schema
dicts, so the effective pipeline is the custom coercion followed by
validation. Coercion semantics mirror pi exactly (they are observable by the
model). Validation runs on `jsonschema`, so error message *text* differs from
TypeBox's localized strings while the error structure ("  - path: message"
lines, required-property paths, the "Validation failed" envelope) mirrors pi.
"""

import copy
import json
import math
import re
import threading
from typing import Any

from jsonschema import Draft202012Validator

from pidrei_ai.types import Tool, ToolCall


_validator_cache: dict[str, Draft202012Validator] = {}
_cache_guard = threading.Lock()


def _get_validator(schema: dict) -> Draft202012Validator:
    key = json.dumps(schema, sort_keys=True, default=str)
    with _cache_guard:
        validator = _validator_cache.get(key)
        if validator is None:
            validator = Draft202012Validator(schema)
            _validator_cache[key] = validator
        return validator


def _sub_schema_check(schema: dict, value: Any) -> bool:
    try:
        return _get_validator(schema).is_valid(value)
    except Exception:
        return False


def _get_schema_types(schema: dict) -> list[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [entry for entry in schema_type if isinstance(entry, str)]
    return []


def _matches_json_type(value: Any, type_name: str) -> bool:
    if type_name == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if type_name == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if type_name == "boolean":
        return isinstance(value, bool)
    if type_name == "string":
        return isinstance(value, str)
    if type_name == "null":
        return value is None
    if type_name == "array":
        return isinstance(value, list)
    if type_name == "object":
        return isinstance(value, dict)
    return False


def _js_number(text: str) -> float | None:
    """JS `Number(text)` for the trimmed-non-empty string case."""
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _js_string(value: float | bool) -> str:
    """JS `String(value)` for numbers and booleans."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _coerce_primitive_by_type(value: Any, type_name: str) -> Any:
    """Mirror of pi's coercePrimitiveByType; returns `value` itself when unchanged."""
    if type_name == "number":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip() != "":
            parsed = _js_number(value)
            if parsed is not None:
                return int(parsed) if parsed.is_integer() else parsed
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    if type_name == "integer":
        if value is None:
            return 0
        if isinstance(value, str) and value.strip() != "":
            parsed = _js_number(value)
            if parsed is not None and parsed.is_integer():
                return int(parsed)
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    if type_name == "boolean":
        if value is None:
            return False
        if isinstance(value, str):
            if value == "true":
                return True
            if value == "false":
                return False
        if isinstance(value, int | float) and not isinstance(value, bool):
            if value == 1:
                return True
            if value == 0:
                return False
        return value

    if type_name == "string":
        if value is None:
            return ""
        if isinstance(value, int | float | bool):
            return _js_string(value)
        return value

    if type_name == "null":
        if isinstance(value, str) and value == "":
            return None
        if isinstance(value, bool):
            return None if value is False else value
        if isinstance(value, int | float) and value == 0:
            return None
        return value

    return value


def _apply_schema_object_coercion(value: dict, schema: dict) -> None:
    properties = schema.get("properties")
    defined_keys = set(properties.keys()) if isinstance(properties, dict) else set()

    if isinstance(properties, dict):
        for key, property_schema in properties.items():
            if key not in value or not isinstance(property_schema, dict):
                continue
            value[key] = _coerce_with_json_schema(value[key], property_schema)

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key, property_value in value.items():
            if key in defined_keys:
                continue
            value[key] = _coerce_with_json_schema(property_value, additional)


def _apply_schema_array_coercion(value: list, schema: dict) -> None:
    items = schema.get("items")
    if isinstance(items, list):
        for index in range(len(value)):
            if index >= len(items) or not isinstance(items[index], dict):
                continue
            value[index] = _coerce_with_json_schema(value[index], items[index])
        return

    if isinstance(items, dict):
        for index in range(len(value)):
            value[index] = _coerce_with_json_schema(value[index], items)


def _coerce_with_union_schema(value: Any, schemas: list) -> Any:
    for schema in schemas:
        if isinstance(schema, dict) and _sub_schema_check(schema, value):
            return value

    for schema in schemas:
        if not isinstance(schema, dict):
            continue
        candidate = copy.deepcopy(value)
        coerced = _coerce_with_json_schema(candidate, schema)
        if _sub_schema_check(schema, coerced):
            return coerced
    return value


def _coerce_with_json_schema(value: Any, schema: dict) -> Any:
    next_value = value

    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for nested in all_of:
            if isinstance(nested, dict):
                next_value = _coerce_with_json_schema(next_value, nested)

    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        next_value = _coerce_with_union_schema(next_value, any_of)

    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        next_value = _coerce_with_union_schema(next_value, one_of)

    schema_types = _get_schema_types(schema)
    matches_union_member = len(schema_types) > 1 and any(
        _matches_json_type(next_value, schema_type) for schema_type in schema_types
    )
    if schema_types and not matches_union_member:
        for schema_type in schema_types:
            candidate = _coerce_primitive_by_type(next_value, schema_type)
            if candidate is not next_value:
                next_value = candidate
                break

    if "object" in schema_types and isinstance(next_value, dict):
        _apply_schema_object_coercion(next_value, schema)

    if "array" in schema_types and isinstance(next_value, list):
        _apply_schema_array_coercion(next_value, schema)

    return next_value


_REQUIRED_PROPERTY_RE = re.compile(r"'(.+?)' is a required property")


def _format_validation_path(error) -> str:
    parts = [str(part) for part in error.absolute_path]
    if error.validator == "required":
        match = _REQUIRED_PROPERTY_RE.search(error.message)
        if match:
            base_path = ".".join(parts)
            return f"{base_path}.{match.group(1)}" if base_path else match.group(1)
    return ".".join(parts) or "root"


def validate_tool_call(tools: list[Tool], tool_call: ToolCall) -> Any:
    """Find a tool by name and validate the call's arguments against its schema."""
    tool = next((entry for entry in tools if entry.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> Any:
    """Validate tool call arguments, coercing pi-style; raises on failure."""
    args = copy.deepcopy(tool_call.arguments)
    coerced = _coerce_with_json_schema(args, tool.parameters)

    candidate = args
    if coerced is not args:
        if isinstance(args, dict) and isinstance(coerced, dict):
            candidate = coerced
        elif _sub_schema_check(tool.parameters, coerced):
            return coerced

    validator = _get_validator(tool.parameters)
    if validator.is_valid(candidate):
        return candidate

    error_lines = "\n".join(
        f"  - {_format_validation_path(error)}: {error.message}"
        for error in sorted(validator.iter_errors(candidate), key=lambda error: [str(p) for p in error.absolute_path])
    )
    errors = error_lines or "Unknown validation error"

    raise ValueError(
        f'Validation failed for tool "{tool_call.name}":\n{errors}\n\n'
        f"Received arguments:\n{json.dumps(tool_call.arguments, indent=2, ensure_ascii=False)}"
    )
