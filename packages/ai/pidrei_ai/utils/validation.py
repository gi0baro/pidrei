"""Port of pi's tool-argument validation (packages/ai/src/utils/validation.ts).

Arguments are validated with pi-schema's rules (`pidrei_ai.utils.schema`):
coerced first (`parse()`), then checked, and a failure lists pi-schema's
errors, one line per failing keyword at the dotted path of the value, before
the arguments the model sent.
"""

import copy
import json
import threading
from typing import Any

from pidrei_ai.types import Tool, ToolCall
from pidrei_ai.utils.schema import FORMATS, CoercingValidator, compile


# One compiled validator per schema. pi keys a WeakMap on the schema object;
# dicts cannot be weakly referenced, so the validator is found by the schema's
# identity first, then by its JSON (a reloaded tool set rebuilds equal schemas).
# Identity entries hold the schema so its id cannot be recycled while cached;
# the bound keeps reloaded tool sets from pinning stale schemas forever.
_validator_cache: dict[str, CoercingValidator] = {}
_identity_cache: dict[int, tuple[dict, CoercingValidator]] = {}
_IDENTITY_CACHE_LIMIT = 512
_cache_guard = threading.Lock()


def _get_validator(schema: dict) -> CoercingValidator:
    schema_id = id(schema)
    with _cache_guard:
        entry = _identity_cache.get(schema_id)
        if entry is not None and entry[0] is schema:
            return entry[1]
    key = json.dumps(schema, sort_keys=True, default=str)
    with _cache_guard:
        validator = _validator_cache.get(key)
        if validator is None:
            # A $ref that names nothing (a dangling definition in an MCP server's schema) rejects only values at its
            # place, instead of making every call of the tool fail.
            validator = compile(schema, formats=FORMATS, coerce=True, unresolved_refs="reject")
            _validator_cache[key] = validator
        if len(_identity_cache) >= _IDENTITY_CACHE_LIMIT:
            del _identity_cache[next(iter(_identity_cache))]
        _identity_cache[schema_id] = (schema, validator)
        return validator


def _format_validation_path(pointer: str) -> str:
    """A pi-schema error path (a JSON Pointer) as the dotted path the model sees: `/edits/0/oldText` -> `edits.0.oldText`."""
    if pointer == "":
        return "root"
    return ".".join(segment.replace("~1", "/").replace("~0", "~") for segment in pointer[1:].split("/"))


def validate_tool_call(tools: list[Tool], tool_call: ToolCall) -> Any:
    """Find a tool by name and validate the call's arguments against its schema."""
    tool = next((entry for entry in tools if entry.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(tool: Tool, tool_call: ToolCall) -> Any:
    """Validate tool call arguments against the tool's JSON Schema, coercing them first: models often send `"42"`
    for a number, a single value for an array, or `null` for an omitted optional property. Returns the validated
    (and possibly coerced) arguments; raises `ValueError` with the formatted errors."""
    if tool_call.arguments_error:
        raise ValueError(f'Invalid arguments for tool "{tool_call.name}": {tool_call.arguments_error}')
    # A copy, so tools that modify their arguments never modify the tool call in the transcript.
    result = _get_validator(tool.parameters).parse(copy.deepcopy(tool_call.arguments))
    if result.ok:
        return result.value

    errors = "\n".join(f"  - {_format_validation_path(error.path)}: {error.message}" for error in result.errors)
    raise ValueError(
        f'Validation failed for tool "{tool_call.name}":\n{errors}\n\n'
        f"Received arguments:\n{json.dumps(tool_call.arguments, indent=2, ensure_ascii=False)}"
    )
