"""Mirror of pi coding-agent src/core/model-config.ts.

Immutable, credential-blind models.json snapshot. Provider configs stay in
their raw camelCase dict shape (pi keeps the frozen TypeBox-validated JS
objects); the composer converts to typed models at composition time.

pi validates with TypeBox; here the equivalent JSON Schema runs through
`jsonschema`. Validation-error *text* differs from TypeBox's; the message
frame ("Invalid models.json schema:\\n... \\n\\nFile: path") is preserved.
"""

import copy
import json
from typing import Any

import jsonschema
from tonio.colored import fs

from pidrei_ai.providers.compat_schema import PROVIDER_COMPAT_SCHEMA
from pidrei_ai.providers.model_schema import (
    MODEL_COST_SCHEMA,
    MODEL_INPUT_LIMITS_SCHEMA,
    MODEL_INPUT_MODALITY_SCHEMA,
    MODEL_PROMPT_CACHE_SCHEMA,
    THINKING_LEVEL_MAP_SCHEMA,
)

from ..utils.json_util import strip_json_comments
from ..utils.paths import normalize_path
from ..utils.text import strip_bom


def _non_empty_string(description: str) -> dict[str, Any]:
    return {"type": "string", "minLength": 1, "description": description}


_SAMPLING_PARAMS = {"type": "object", "additionalProperties": {}}
_SAMPLING_PARAMS_BY_THINKING_LEVEL = {
    "type": "object",
    "properties": dict.fromkeys(("off", "minimal", "low", "medium", "high", "xhigh", "max"), _SAMPLING_PARAMS),
    "description": "samplingParams per thinking level, merged over samplingParams.",
}

_SAMPLING_PARAMS_FIELD = {
    **_SAMPLING_PARAMS,
    "description": "Fields merged into every request body. Applies only to OpenAI-compatible APIs.",
}
_HEADERS = {
    "type": "object",
    "additionalProperties": {"type": "string"},
    "description": "Request headers. Values resolve like apiKey.",
}

# Model fields shared by models entries and modelOverrides entries. Defaults apply only to models entries.
_MODEL_FIELD_DESCRIPTIONS = {
    "name": "Display name, also matched by --model patterns.",
    "reasoning": "Whether the model supports thinking.",
    "contextWindow": "Context window in tokens.",
    "maxTokens": "Maximum output tokens.",
}

_INPUT_MODALITIES = {"type": "array", "items": MODEL_INPUT_MODALITY_SCHEMA}

_MODEL_DEFINITION = {
    "type": "object",
    "required": ["id"],
    "properties": {
        "id": _non_empty_string("Model ID sent to the API."),
        "name": _non_empty_string(_MODEL_FIELD_DESCRIPTIONS["name"]),
        "api": _non_empty_string("Overrides the provider api."),
        "baseUrl": _non_empty_string("Overrides the provider baseUrl."),
        "reasoning": {"type": "boolean", "default": False, "description": _MODEL_FIELD_DESCRIPTIONS["reasoning"]},
        "thinkingLevelMap": THINKING_LEVEL_MAP_SCHEMA,
        "input": {**_INPUT_MODALITIES, "default": ["text"]},
        "inputLimits": MODEL_INPUT_LIMITS_SCHEMA,
        "cost": MODEL_COST_SCHEMA,
        "promptCache": MODEL_PROMPT_CACHE_SCHEMA,
        "contextWindow": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 128000,
            "description": _MODEL_FIELD_DESCRIPTIONS["contextWindow"],
        },
        "maxTokens": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 16384,
            "description": _MODEL_FIELD_DESCRIPTIONS["maxTokens"],
        },
        "samplingParams": _SAMPLING_PARAMS_FIELD,
        "samplingParamsByThinkingLevel": _SAMPLING_PARAMS_BY_THINKING_LEVEL,
        "headers": _HEADERS,
        "compat": PROVIDER_COMPAT_SCHEMA,
    },
}

# pi's `Type.Partial(ModelCostSchema)`: every rate optional.
_PARTIAL_MODEL_COST = {key: value for key, value in MODEL_COST_SCHEMA.items() if key != "required"}

_MODEL_OVERRIDE = {
    "type": "object",
    "properties": {
        "name": _non_empty_string(_MODEL_FIELD_DESCRIPTIONS["name"]),
        "reasoning": {"type": "boolean", "description": _MODEL_FIELD_DESCRIPTIONS["reasoning"]},
        "thinkingLevelMap": THINKING_LEVEL_MAP_SCHEMA,
        "input": _INPUT_MODALITIES,
        "inputLimits": MODEL_INPUT_LIMITS_SCHEMA,
        "cost": _PARTIAL_MODEL_COST,
        "promptCache": MODEL_PROMPT_CACHE_SCHEMA,
        "contextWindow": {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": _MODEL_FIELD_DESCRIPTIONS["contextWindow"],
        },
        "maxTokens": {"type": "number", "exclusiveMinimum": 0, "description": _MODEL_FIELD_DESCRIPTIONS["maxTokens"]},
        "samplingParams": _SAMPLING_PARAMS_FIELD,
        "samplingParamsByThinkingLevel": _SAMPLING_PARAMS_BY_THINKING_LEVEL,
        "headers": _HEADERS,
        "compat": PROVIDER_COMPAT_SCHEMA,
    },
}

_PROVIDER_CONFIG = {
    "type": "object",
    "properties": {
        "name": _non_empty_string("Display name."),
        "baseUrl": _non_empty_string("API base URL. Also applies to the provider's built-in models."),
        "apiKey": _non_empty_string('Literal key, "$NAME" environment interpolation, or "!command" output.'),
        "api": _non_empty_string(
            'API for the provider\'s models, such as "openai-completions", "openai-responses", '
            '"anthropic-messages", or "google-generative-ai".'
        ),
        "headers": _HEADERS,
        "compat": PROVIDER_COMPAT_SCHEMA,
        "authHeader": {
            "type": "boolean",
            "default": False,
            "description": "Whether to send the resolved API key as Authorization: Bearer.",
        },
        "models": {"type": "array", "items": _MODEL_DEFINITION},
        "modelOverrides": {"type": "object", "additionalProperties": _MODEL_OVERRIDE},
    },
}

MODELS_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["providers"],
    "properties": {
        "$schema": {"type": "string"},
        "providers": {"type": "object", "additionalProperties": _PROVIDER_CONFIG},
    },
}

# Entry schemas rendered into docs/models-configuration.md.
MODELS_JSON_SCHEMAS: dict[str, dict[str, Any]] = {
    "provider": _PROVIDER_CONFIG,
    "model": _MODEL_DEFINITION,
    "modelOverride": _MODEL_OVERRIDE,
}

_VALIDATOR = jsonschema.Draft202012Validator(MODELS_CONFIG_SCHEMA)


def _format_validation_path(error: jsonschema.ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    if error.validator == "required":
        missing = error.message.split("'")[1] if "'" in error.message else None
        if missing:
            return f"{path}.{missing}" if path else missing
    return path or "root"


class ModelConfig:
    """One immutable load of models.json."""

    def __init__(self, providers: dict[str, dict[str, Any]], error: str | None = None):
        self._providers = providers
        self._error = error

    @staticmethod
    async def load(models_json_path: str | None) -> ModelConfig:
        if not models_json_path:
            return ModelConfig({})
        path = normalize_path(models_json_path)

        try:
            content = await fs.Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ModelConfig({})
        except Exception as error:
            return ModelConfig({}, f"Failed to load models.json: {error}\n\nFile: {path}")

        try:
            parsed = json.loads(strip_json_comments(strip_bom(content)))
        except Exception as error:
            return ModelConfig({}, f"Failed to parse models.json: {error}\n\nFile: {path}")

        errors = sorted(_VALIDATOR.iter_errors(parsed), key=lambda e: list(map(str, e.absolute_path)))
        if errors:
            formatted = "\n".join(f"  - {_format_validation_path(error)}: {error.message}" for error in errors)
            return ModelConfig(
                {}, f"Invalid models.json schema:\n{formatted or 'Unknown schema error'}\n\nFile: {path}"
            )

        providers = {provider_id: copy.deepcopy(provider) for provider_id, provider in parsed["providers"].items()}
        return ModelConfig(providers)

    def get_provider(self, provider_id: str) -> dict[str, Any] | None:
        return self._providers.get(provider_id)

    def get_provider_ids(self) -> list[str]:
        return list(self._providers.keys())

    def get_error(self) -> str | None:
        return self._error
