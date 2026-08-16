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

from ..utils.json_util import strip_json_comments
from ..utils.paths import normalize_path


_STRING = {"type": "string"}
_NON_EMPTY_STRING = {"type": "string", "minLength": 1}
_NUMBER = {"type": "number"}
_BOOLEAN = {"type": "boolean"}

_PERCENTILE_CUTOFFS = {
    "type": "object",
    "properties": {"p50": _NUMBER, "p75": _NUMBER, "p90": _NUMBER, "p99": _NUMBER},
}

_STRING_ARRAY = {"type": "array", "items": _STRING}

_MAX_PRICE_VALUE = {"anyOf": [_NUMBER, _STRING]}

_OPENROUTER_ROUTING = {
    "type": "object",
    "properties": {
        "allow_fallbacks": _BOOLEAN,
        "require_parameters": _BOOLEAN,
        "data_collection": {"enum": ["deny", "allow"]},
        "zdr": _BOOLEAN,
        "enforce_distillable_text": _BOOLEAN,
        "order": _STRING_ARRAY,
        "only": _STRING_ARRAY,
        "ignore": _STRING_ARRAY,
        "quantizations": _STRING_ARRAY,
        "sort": {
            "anyOf": [
                _STRING,
                {
                    "type": "object",
                    "properties": {"by": _STRING, "partition": {"anyOf": [_STRING, {"type": "null"}]}},
                },
            ]
        },
        "max_price": {
            "type": "object",
            "properties": {
                "prompt": _MAX_PRICE_VALUE,
                "completion": _MAX_PRICE_VALUE,
                "image": _MAX_PRICE_VALUE,
                "audio": _MAX_PRICE_VALUE,
                "request": _MAX_PRICE_VALUE,
            },
        },
        "preferred_min_throughput": {"anyOf": [_NUMBER, _PERCENTILE_CUTOFFS]},
        "preferred_max_latency": {"anyOf": [_NUMBER, _PERCENTILE_CUTOFFS]},
    },
}

_VERCEL_GATEWAY_ROUTING = {
    "type": "object",
    "properties": {"only": _STRING_ARRAY, "order": _STRING_ARRAY},
}

_THINKING_LEVEL_MAP_VALUE = {"anyOf": [_STRING, {"type": "null"}]}
_THINKING_LEVEL_MAP = {
    "type": "object",
    "properties": dict.fromkeys(("off", "minimal", "low", "medium", "high", "xhigh", "max"), _THINKING_LEVEL_MAP_VALUE),
}

_CHAT_TEMPLATE_KWARG = {
    "anyOf": [
        _STRING,
        _NUMBER,
        _BOOLEAN,
        {"type": "null"},
        {
            "type": "object",
            "properties": {
                "$var": {"enum": ["thinking.enabled", "thinking.effort"]},
                "omitWhenOff": _BOOLEAN,
            },
            "required": ["$var"],
        },
    ]
}

_SESSION_AFFINITY_FORMAT = {"enum": ["openai", "openai-nosession", "openrouter"]}

_OPENAI_COMPLETIONS_COMPAT = {
    "type": "object",
    "properties": {
        "supportsStore": _BOOLEAN,
        "supportsDeveloperRole": _BOOLEAN,
        "supportsReasoningEffort": _BOOLEAN,
        "supportsUsageInStreaming": _BOOLEAN,
        "maxTokensField": {"enum": ["max_completion_tokens", "max_tokens"]},
        "requiresToolResultName": _BOOLEAN,
        "requiresAssistantAfterToolResult": _BOOLEAN,
        "requiresThinkingAsText": _BOOLEAN,
        "requiresReasoningContentOnAssistantMessages": _BOOLEAN,
        "thinkingFormat": {
            "enum": [
                "openai",
                "openrouter",
                "together",
                "baseten",
                "deepseek",
                "zai",
                "qwen",
                "chat-template",
                "qwen-chat-template",
                "string-thinking",
                "ant-ling",
            ]
        },
        "chatTemplateKwargs": {"type": "object", "additionalProperties": _CHAT_TEMPLATE_KWARG},
        "chatTemplateArgs": {"type": "object", "additionalProperties": _CHAT_TEMPLATE_KWARG},
        "cacheControlFormat": {"const": "anthropic"},
        "openRouterRouting": _OPENROUTER_ROUTING,
        "vercelGatewayRouting": _VERCEL_GATEWAY_ROUTING,
        "supportsOpenAIGrammarTools": _BOOLEAN,
        "supportsStrictMode": _BOOLEAN,
        "sendSessionAffinityHeaders": _BOOLEAN,
        "deferredToolsMode": {"const": "kimi"},
        "sessionAffinityFormat": _SESSION_AFFINITY_FORMAT,
        "supportsLongCacheRetention": _BOOLEAN,
    },
}

_OPENAI_RESPONSES_COMPAT = {
    "type": "object",
    "properties": {
        "supportsDeveloperRole": _BOOLEAN,
        "sessionAffinityFormat": _SESSION_AFFINITY_FORMAT,
        "supportsLongCacheRetention": _BOOLEAN,
        "supportsStrictMode": _BOOLEAN,
        "supportsOpenAIGrammarTools": _BOOLEAN,
        "supportsAdditionalTools": _BOOLEAN,
        "supportsToolSearch": _BOOLEAN,
    },
}

_ANTHROPIC_MESSAGES_COMPAT = {
    "type": "object",
    "properties": {
        "supportsEagerToolInputStreaming": _BOOLEAN,
        "supportsLongCacheRetention": _BOOLEAN,
        "sendSessionAffinityHeaders": _BOOLEAN,
        "supportsCacheControlOnTools": _BOOLEAN,
        "supportsTemperature": _BOOLEAN,
        "forceAdaptiveThinking": _BOOLEAN,
        "allowEmptySignature": _BOOLEAN,
        "supportsStrictTools": _BOOLEAN,
        "supportsToolReferences": _BOOLEAN,
    },
}

_PROVIDER_COMPAT = {"anyOf": [_OPENAI_COMPLETIONS_COMPAT, _OPENAI_RESPONSES_COMPAT, _ANTHROPIC_MESSAGES_COMPAT]}

_MODEL_COST_RATES = {
    "input": _NUMBER,
    "output": _NUMBER,
    "cacheRead": _NUMBER,
    "cacheWrite": _NUMBER,
}
_MODEL_COST_TIER = {
    "type": "object",
    "properties": {"inputTokensAbove": _NUMBER, **_MODEL_COST_RATES},
    "required": ["inputTokensAbove", "input", "output", "cacheRead", "cacheWrite"],
}
_MODEL_COST = {
    "type": "object",
    "properties": {**_MODEL_COST_RATES, "tiers": {"type": "array", "items": _MODEL_COST_TIER}},
    "required": ["input", "output", "cacheRead", "cacheWrite"],
}

_STRING_RECORD = {"type": "object", "additionalProperties": _STRING}

_INPUT_MODALITIES = {"type": "array", "items": {"enum": ["text", "image"]}}

_MODEL_DEFINITION = {
    "type": "object",
    "properties": {
        "id": _NON_EMPTY_STRING,
        "name": _NON_EMPTY_STRING,
        "api": _NON_EMPTY_STRING,
        "baseUrl": _NON_EMPTY_STRING,
        "reasoning": _BOOLEAN,
        "thinkingLevelMap": _THINKING_LEVEL_MAP,
        "input": _INPUT_MODALITIES,
        "cost": _MODEL_COST,
        "contextWindow": _NUMBER,
        "maxTokens": _NUMBER,
        "samplingParams": {"type": "object"},
        "headers": _STRING_RECORD,
        "compat": _PROVIDER_COMPAT,
    },
    "required": ["id"],
}

_MODEL_OVERRIDE = {
    "type": "object",
    "properties": {
        "name": _NON_EMPTY_STRING,
        "reasoning": _BOOLEAN,
        "thinkingLevelMap": _THINKING_LEVEL_MAP,
        "input": _INPUT_MODALITIES,
        "cost": {
            "type": "object",
            "properties": {**_MODEL_COST_RATES, "tiers": {"type": "array", "items": _MODEL_COST_TIER}},
        },
        "contextWindow": _NUMBER,
        "maxTokens": _NUMBER,
        "samplingParams": {"type": "object"},
        "headers": _STRING_RECORD,
        "compat": _PROVIDER_COMPAT,
    },
}

_PROVIDER_CONFIG = {
    "type": "object",
    "properties": {
        "name": _NON_EMPTY_STRING,
        "baseUrl": _NON_EMPTY_STRING,
        "apiKey": _NON_EMPTY_STRING,
        "api": _NON_EMPTY_STRING,
        "headers": _STRING_RECORD,
        "compat": _PROVIDER_COMPAT,
        "authHeader": _BOOLEAN,
        "models": {"type": "array", "items": _MODEL_DEFINITION},
        "modelOverrides": {"type": "object", "additionalProperties": _MODEL_OVERRIDE},
    },
}

_MODELS_CONFIG_SCHEMA = {
    "type": "object",
    "properties": {"providers": {"type": "object", "additionalProperties": _PROVIDER_CONFIG}},
    "required": ["providers"],
}

_VALIDATOR = jsonschema.Draft202012Validator(_MODELS_CONFIG_SCHEMA)


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
            parsed = json.loads(strip_json_comments(content))
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
