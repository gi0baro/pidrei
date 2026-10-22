"""Mirror of pi ai src/providers/model-schema.ts.

Model metadata schemas shared by models.json validation and the published
configuration schemas. pi builds them with TypeBox; here they are the JSON
Schema TypeBox emits, as plain dicts (validated with `jsonschema`).
"""

from typing import Any


THINKING_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")
MODEL_THINKING_LEVELS = ("off", *THINKING_LEVELS)
MODEL_INPUT_MODALITIES = ("text", "image")
CACHE_RETENTIONS = ("none", "short", "long")
MODEL_PROMPT_CACHE_RETENTIONS = ("short", "long")

THINKING_LEVEL_SCHEMA: dict[str, Any] = {"enum": list(THINKING_LEVELS)}
MODEL_THINKING_LEVEL_SCHEMA: dict[str, Any] = {"enum": list(MODEL_THINKING_LEVELS)}
MODEL_INPUT_MODALITY_SCHEMA: dict[str, Any] = {"enum": list(MODEL_INPUT_MODALITIES)}
CACHE_RETENTION_SCHEMA: dict[str, Any] = {"enum": list(CACHE_RETENTIONS)}

_THINKING_LEVEL_MAP_VALUE_SCHEMA = {"anyOf": [{"type": "string"}, {"type": "null"}]}
THINKING_LEVEL_MAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": dict.fromkeys(MODEL_THINKING_LEVELS, _THINKING_LEVEL_MAP_VALUE_SCHEMA),
    "description": (
        "Provider value sent for each thinking level. null marks a level unsupported. Levels the model does "
        "not map use the provider's default value through high; xhigh and max are unsupported."
    ),
}

MODEL_PROMPT_CACHE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {retention: {"type": "number", "exclusiveMinimum": 0} for retention in MODEL_PROMPT_CACHE_RETENTIONS},
    "description": (
        "Best-effort prompt cache lifetime in seconds for the short and long retention tiers. PiDrei warms "
        "only tiers with a known lifetime."
    ),
}

_MODEL_COST_RATES_PROPERTIES = {
    "input": {"type": "number", "description": "Input cost in USD per million tokens."},
    "output": {"type": "number", "description": "Output cost in USD per million tokens."},
    "cacheRead": {"type": "number", "description": "Cache-read cost in USD per million tokens."},
    "cacheWrite": {"type": "number", "description": "Cache-write cost in USD per million tokens."},
}

MODEL_COST_RATES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": list(_MODEL_COST_RATES_PROPERTIES),
    "properties": _MODEL_COST_RATES_PROPERTIES,
}
MODEL_COST_TIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["inputTokensAbove", *_MODEL_COST_RATES_PROPERTIES],
    "properties": {
        "inputTokensAbove": {
            "type": "number",
            "description": "Use this tier when total request input exceeds this token count.",
        },
        **_MODEL_COST_RATES_PROPERTIES,
    },
}
MODEL_COST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": list(_MODEL_COST_RATES_PROPERTIES),
    "properties": {
        **_MODEL_COST_RATES_PROPERTIES,
        "tiers": {
            "type": "array",
            "items": MODEL_COST_TIER_SCHEMA,
            "description": "Request-wide pricing tiers. The highest matching input threshold applies to the full request.",
        },
    },
}


def _model_image_resize_options(**options: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "maxWidth": {"type": "integer", "minimum": 1},
            "maxHeight": {"type": "integer", "minimum": 1},
            "maxBytes": {
                "type": "integer",
                "minimum": 1,
                "description": "Maximum base64-encoded payload size in bytes.",
            },
            "jpegQuality": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        **options,
    }


MODEL_IMAGE_RESIZE_OPTIONS_SCHEMA = _model_image_resize_options()

MODEL_IMAGE_INPUT_LIMITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "resize": _model_image_resize_options(
            description="Cache-safe resize profile applied before a new image enters conversation history."
        ),
        "maxPerMessage": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum images accepted in one provider message.",
        },
        "maxPerRequest": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum images accepted across one provider request.",
        },
    },
}

MODEL_INPUT_LIMITS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "maxRequestBytes": {
            "type": "integer",
            "minimum": 1,
            "description": "Maximum serialized provider request size in bytes.",
        },
        "images": MODEL_IMAGE_INPUT_LIMITS_SCHEMA,
    },
}
