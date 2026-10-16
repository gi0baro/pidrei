"""Mirror of pi ai src/providers/compat-schema.ts.

Compatibility settings schemas: one per API, plus the API-agnostic superset
models.json validates provider and model `compat` against. pi builds them with
TypeBox; here they are the JSON Schema TypeBox emits, as plain dicts. The
compat value types themselves stay in `pidrei_ai.types`.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from .model_schema import MODEL_COST_SCHEMA


def _literals(*values: str, **options: Any) -> dict[str, Any]:
    return {"anyOf": [{"type": "string", "const": value} for value in values], **options}


def _boolean(**options: Any) -> dict[str, Any]:
    return {"type": "boolean", **options}


def _session_affinity_format(**options: Any) -> dict[str, Any]:
    return _literals("openai", "openai-nosession", "openrouter", **options)


SESSION_AFFINITY_FORMAT_SCHEMA = _session_affinity_format(
    description="Session-affinity header format used to route related requests consistently."
)


def _thinking_token_budget_field(**options: Any) -> dict[str, Any]:
    return _literals("thinking_token_budget", "thinking_budget", "thinking_budget_tokens", **options)


THINKING_TOKEN_BUDGET_FIELD_SCHEMA = _thinking_token_budget_field(
    description=(
        "Top-level request field name used by OpenAI-compatible endpoints to cap reasoning tokens. "
        '"thinking_token_budget" is used by vLLM, "thinking_budget" by Qwen, DashScope, or SGLang, and '
        '"thinking_budget_tokens" by llama.cpp.'
    )
)

CHAT_TEMPLATE_KWARG_VALUE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "number"},
        {"type": "boolean"},
        {"type": "null"},
        {
            "type": "object",
            "required": ["$var"],
            "properties": {
                "$var": _literals("thinking.enabled", "thinking.effort", "thinking.budget"),
                "omitWhenOff": {"type": "boolean"},
            },
        },
    ]
}


def _percentile_cutoffs(metric: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            percentile: {"type": "number", "description": f"{metric} at the {nth} percentile."}
            for percentile, nth in (("p50", "50th"), ("p75", "75th"), ("p90", "90th"), ("p99", "99th"))
        },
    }


def _number_or_string(description: str) -> dict[str, Any]:
    return {"anyOf": [{"type": "number"}, {"type": "string"}], "description": description}


def _string_array(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


OPENROUTER_ROUTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "allow_fallbacks": _boolean(description="Whether to allow backup providers to serve requests.", default=True),
        "require_parameters": _boolean(
            description="Whether to filter providers to only those that support all parameters in the request.",
            default=False,
        ),
        "data_collection": _literals(
            "deny",
            "allow",
            description=(
                'Data collection setting. "allow": allow providers that may store or train on data. "deny": only '
                "use providers that do not collect user data."
            ),
            default="allow",
        ),
        "zdr": _boolean(description="Whether to restrict routing to only ZDR (Zero Data Retention) endpoints."),
        "enforce_distillable_text": _boolean(
            description="Whether to restrict routing to only models that allow text distillation."
        ),
        "order": _string_array(
            "An ordered list of provider names or slugs to try in sequence, falling back to the next if unavailable."
        ),
        "only": _string_array("List of provider names or slugs to exclusively allow for this request."),
        "ignore": _string_array("List of provider names or slugs to skip for this request."),
        "quantizations": _string_array(
            'A list of quantization levels to filter providers by, for example ["fp16", "bf16", "fp8", "fp6", '
            '"int8", "int4", "fp4", "fp32"].'
        ),
        "sort": {
            "anyOf": [
                {"type": "string"},
                {
                    "type": "object",
                    "properties": {
                        "by": {
                            "type": "string",
                            "description": 'The sorting metric, such as "price", "throughput", or "latency".',
                        },
                        "partition": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": 'Partitioning strategy: "model" or "none".',
                            "default": "model",
                        },
                    },
                },
            ],
            "description": 'Sorting strategy. Can be a string such as "price", "throughput", or "latency", or an object.',
        },
        "max_price": {
            "type": "object",
            "properties": {
                "prompt": _number_or_string("Price per million prompt tokens."),
                "completion": _number_or_string("Price per million completion tokens."),
                "image": _number_or_string("Price per image."),
                "audio": _number_or_string("Price per audio unit."),
                "request": _number_or_string("Price per request."),
            },
            "description": "Maximum price per million tokens in USD.",
        },
        "preferred_min_throughput": {
            "anyOf": [{"type": "number"}, _percentile_cutoffs("Minimum tokens per second")],
            "description": "Preferred minimum throughput in tokens per second. A number applies to p50.",
        },
        "preferred_max_latency": {
            "anyOf": [{"type": "number"}, _percentile_cutoffs("Maximum latency in seconds")],
            "description": "Preferred maximum latency in seconds. A number applies to p50.",
        },
    },
    "description": (
        "OpenRouter provider routing preferences. Controls which upstream providers OpenRouter routes requests "
        "to. Sent as the provider field in the OpenRouter API request body. See "
        "https://openrouter.ai/docs/guides/routing/provider-selection."
    ),
}

VERCEL_GATEWAY_ROUTING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "only": _string_array(
            'List of provider slugs to exclusively use for this request, for example ["bedrock", "anthropic"].'
        ),
        "order": _string_array('List of provider slugs to try in order, for example ["anthropic", "openai"].'),
    },
    "description": (
        "Vercel AI Gateway routing preferences. Controls which upstream providers the gateway routes requests "
        "to. See https://vercel.com/docs/ai-gateway/models-and-providers/provider-options."
    ),
}

ANTHROPIC_ALLOWED_FALLBACK_MODEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["provider", "model", "cost"],
    "properties": {
        "provider": {"type": "string", "minLength": 1},
        "model": {"type": "string", "minLength": 1},
        "cost": MODEL_COST_SCHEMA,
    },
    "description": "An Anthropic server-side refusal fallback model with local pricing metadata.",
}

OPENAI_COMPLETIONS_COMPAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supportsStore": _boolean(
            description="Whether the provider supports the store field. Default: auto-detected from URL."
        ),
        "supportsDeveloperRole": _boolean(
            description=(
                "Whether the provider supports the developer role instead of system. Default: auto-detected from URL."
            )
        ),
        "supportsReasoningEffort": _boolean(
            description="Whether the provider supports reasoning_effort. Default: auto-detected from URL."
        ),
        "supportsUsageInStreaming": _boolean(
            description=(
                "Whether the provider supports stream_options.include_usage for token usage in streaming responses."
            ),
            default=True,
        ),
        "supportsFinishReason": _boolean(
            description=(
                "Whether streamed responses include finish_reason. When false, pidrei infers stop or toolUse when "
                "the stream ends."
            ),
            default=True,
        ),
        "maxTokensField": _literals(
            "max_completion_tokens",
            "max_tokens",
            description="Which field to use for max tokens. Default: auto-detected from URL.",
        ),
        "requiresToolResultName": _boolean(
            description="Whether tool results require the name field. Default: auto-detected from URL."
        ),
        "requiresAssistantAfterToolResult": _boolean(
            description=(
                "Whether a user message after tool results requires an assistant message in between. Default: "
                "auto-detected from URL."
            )
        ),
        "requiresThinkingAsText": _boolean(
            description=(
                "Whether thinking blocks must be converted to text blocks with <thinking> delimiters. Default: "
                "auto-detected from URL."
            )
        ),
        "requiresReasoningContentOnAssistantMessages": _boolean(
            description=(
                "Whether all replayed assistant messages must include an empty reasoning_content field when "
                "reasoning is enabled. Default: auto-detected from URL."
            )
        ),
        "thinkingFormat": _literals(
            "openai",
            "openrouter",
            "deepseek",
            "together",
            "baseten",
            "zai",
            "qwen",
            "chat-template",
            "qwen-chat-template",
            "string-thinking",
            "ant-ling",
            description=(
                "Format for reasoning or thinking parameters. When omitted, PiDrei auto-detects the format from "
                'the provider URL. "openai" uses reasoning_effort, "openrouter" uses reasoning.effort, "deepseek" '
                'uses thinking.type plus reasoning_effort when supported, "together" uses reasoning.enabled plus '
                'reasoning_effort when supported, "baseten" uses configurable chat_template_args plus '
                'reasoning_effort when supported, "zai" uses thinking.type, "qwen" uses top-level '
                'enable_thinking, "qwen-chat-template" uses chat_template_kwargs.enable_thinking and '
                'preserve_thinking, "chat-template" uses configurable chat_template_kwargs, "string-thinking" uses '
                'top-level thinking, and "ant-ling" uses reasoning.effort only when the mapped effort is non-null.'
            ),
        ),
        "chatTemplateKwargs": {
            "type": "object",
            "additionalProperties": CHAT_TEMPLATE_KWARG_VALUE_SCHEMA,
            "description": (
                'Kwargs sent as chat_template_kwargs when thinkingFormat is "chat-template". Use $var with '
                '"thinking.enabled", "thinking.effort", or "thinking.budget" for pidrei-controlled values.'
            ),
        },
        "chatTemplateArgs": {
            "type": "object",
            "additionalProperties": CHAT_TEMPLATE_KWARG_VALUE_SCHEMA,
            "description": (
                'Arguments sent as chat_template_args when thinkingFormat is "baseten". Use $var with '
                '"thinking.enabled", "thinking.effort", or "thinking.budget" for pidrei-controlled values.'
            ),
        },
        "openRouterRouting": OPENROUTER_ROUTING_SCHEMA,
        "vercelGatewayRouting": VERCEL_GATEWAY_ROUTING_SCHEMA,
        "zaiToolStream": _boolean(
            description="Whether z.ai supports top-level tool_stream for streaming tool call deltas.",
            default=False,
        ),
        "thinkingTokenBudgetField": _thinking_token_budget_field(
            description=(
                "Top-level request field used to cap reasoning tokens from thinkingBudgets. Reasoning and the "
                'answer share max_tokens on these endpoints. "thinking_token_budget" is vLLM, "thinking_budget" '
                'is Qwen, DashScope, or SGLang, and "thinking_budget_tokens" is llama.cpp. Off by default and not '
                "set on the generated catalog."
            )
        ),
        "supportsThinkingTokenBudget": _boolean(
            description=(
                'Alias for thinkingTokenBudgetField: "thinking_token_budget" (vLLM). Prefer thinkingTokenBudgetField.'
            ),
            default=False,
        ),
        "supportsOpenAIGrammarTools": _boolean(
            description=(
                "Whether the provider supports OpenAI custom tools with Lark or regex grammar formats. When false, "
                "grammar-constrained tools fall back to normal function tools. The generated catalog enables this "
                "for capable models."
            ),
            default=False,
        ),
        "supportsMidConvoSystemMessages": _boolean(
            description=(
                "Whether the exact model accepts system or developer messages after the conversation has started. "
                "When false, later system messages are folded into the leading system message. The generated "
                "catalog enables this for verified models."
            ),
            default=False,
        ),
        "supportsMidConvoToolAdditions": _boolean(
            description=(
                "Whether system messages can introduce additional tools mid-conversation. Requires "
                "supportsMidConvoSystemMessages. The generated catalog enables this for capable models."
            ),
            default=False,
        ),
        "supportsStrictMode": _boolean(
            description=(
                "Whether the provider supports the strict field in tool definitions. Generated capable models "
                "enable it explicitly."
            ),
            default=False,
        ),
        "cacheControlFormat": {
            "type": "string",
            "const": "anthropic",
            "description": (
                "Cache control convention for prompt caching. Anthropic applies cache_control markers to the "
                "system prompt, last tool definition, and last user, assistant, or tool-result text content."
            ),
        },
        "sendSessionAffinityHeaders": _boolean(
            description=(
                "Whether to send session-affinity data from options.sessionId. Default: true for OpenRouter "
                "endpoints, false otherwise."
            )
        ),
        "sessionAffinityFormat": _session_affinity_format(
            description=(
                "Session-affinity header format. openai sends session_id, x-client-request-id, and "
                "x-session-affinity; openai-nosession sends x-client-request-id and x-session-affinity; openrouter "
                "sends x-session-id. Does not affect prompt_cache_key. Default: auto-detected."
            )
        ),
        "supportsLongCacheRetention": _boolean(
            description=(
                'Whether the provider supports long prompt cache retention (prompt_cache_retention: "24h" or '
                'Anthropic-style cache_control.ttl: "1h", depending on format). Default: auto-detected from '
                "provider and URL."
            )
        ),
        "vllmPriority": {
            "type": "number",
            "description": (
                "vLLM scheduler priority sent as the top-level priority request field. Lower values are handled "
                "earlier and the server default is 0. Only meaningful with --scheduling-policy priority. Off by "
                "default and not set on the generated catalog."
            ),
        },
    },
    "description": (
        "Compatibility settings for OpenAI-compatible completions APIs. Use this to override URL-based "
        "auto-detection for custom providers."
    ),
    "additionalProperties": True,
}

OPENAI_RESPONSES_COMPAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supportsDeveloperRole": _boolean(
            description="Whether the provider supports the developer role instead of system.", default=True
        ),
        "supportsMidConvoSystemMessages": _boolean(
            description=(
                "Whether the exact model accepts developer or system messages after the conversation has started. "
                "When false, later system messages are folded into the leading system message. The generated "
                "catalog enables this for verified models."
            ),
            default=False,
        ),
        "sessionAffinityFormat": _session_affinity_format(
            description=(
                "Session-affinity header format. openai sends session_id and x-client-request-id; "
                "openai-nosession sends x-client-request-id; openrouter sends x-session-id. Does not affect "
                "prompt_cache_key. Default: auto-detected."
            )
        ),
        "supportsLongCacheRetention": _boolean(
            description=(
                'Whether the provider supports long prompt cache retention. This uses prompt_cache_options.ttl: "30m" '
                'on GPT-5.6+ and prompt_cache_retention: "24h" on earlier models.'
            ),
            default=True,
        ),
        "supportsStrictMode": _boolean(
            description=(
                "Whether the provider supports strict JSON-schema function tools. Defaults are API-specific; "
                "generated OpenAI models enable it explicitly."
            )
        ),
        "supportsOpenAIGrammarTools": _boolean(
            description=(
                "Whether to emit OpenAI custom tools with Lark or regex grammar formats. When false, "
                "grammar-constrained tools fall back to normal function tools. The generated catalog enables this "
                "for capable models."
            ),
            default=False,
        ),
        "supportsAdditionalTools": _boolean(
            description="Whether the model supports message-anchored additional_tools input items.", default=False
        ),
        "supportsToolSearch": _boolean(
            description="Whether the model supports client-executed tool search for transcript-anchored additions.",
            default=False,
        ),
        "supportsExplicitPromptCacheMode": _boolean(
            description="Whether the model accepts prompt_cache_options. Older OpenAI models reject the parameter.",
            default=False,
        ),
        "supportsMidConvoEffort": _boolean(
            description=(
                "Whether thinking level changes are sent as configuration_update input items instead of changing "
                "reasoning.effort, so a change keeps the prompt cache. Only used by the openai-responses API, and "
                "ignored with Sign in with ChatGPT on the openai provider."
            ),
            default=False,
        ),
        "supportsMaxOutputTokens": _boolean(
            description="Whether the provider accepts max_output_tokens. Some Codex-protocol gateways reject it.",
            default=True,
        ),
    },
    "description": "Compatibility settings for OpenAI Responses APIs.",
    "additionalProperties": True,
}

ANTHROPIC_MESSAGES_COMPAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supportsEagerToolInputStreaming": _boolean(
            description=(
                "Whether the provider accepts per-tool eager_input_streaming. When false, the Anthropic provider "
                "omits tools[].eager_input_streaming and sends the legacy fine-grained-tool-streaming-2025-05-14 "
                "beta header for tool-enabled requests."
            ),
            default=True,
        ),
        "supportsLongCacheRetention": _boolean(
            description="Whether the provider supports Anthropic long cache retention through cache_control.ttl.",
            default=True,
        ),
        "sendSessionAffinityHeaders": _boolean(
            description=(
                "Whether to send x-session-affinity from options.sessionId when caching is enabled. Required for "
                "providers like Fireworks that use session affinity for prompt cache routing; requests to the same "
                "replica maximize cache hits. Default: true for OpenRouter endpoints, false otherwise."
            )
        ),
        "sessionAffinityFormat": {
            "type": "string",
            "const": "openrouter",
            "description": "Session-affinity format. openrouter sends x-session-id; when unset, sends x-session-affinity.",
        },
        "supportsCacheControlOnTools": _boolean(
            description=(
                "Whether the provider supports Anthropic-style cache_control markers on tool definitions. When "
                "false, cache_control is omitted from tool parameters. Some Anthropic-compatible providers, such as "
                "Fireworks, do not support this field on tools and may reject or ignore it."
            ),
            default=True,
        ),
        "supportsTemperature": _boolean(
            description=(
                "Whether the model accepts the Anthropic temperature request field. Claude Opus 4.7+ rejects "
                "non-default values."
            ),
            default=True,
        ),
        "forceAdaptiveThinking": _boolean(
            description=(
                "Whether to force adaptive thinking (thinking.type: adaptive plus output_config.effort) regardless "
                "of model ID. Built-in models that require adaptive thinking set this in generated metadata. Custom "
                "Anthropic-compatible providers can set this to true for any model whose upstream requires the "
                "adaptive format. Set false to opt out on overridden built-in models."
            ),
            default=False,
        ),
        "allowEmptySignature": _boolean(
            description="Whether to replay empty thinking signatures instead of converting thinking to text.",
            default=False,
        ),
        "supportsStrictTools": _boolean(
            description=(
                "Whether the provider supports Anthropic strict tool schemas. Generated Anthropic models enable it "
                "explicitly."
            ),
            default=False,
        ),
        "supportsMidConvoEffort": _boolean(
            description=(
                "Whether the exact model transport supports effort-only system messages and thinking binding controls."
            ),
            default=False,
        ),
        "supportsMidConvoSystemMessages": _boolean(
            description=(
                "Whether the exact model accepts system-role messages inside the conversation. When false, later "
                "system messages are folded into the top-level system prompt."
            ),
            default=False,
        ),
        "supportsMidConvoToolChanges": _boolean(
            description=(
                "Whether the exact model accepts mid-conversation tool_addition and tool_removal blocks. Requires "
                "supportsMidConvoSystemMessages."
            ),
            default=False,
        ),
        "allowedFallbackModels": {
            "type": "array",
            "items": ANTHROPIC_ALLOWED_FALLBACK_MODEL_SCHEMA,
            "maxItems": 3,
            "description": (
                "Models Anthropic accepts for server-side refusal fallback, with local pricing metadata for "
                "returned fallback responses. When absent or empty, callers must omit fallbacks; Anthropic rejects "
                "the field for models with no permitted fallback targets."
            ),
        },
    },
    "description": "Compatibility settings for Anthropic Messages-compatible APIs.",
    "additionalProperties": True,
}

BEDROCK_COMPAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supportsStrictMode": _boolean(
            description="Whether the model supports Bedrock strict tool schemas.", default=False
        ),
    },
    "description": "Compatibility settings for Amazon Bedrock models.",
    "additionalProperties": True,
}

MISTRAL_CONVERSATIONS_COMPAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supportsMidConvoSystemMessages": _boolean(
            description=(
                "Whether the exact model accepts system messages after the conversation has started. When false, "
                "later system messages are folded into the leading system message."
            ),
            default=False,
        ),
    },
    "description": "Compatibility settings for the Mistral chat API.",
    "additionalProperties": True,
}

_PROVIDER_COMPAT_PROPERTY_OVERRIDES: dict[str, Any] = {
    "supportsDeveloperRole": _boolean(
        description="Whether the provider supports the developer role instead of system. Defaults are API-specific."
    ),
    "supportsMidConvoSystemMessages": _boolean(
        description=(
            "Whether the exact model accepts system or developer messages after the conversation has started. When "
            "false, later system messages are folded into the leading system message."
        ),
        default=False,
    ),
    "sessionAffinityFormat": _session_affinity_format(
        description="Session-affinity header format. Defaults are API-specific or auto-detected."
    ),
    "supportsMidConvoEffort": _boolean(
        description=(
            "Whether thinking level changes are sent inside the conversation instead of changing the request's "
            "effort, so a change keeps the prompt cache. Used by the anthropic-messages and openai-responses APIs."
        ),
        default=False,
    ),
    "supportsLongCacheRetention": _boolean(
        description="Whether the provider supports long prompt cache retention. Defaults are API-specific or auto-detected."
    ),
    "supportsStrictMode": _boolean(
        description="Whether the provider supports strict tool schemas. Defaults are API-specific."
    ),
    "supportsOpenAIGrammarTools": _boolean(
        description=(
            "Whether the provider supports OpenAI custom tools with Lark or regex grammar formats. When false, "
            "grammar-constrained tools fall back to normal function tools."
        ),
        default=False,
    ),
    "sendSessionAffinityHeaders": _boolean(
        description="Whether to send session-affinity data from options.sessionId. Defaults are API-specific."
    ),
}


def _merge_compat_properties(
    property_groups: Sequence[Mapping[str, Any]], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Provider-level models.json compatibility is API-agnostic, so it needs one superset. Every
    duplicate property requires explicit generic metadata, while API-specific schemas retain their
    own defaults and descriptions."""
    merged: dict[str, Any] = {}
    duplicate_names: set[str] = set()

    for properties in property_groups:
        for name, schema in properties.items():
            if name in merged:
                duplicate_names.add(name)
                if name not in overrides:
                    raise ValueError(f"Duplicate compatibility schema property requires an override: {name}")
            merged[name] = schema

    for name in overrides:
        if name not in duplicate_names:
            raise ValueError(f"Compatibility schema override does not resolve a duplicate property: {name}")

    return {**merged, **overrides}


_COMPAT_SCHEMAS_BY_API = {
    "openai-completions": OPENAI_COMPLETIONS_COMPAT_SCHEMA,
    "openai-responses": OPENAI_RESPONSES_COMPAT_SCHEMA,
    "anthropic-messages": ANTHROPIC_MESSAGES_COMPAT_SCHEMA,
    "bedrock-converse-stream": BEDROCK_COMPAT_SCHEMA,
    "mistral-conversations": MISTRAL_CONVERSATIONS_COMPAT_SCHEMA,
}

PROVIDER_COMPAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": _merge_compat_properties(
        [schema["properties"] for schema in _COMPAT_SCHEMAS_BY_API.values()], _PROVIDER_COMPAT_PROPERTY_OVERRIDES
    ),
    "description": "Provider and model compatibility overrides.",
    "additionalProperties": True,
}
