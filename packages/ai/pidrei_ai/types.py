"""Port of pi's core types (packages/ai/src/types.ts).

Attribute names are snake_case; every observable string *value* (stop reasons,
roles, event type tags, thinking levels, …) is byte-identical to pi's, since
those appear in the event protocol and in serialized sessions.

TypeBox tool schemas are plain JSON Schema dicts here; pi's validation/coercion
semantics are ported separately (`utils/validation`, Phase 1).
"""

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypedDict

from pidrei_ai.utils.cancel import CancelToken


type KnownApi = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-vertex",
    "pi-messages",
]
type Api = str  # KnownApi or any custom string

type KnownImagesApi = Literal["openrouter-images"]
type ImagesApi = str  # KnownImagesApi or any custom string

type KnownProvider = Literal[
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "google",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "nvidia",
    "deepseek",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "zai-coding-cn",
    "mistral",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "huggingface",
    "fireworks",
    "together",
    "opencode",
    "opencode-go",
    "kimi-coding",
    "cloudflare-workers-ai",
    "cloudflare-ai-gateway",
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "xiaomi",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-sgp",
]
type ProviderId = str  # KnownProvider or any custom string

type KnownImagesProvider = Literal["openrouter"]
type ImagesProviderId = str  # KnownImagesProvider or any custom string

type ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh", "max"]
type ModelThinkingLevel = Literal["off", "minimal", "low", "medium", "high", "xhigh", "max"]
# Maps pi thinking levels to provider/model-specific values; None marks a level unsupported.
type ThinkingLevelMap = Mapping[str, str | None]
type ChatTemplateKwargValue = str | float | bool | None | dict[str, Any]

type CacheRetention = Literal["none", "short", "long"]
type Transport = Literal["sse", "websocket", "websocket-cached", "auto"]
type SessionAffinityFormat = Literal["openai", "openai-nosession", "openrouter"]

# Provider-scoped environment overrides. Values take precedence over os.environ.
type ProviderEnv = Mapping[str, str]
# A None value suppresses a provider/API default header with the same name.
type ProviderHeaders = Mapping[str, str | None]

type StopReason = Literal["pending", "stop", "length", "toolUse", "error", "aborted"]


class TextSignatureV1(TypedDict, total=False):
    v: Literal[1]
    id: str
    phase: Literal["commentary", "final_answer"]


@dataclass(slots=True)
class ThinkingBudgets:
    """Token budgets for each thinking level (token-based providers only)."""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


# --- content blocks -----------------------------------------------------------


@dataclass(slots=True)
class TextContent:
    text: str
    # e.g., for OpenAI responses, message metadata (legacy id string or TextSignatureV1 JSON)
    text_signature: str | None = None
    type: Literal["text"] = "text"


@dataclass(slots=True)
class ThinkingContent:
    thinking: str
    thinking_signature: str | None = None  # e.g., for OpenAI responses, the reasoning item ID
    # When True, the thinking content was redacted by safety filters. The opaque encrypted
    # payload is stored in `thinking_signature` so it can be replayed for multi-turn continuity.
    redacted: bool = False
    type: Literal["thinking"] = "thinking"


@dataclass(slots=True)
class ImageContent:
    data: str  # base64 encoded image data
    mime_type: str  # e.g., "image/jpeg", "image/png"
    type: Literal["image"] = "image"


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None  # Google-specific: opaque signature for thought context reuse
    type: Literal["toolCall"] = "toolCall"


type UserContent = TextContent | ImageContent
type AssistantContent = TextContent | ThinkingContent | ToolCall
type ToolResultContent = TextContent | ImageContent


# --- usage / cost -------------------------------------------------------------


@dataclass(slots=True)
class UsageCost:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


@dataclass(slots=True)
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    # Subset of `cache_write` written with 1h retention. Only Anthropic reports this split.
    cache_write_1h: int | None = None
    # Reasoning/thinking tokens when the provider reports them; a subset of `output`.
    # Set (possibly 0) by providers that expose a breakdown, None by providers that don't.
    reasoning: int | None = None
    total_tokens: int = 0
    cost: UsageCost = field(default_factory=UsageCost)


# --- diagnostics --------------------------------------------------------------


@dataclass(slots=True)
class DiagnosticErrorInfo:
    message: str
    name: str | None = None
    stack: str | None = None
    code: str | int | None = None


@dataclass(slots=True)
class AssistantMessageDiagnostic:
    """Redacted provider/runtime diagnostics for failures and recoveries."""

    type: str
    timestamp: int
    error: DiagnosticErrorInfo | None = None
    details: dict[str, Any] | None = None


# --- messages -----------------------------------------------------------------


@dataclass(slots=True)
class UserMessage:
    content: str | list[UserContent]
    timestamp: int  # Unix timestamp in milliseconds
    role: Literal["user"] = "user"


@dataclass(slots=True)
class AssistantMessage:
    content: list[AssistantContent]
    api: Api
    provider: ProviderId
    model: str
    usage: Usage
    stop_reason: StopReason
    timestamp: int  # Unix timestamp in milliseconds
    # Concrete response model when different from the requested one (e.g. OpenRouter `auto`).
    response_model: str | None = None
    response_id: str | None = None  # Provider-specific response/message identifier
    diagnostics: list[AssistantMessageDiagnostic] | None = None
    error_message: str | None = None
    raw_stop_reason: str | None = None
    role: Literal["assistant"] = "assistant"


@dataclass(slots=True)
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[ToolResultContent]  # Supports text and images
    is_error: bool
    timestamp: int  # Unix timestamp in milliseconds
    details: Any = None
    # Usage from the tool execution itself, if available. Not part of main LLM context accounting.
    usage: Usage | None = None
    # Names from `Context.tools` that became available after this result. Providers with
    # native deferred tool loading use this as the load point; others ignore it.
    added_tool_names: list[str] | None = None
    role: Literal["toolResult"] = "toolResult"


type Message = UserMessage | AssistantMessage | ToolResultMessage


# --- tools / context ----------------------------------------------------------


type GrammarFormat = Literal["openai_lark", "openai_regex"]
type GrammarVariants = Mapping[str, str]


@dataclass(slots=True)
class JsonSchemaConstrainedSampling:
    strict: Literal["prefer", "require"]
    type: Literal["json_schema"] = "json_schema"


@dataclass(slots=True)
class GrammarConstrainedSampling:
    variants: GrammarVariants
    type: Literal["grammar"] = "grammar"


# `False` disables constrained sampling explicitly (pi: `constrainedSampling?: false | ...`).
type ConstrainedSamplingConfig = JsonSchemaConstrainedSampling | GrammarConstrainedSampling


@dataclass(slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema (pi: TypeBox TSchema)
    constrained_sampling: ConstrainedSamplingConfig | Literal[False] | None = None


@dataclass(slots=True)
class Context:
    messages: list[Message] = field(default_factory=list)
    system_prompt: str | None = None
    tools: list[Tool] | None = None


# --- compat matrices ----------------------------------------------------------


class OpenRouterRouting(TypedDict, total=False):
    """OpenRouter provider routing preferences, sent verbatim as the `provider` body field."""

    allow_fallbacks: bool
    require_parameters: bool
    data_collection: Literal["deny", "allow"]
    zdr: bool
    enforce_distillable_text: bool
    order: list[str]
    only: list[str]
    ignore: list[str]
    quantizations: list[str]
    sort: str | dict[str, Any]
    max_price: dict[str, float | str]
    preferred_min_throughput: float | dict[str, float]
    preferred_max_latency: float | dict[str, float]


class VercelGatewayRouting(TypedDict, total=False):
    """Vercel AI Gateway routing preferences."""

    only: list[str]
    order: list[str]


type ThinkingFormat = Literal[
    "openai",
    "openrouter",
    "deepseek",
    "together",
    "zai",
    "qwen",
    "chat-template",
    "qwen-chat-template",
    "string-thinking",
    "ant-ling",
]


@dataclass(slots=True)
class OpenAICompletionsCompat:
    """Compatibility settings for OpenAI-compatible completions APIs.

    `None` fields mean "auto-detect from baseUrl" (or the documented default),
    exactly like an absent key in pi's TS objects.
    """

    supports_store: bool | None = None
    supports_developer_role: bool | None = None
    supports_reasoning_effort: bool | None = None
    supports_usage_in_streaming: bool | None = None  # default True
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] | None = None
    requires_tool_result_name: bool | None = None
    requires_assistant_after_tool_result: bool | None = None
    requires_thinking_as_text: bool | None = None
    requires_reasoning_content_on_assistant_messages: bool | None = None
    thinking_format: ThinkingFormat | None = None  # default "openai"
    chat_template_kwargs: dict[str, ChatTemplateKwargValue] | None = None
    open_router_routing: OpenRouterRouting | None = None
    vercel_gateway_routing: VercelGatewayRouting | None = None
    zai_tool_stream: bool | None = None  # default False
    supports_openai_grammar_tools: bool | None = None  # default False
    supports_strict_mode: bool | None = None  # default True
    cache_control_format: Literal["anthropic"] | None = None
    send_session_affinity_headers: bool | None = None  # default False
    deferred_tools_mode: Literal["kimi"] | None = None
    session_affinity_format: SessionAffinityFormat | None = None
    supports_long_cache_retention: bool | None = None  # default True


@dataclass(slots=True)
class OpenAIResponsesCompat:
    """Compatibility settings for OpenAI Responses APIs."""

    supports_developer_role: bool | None = None  # default True
    session_affinity_format: SessionAffinityFormat | None = None
    supports_long_cache_retention: bool | None = None  # default True
    supports_strict_mode: bool | None = None
    supports_openai_grammar_tools: bool | None = None  # default False
    supports_tool_search: bool | None = None  # default False
    supports_explicit_prompt_cache_mode: bool | None = None  # default False


@dataclass(slots=True)
class AnthropicMessagesCompat:
    """Compatibility settings for Anthropic Messages-compatible APIs."""

    # When False, omit `tools[].eager_input_streaming` and send the legacy
    # fine-grained-tool-streaming beta header for tool-enabled requests. Default True.
    supports_eager_tool_input_streaming: bool | None = None
    supports_long_cache_retention: bool | None = None  # default True
    # Send `x-session-affinity` from `options.session_id` when caching is enabled
    # (required by e.g. Fireworks for prompt-cache routing). Default False.
    send_session_affinity_headers: bool | None = None
    supports_cache_control_on_tools: bool | None = None  # default True
    supports_temperature: bool | None = None  # default True; Opus 4.7+ rejects non-default
    # Force adaptive thinking (`thinking.type: "adaptive"` + `output_config.effort`)
    # regardless of the model id. Default False.
    force_adaptive_thinking: bool | None = None
    # Replay empty thinking signatures as `signature: ""` instead of converting to text.
    allow_empty_signature: bool | None = None  # default False
    supports_strict_tools: bool | None = None  # default False
    # Deferred tools loaded by `tool_reference` blocks in tool results. Default True for
    # first-party Anthropic models except Haiku and pre-4.5 models; False elsewhere.
    supports_tool_references: bool | None = None


@dataclass(slots=True)
class BedrockCompat:
    """Compatibility settings for Amazon Bedrock models."""

    supports_strict_mode: bool | None = None  # default False


type ModelCompat = OpenAICompletionsCompat | OpenAIResponsesCompat | AnthropicMessagesCompat | BedrockCompat


# --- model / cost -------------------------------------------------------------


@dataclass(slots=True)
class ModelCostTier:
    input: float  # $/million tokens
    output: float
    cache_read: float
    cache_write: float
    # Use this tier for requests whose total input usage exceeds this token count.
    input_tokens_above: int = 0


@dataclass(slots=True)
class ModelCost:
    input: float = 0.0  # $/million tokens
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    # Request-wide pricing tiers; the highest matching input threshold applies to the full request.
    tiers: list[ModelCostTier] | None = None


@dataclass(slots=True)
class Model:
    id: str
    name: str
    api: Api
    provider: ProviderId
    base_url: str
    reasoning: bool
    input: list[Literal["text", "image"]]
    cost: ModelCost
    context_window: int
    max_tokens: int
    thinking_level_map: ThinkingLevelMap | None = None
    headers: dict[str, str] | None = None
    # Compatibility overrides; when None, auto-detected from base_url (API-specific).
    compat: ModelCompat | None = None


# --- stream options / provider contracts --------------------------------------


@dataclass(slots=True)
class ProviderResponse:
    status: int
    headers: dict[str, str]


# Optional callbacks; awaitable-returning by contract (pi declares
# `T | Promise<T>` unions here, but two shapes per slot is what let dropped
# coroutines hide — decided 2026-07-28: callback contracts are async-only).
type OnPayload = Callable[[Any, Model], Awaitable[Any]]
type OnResponse = Callable[[ProviderResponse, Model], Awaitable[Any]]


@dataclass(slots=True)
class StreamOptions:
    temperature: float | None = None
    max_tokens: int | None = None
    cancel: CancelToken | None = None  # pi: `signal?: AbortSignal`
    api_key: str | None = None
    # Preferred transport for providers that support multiple transports.
    transport: Transport | None = None
    # Prompt cache retention preference; providers map this to their supported values.
    cache_retention: CacheRetention | None = None  # default "short"
    # Session identifier for providers supporting session-based caching/routing.
    session_id: str | None = None
    # Inspect or replace the provider payload before sending (return None to keep unchanged).
    on_payload: OnPayload | None = None
    # Invoked after an HTTP response is received, before its body stream is consumed.
    on_response: OnResponse | None = None
    # Merged with provider defaults; caller values override; None suppresses a default header.
    headers: ProviderHeaders | None = None
    timeout_ms: float | None = None
    # WebSocket connect timeout (connection/open handshake only; stream idleness uses timeout_ms).
    websocket_connect_timeout_ms: float | None = None
    max_retries: int | None = None
    # Cap on server-requested retry delays; beyond it the request fails immediately so
    # higher-level retry logic can surface it. Default 60000; 0 disables the cap.
    max_retry_delay_ms: float | None = None
    # Provider-extracted metadata (e.g. Anthropic `user_id`).
    metadata: dict[str, Any] | None = None
    # Provider-scoped environment values, taking precedence over os.environ.
    env: ProviderEnv | None = None
    # pi's ProviderStreamOptions allows arbitrary extra provider options.
    extra: dict[str, Any] | None = None
    # Models-only (pi: ModelsStreamTransforms.transformHeaders): transform fully
    # assembled model/auth/request headers before provider dispatch.
    # Awaitable-returning; stripped before options reach the provider.
    transform_headers: Callable[[ProviderHeaders], Awaitable[ProviderHeaders]] | None = None


@dataclass(slots=True)
class SimpleStreamOptions(StreamOptions):
    """Unified options with reasoning, passed to `stream_simple`/`complete_simple`."""

    reasoning: ThinkingLevel | None = None
    # Custom token budgets for thinking levels (token-based providers only).
    thinking_budgets: ThinkingBudgets | None = None


# --- event protocol -----------------------------------------------------------


@dataclass(slots=True)
class StartEvent:
    partial: AssistantMessage
    type: Literal["start"] = "start"


@dataclass(slots=True)
class TextStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["text_start"] = "text_start"


@dataclass(slots=True)
class TextDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["text_delta"] = "text_delta"


@dataclass(slots=True)
class TextEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["text_end"] = "text_end"


@dataclass(slots=True)
class ThinkingStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["thinking_start"] = "thinking_start"


@dataclass(slots=True)
class ThinkingDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["thinking_delta"] = "thinking_delta"


@dataclass(slots=True)
class ThinkingEndEvent:
    content_index: int
    content: str
    partial: AssistantMessage
    type: Literal["thinking_end"] = "thinking_end"


@dataclass(slots=True)
class ToolCallStartEvent:
    content_index: int
    partial: AssistantMessage
    type: Literal["toolcall_start"] = "toolcall_start"


@dataclass(slots=True)
class ToolCallDeltaEvent:
    content_index: int
    delta: str
    partial: AssistantMessage
    type: Literal["toolcall_delta"] = "toolcall_delta"


@dataclass(slots=True)
class ToolCallEndEvent:
    content_index: int
    tool_call: ToolCall
    partial: AssistantMessage
    type: Literal["toolcall_end"] = "toolcall_end"


@dataclass(slots=True)
class DoneEvent:
    reason: Literal["stop", "length", "toolUse"]
    message: AssistantMessage
    type: Literal["done"] = "done"


@dataclass(slots=True)
class ErrorEvent:
    reason: Literal["aborted", "error"]
    error: AssistantMessage
    type: Literal["error"] = "error"


type AssistantMessageEvent = (
    StartEvent
    | TextStartEvent
    | TextDeltaEvent
    | TextEndEvent
    | ThinkingStartEvent
    | ThinkingDeltaEvent
    | ThinkingEndEvent
    | ToolCallStartEvent
    | ToolCallDeltaEvent
    | ToolCallEndEvent
    | DoneEvent
    | ErrorEvent
)


class ProviderStreams(Protocol):
    """The uniform stream contract of an API implementation module.

    Every module under `pidrei_ai/api/` exports exactly `stream` and
    `stream_simple`, so the module itself satisfies this protocol.
    """

    def stream(self, model: Model, context: Context, options: StreamOptions | None = None) -> Any: ...

    def stream_simple(self, model: Model, context: Context, options: SimpleStreamOptions | None = None) -> Any: ...


# StreamFunction contract (pi types.ts:303-315): must return an
# AssistantMessageEventStream; once invoked, failures are encoded in the stream
# (an `error` event with stopReason "error"/"aborted"), never thrown.
type StreamFunction = Callable[..., Any]


# --- images ------------------------------------------------------------------

type ImagesInputContent = TextContent | ImageContent
type ImagesOutputContent = TextContent | ImageContent
type ImagesStopReason = Literal["stop", "error", "aborted"]


@dataclass(slots=True)
class ImagesModel:
    """`Model` minus the chat-only fields, plus the produced modalities."""

    id: str
    name: str
    api: ImagesApi
    provider: ImagesProviderId
    base_url: str
    input: list[Literal["text", "image"]]
    output: list[Literal["text", "image"]]
    cost: ModelCost
    headers: dict[str, str] | None = None


@dataclass(slots=True)
class ImagesContext:
    input: list[ImagesInputContent] = field(default_factory=list)


@dataclass(slots=True)
class AssistantImages:
    api: ImagesApi
    provider: ImagesProviderId
    model: str
    output: list[ImagesOutputContent]
    stop_reason: ImagesStopReason
    timestamp: int  # Unix timestamp in milliseconds
    response_id: str | None = None
    usage: Usage | None = None
    error_message: str | None = None


@dataclass(slots=True)
class ImagesOptions:
    cancel: CancelToken | None = None  # pi: `signal?: AbortSignal`
    api_key: str | None = None
    # Provider-scoped environment values, taking precedence over os.environ.
    env: ProviderEnv | None = None
    # Inspect or replace the provider payload before sending (return None to keep unchanged).
    on_payload: OnPayload | None = None
    # Invoked after an HTTP response is received.
    on_response: OnResponse | None = None
    # Merged with provider defaults; caller values override; None suppresses a default header.
    headers: ProviderHeaders | None = None
    timeout_ms: float | None = None
    max_retries: int | None = None
    max_retry_delay_ms: float | None = None
    metadata: dict[str, Any] | None = None


class ProviderImages(Protocol):
    async def generate_images(
        self, model: ImagesModel, context: ImagesContext, options: ImagesOptions | None = None
    ) -> AssistantImages: ...
