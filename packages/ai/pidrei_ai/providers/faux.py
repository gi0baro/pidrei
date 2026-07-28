"""Port of pi's faux test provider (packages/ai/src/providers/faux.ts).

A fully scripted streaming provider for tests: queued responses (messages or
factories), delta streaming with configurable token chunking and optional
token/second pacing, usage estimation from serialized context, and prompt-cache
simulation per session id.

pi additionally registers faux cores into the deprecated compat api-registry
(`registerFauxProvider`); pidrei uses explicit `Models` collections, so
`faux_provider()` returning a `Provider` is the only wiring.
"""

import copy
import json
import random
import secrets
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

import tonio.colored as tonio
from tonio.colored import time as tonio_time

from pidrei_ai.auth.types import ApiKeyAuth, AuthResult, ModelAuth, ProviderAuth
from pidrei_ai.registry import Provider, create_provider
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    DoneEvent,
    ErrorEvent,
    Message,
    Model,
    ModelCost,
    ProviderResponse,
    SimpleStreamOptions,
    StartEvent,
    StopReason,
    StreamOptions,
    TextContent,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingContent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    ToolCall,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultMessage,
    Usage,
)
from pidrei_ai.utils.estimate import _tool_json_shape
from pidrei_ai.utils.event_stream import AssistantMessageEventStream


DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"
DEFAULT_MIN_TOKEN_SIZE = 3
DEFAULT_MAX_TOKEN_SIZE = 5

type FauxContentBlock = TextContent | ThinkingContent | ToolCall
# A queued step: a ready message, or a factory (context, options, state, model)
# returning an awaitable of AssistantMessage (async-only callback policy; pi's
# `FauxResponseFactory` is a `T | Promise<T>` union).
type FauxResponseStep = AssistantMessage | Callable[..., Awaitable[AssistantMessage]]


class FauxState:
    __slots__ = ("call_count",)

    def __init__(self) -> None:
        self.call_count = 0


@dataclass(slots=True)
class FauxModelDefinition:
    id: str
    name: str | None = None
    reasoning: bool | None = None
    input: list | None = None
    cost: ModelCost | None = None
    context_window: int | None = None
    max_tokens: int | None = None


def _random_id(prefix: str) -> str:
    return f"{prefix}:{int(time.time() * 1000)}:{secrets.token_hex(6)}"


def faux_text(text: str) -> TextContent:
    return TextContent(text=text)


def faux_thinking(thinking: str) -> ThinkingContent:
    return ThinkingContent(thinking=thinking)


def faux_tool_call(name: str, arguments: dict, *, id: str | None = None) -> ToolCall:
    return ToolCall(id=id if id is not None else _random_id("tool"), name=name, arguments=arguments)


def _normalize_content(content: str | FauxContentBlock | list[FauxContentBlock]) -> list[FauxContentBlock]:
    if isinstance(content, str):
        return [faux_text(content)]
    return content if isinstance(content, list) else [content]


def faux_assistant_message(
    content: str | FauxContentBlock | list[FauxContentBlock],
    *,
    stop_reason: StopReason = "stop",
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=_normalize_content(content),
        api=DEFAULT_API,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL_ID,
        usage=Usage(),
        stop_reason=stop_reason,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp if timestamp is not None else int(time.time() * 1000),
    )


# --- context serialization / usage estimation ---------------------------------


def _estimate_tokens(text: str) -> int:
    return -(-len(text) // 4)


def _content_to_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if block.type == "text":
            parts.append(block.text)
        else:
            parts.append(f"[image:{block.mime_type}:{len(block.data)}]")
    return "\n".join(parts)


def _assistant_content_to_text(content: list) -> str:
    parts = []
    for block in content:
        if block.type == "text":
            parts.append(block.text)
        elif block.type == "thinking":
            parts.append(block.thinking)
        else:
            parts.append(f"{block.name}:{json.dumps(block.arguments, separators=(',', ':'), ensure_ascii=False)}")
    return "\n".join(parts)


def _tool_result_to_text(message: ToolResultMessage) -> str:
    return "\n".join([message.tool_name, *(_content_to_text([block]) for block in message.content)])


def _message_to_text(message: Message) -> str:
    if message.role == "user":
        return _content_to_text(message.content)
    if message.role == "assistant":
        return _assistant_content_to_text(message.content)
    return _tool_result_to_text(message)


def serialize_faux_context(context: Context) -> str:
    parts: list[str] = []
    if context.system_prompt:
        parts.append(f"system:{context.system_prompt}")
    for message in context.messages:
        parts.append(f"{message.role}:{_message_to_text(message)}")
    if context.tools:
        tools_json = json.dumps(
            [_tool_json_shape(tool) for tool in context.tools], separators=(",", ":"), ensure_ascii=False
        )
        parts.append(f"tools:{tools_json}")
    return "\n\n".join(parts)


def _common_prefix_length(a: str, b: str) -> int:
    length = min(len(a), len(b))
    index = 0
    while index < length and a[index] == b[index]:
        index += 1
    return index


# --- streaming ----------------------------------------------------------------


def _split_string_by_token_size(text: str, min_token_size: int, max_token_size: int) -> list[str]:
    chunks: list[str] = []
    index = 0
    while index < len(text):
        token_size = random.randint(min_token_size, max_token_size)  # noqa: S311
        char_size = max(1, token_size * 4)
        chunks.append(text[index : index + char_size])
        index += char_size
    return chunks if chunks else [""]


def _create_aborted_message(partial: AssistantMessage) -> AssistantMessage:
    return replace(
        partial, stop_reason="aborted", error_message="Request was aborted", timestamp=int(time.time() * 1000)
    )


async def _schedule_chunk(chunk: str, tokens_per_second: float | None) -> None:
    if not tokens_per_second or tokens_per_second <= 0:
        await tonio.yield_now()
        return
    await tonio_time.sleep(_estimate_tokens(chunk) / tokens_per_second)


async def _stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_token_size: int,
    max_token_size: int,
    tokens_per_second: float | None,
    cancel,
) -> None:
    partial = replace(message, content=[])

    def aborted_now() -> bool:
        return cancel is not None and cancel.cancelled

    def emit_aborted() -> None:
        aborted = _create_aborted_message(partial)
        stream.push(ErrorEvent(reason="aborted", error=aborted))
        stream.end(aborted)

    if aborted_now():
        emit_aborted()
        return

    stream.push(StartEvent(partial=replace(partial)))

    for index, block in enumerate(message.content):
        if aborted_now():
            emit_aborted()
            return

        if block.type == "thinking":
            partial.content = [*partial.content, ThinkingContent(thinking="")]
            stream.push(ThinkingStartEvent(content_index=index, partial=replace(partial)))
            for chunk in _split_string_by_token_size(block.thinking, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if aborted_now():
                    emit_aborted()
                    return
                partial.content[index].thinking += chunk
                stream.push(ThinkingDeltaEvent(content_index=index, delta=chunk, partial=replace(partial)))
            stream.push(ThinkingEndEvent(content_index=index, content=block.thinking, partial=replace(partial)))
            continue

        if block.type == "text":
            partial.content = [*partial.content, TextContent(text="")]
            stream.push(TextStartEvent(content_index=index, partial=replace(partial)))
            for chunk in _split_string_by_token_size(block.text, min_token_size, max_token_size):
                await _schedule_chunk(chunk, tokens_per_second)
                if aborted_now():
                    emit_aborted()
                    return
                partial.content[index].text += chunk
                stream.push(TextDeltaEvent(content_index=index, delta=chunk, partial=replace(partial)))
            stream.push(TextEndEvent(content_index=index, content=block.text, partial=replace(partial)))
            continue

        partial.content = [*partial.content, ToolCall(id=block.id, name=block.name, arguments={})]
        stream.push(ToolCallStartEvent(content_index=index, partial=replace(partial)))
        arguments_json = json.dumps(block.arguments, separators=(",", ":"), ensure_ascii=False)
        for chunk in _split_string_by_token_size(arguments_json, min_token_size, max_token_size):
            await _schedule_chunk(chunk, tokens_per_second)
            if aborted_now():
                emit_aborted()
                return
            stream.push(ToolCallDeltaEvent(content_index=index, delta=chunk, partial=replace(partial)))
        partial.content[index].arguments = block.arguments
        stream.push(ToolCallEndEvent(content_index=index, tool_call=block, partial=replace(partial)))

    if message.stop_reason in ("error", "aborted"):
        stream.push(ErrorEvent(reason=message.stop_reason, error=message))
        stream.end(message)
        return

    stream.push(DoneEvent(reason=message.stop_reason, message=message))
    stream.end(message)


# --- core ---------------------------------------------------------------------


class FauxCore:
    """The scripted streaming engine; also satisfies the ProviderStreams shape."""

    def __init__(
        self,
        *,
        api: str | None = None,
        provider: str | None = None,
        models: list[FauxModelDefinition] | None = None,
        tokens_per_second: float | None = None,
        token_size_min: int | None = None,
        token_size_max: int | None = None,
    ):
        self.api = api if api is not None else _random_id(DEFAULT_API)
        self.provider_id = provider if provider is not None else DEFAULT_PROVIDER
        requested_min = token_size_min if token_size_min is not None else DEFAULT_MIN_TOKEN_SIZE
        requested_max = token_size_max if token_size_max is not None else DEFAULT_MAX_TOKEN_SIZE
        self._min_token_size = max(1, min(requested_min, requested_max))
        self._max_token_size = max(self._min_token_size, requested_max)
        self._tokens_per_second = tokens_per_second
        self.state = FauxState()
        self._guard = threading.Lock()
        self._pending_responses: list[FauxResponseStep] = []
        self._prompt_cache: dict[str, str] = {}

        definitions = models if models else [FauxModelDefinition(id=DEFAULT_MODEL_ID, name=DEFAULT_MODEL_NAME)]
        self.models: list[Model] = [
            Model(
                id=definition.id,
                name=definition.name if definition.name is not None else definition.id,
                api=self.api,
                provider=self.provider_id,
                base_url=DEFAULT_BASE_URL,
                reasoning=definition.reasoning if definition.reasoning is not None else False,
                input=list(definition.input) if definition.input is not None else ["text", "image"],
                cost=definition.cost if definition.cost is not None else ModelCost(),
                context_window=definition.context_window if definition.context_window is not None else 128000,
                max_tokens=definition.max_tokens if definition.max_tokens is not None else 16384,
            )
            for definition in definitions
        ]

    # -- queue management ------------------------------------------------------

    def set_responses(self, responses: list[FauxResponseStep]) -> None:
        with self._guard:
            self._pending_responses = list(responses)

    def append_responses(self, responses: list[FauxResponseStep]) -> None:
        with self._guard:
            self._pending_responses.extend(responses)

    def get_pending_response_count(self) -> int:
        with self._guard:
            return len(self._pending_responses)

    def get_model(self, model_id: str | None = None) -> Model | None:
        if model_id is None:
            return self.models[0]
        return next((candidate for candidate in self.models if candidate.id == model_id), None)

    # -- usage simulation ------------------------------------------------------

    def _with_usage_estimate(
        self,
        message: AssistantMessage,
        context: Context,
        options: StreamOptions | None,
    ) -> AssistantMessage:
        prompt_text = serialize_faux_context(context)
        prompt_tokens = _estimate_tokens(prompt_text)
        output_tokens = _estimate_tokens(_assistant_content_to_text(message.content))
        input_tokens = prompt_tokens
        cache_read = 0
        cache_write = 0
        session_id = options.session_id if options is not None else None

        if session_id and (options is None or options.cache_retention != "none"):
            with self._guard:
                previous_prompt = self._prompt_cache.get(session_id)
                if previous_prompt:
                    cached_chars = _common_prefix_length(previous_prompt, prompt_text)
                    cache_read = _estimate_tokens(previous_prompt[:cached_chars])
                    cache_write = _estimate_tokens(prompt_text[cached_chars:])
                    input_tokens = max(0, prompt_tokens - cache_read)
                else:
                    cache_write = prompt_tokens
                self._prompt_cache[session_id] = prompt_text

        return replace(
            message,
            usage=Usage(
                input=input_tokens,
                output=output_tokens,
                cache_read=cache_read,
                cache_write=cache_write,
                total_tokens=input_tokens + output_tokens + cache_read + cache_write,
            ),
        )

    def _clone_message(self, message: AssistantMessage, model_id: str) -> AssistantMessage:
        cloned = copy.deepcopy(message)
        return replace(cloned, api=self.api, provider=self.provider_id, model=model_id)

    def _create_error_message(self, error: Any, model_id: str) -> AssistantMessage:
        return AssistantMessage(
            content=[],
            api=self.api,
            provider=self.provider_id,
            model=model_id,
            usage=Usage(),
            stop_reason="error",
            error_message=str(error) if str(error) else repr(error),
            timestamp=int(time.time() * 1000),
        )

    # -- ProviderStreams -------------------------------------------------------

    def stream(
        self,
        request_model: Model,
        context: Context,
        stream_options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        outer = AssistantMessageEventStream()
        with self._guard:
            step = self._pending_responses.pop(0) if self._pending_responses else None
            self.state.call_count += 1

        async def _run() -> None:
            try:
                if stream_options is not None and stream_options.on_response is not None:
                    await stream_options.on_response(ProviderResponse(status=200, headers={}), request_model)
                if step is None:
                    message = self._create_error_message("No more faux responses queued", request_model.id)
                    message = self._with_usage_estimate(message, context, stream_options)
                    outer.push(ErrorEvent(reason="error", error=message))
                    outer.end(message)
                    return

                if callable(step):
                    resolved = await step(context, stream_options, self.state, request_model)
                else:
                    resolved = step
                message = self._clone_message(resolved, request_model.id)
                message = self._with_usage_estimate(message, context, stream_options)
                await _stream_with_deltas(
                    outer,
                    message,
                    self._min_token_size,
                    self._max_token_size,
                    self._tokens_per_second,
                    stream_options.cancel if stream_options is not None else None,
                )
            except Exception as error:
                message = self._create_error_message(error, request_model.id)
                outer.push(ErrorEvent(reason="error", error=message))
                outer.end(message)

        tonio.spawn.without_tracking(_run())
        return outer

    def stream_simple(
        self,
        request_model: Model,
        context: Context,
        stream_options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return self.stream(request_model, context, stream_options)


@dataclass(slots=True)
class FauxProviderHandle:
    provider: Provider
    api: str
    models: list[Model]
    state: FauxState
    get_model: Callable[..., Model | None]
    set_responses: Callable[[list[FauxResponseStep]], None]
    append_responses: Callable[[list[FauxResponseStep]], None]
    get_pending_response_count: Callable[[], int]


def create_faux_core(**options) -> FauxCore:
    return FauxCore(**options)


def faux_provider(**options) -> FauxProviderHandle:
    """Faux provider for tests built on explicit `Models` collections:

    faux = faux_provider()
    models = create_models()
    models.set_provider(faux.provider)
    faux.set_responses([faux_assistant_message("hi")])
    """
    core = create_faux_core(**options)

    async def resolve(_ctx, _credential) -> AuthResult:
        return AuthResult(auth=ModelAuth())

    provider = create_provider(
        id=core.provider_id,
        auth=ProviderAuth(api_key=ApiKeyAuth(name="Faux", resolve=resolve)),
        models=core.models,
        api=core,
    )
    return FauxProviderHandle(
        provider=provider,
        api=core.api,
        models=core.models,
        state=core.state,
        get_model=core.get_model,
        set_responses=core.set_responses,
        append_responses=core.append_responses,
        get_pending_response_count=core.get_pending_response_count,
    )
