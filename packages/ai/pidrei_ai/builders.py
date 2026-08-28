"""Producer-private mutable builders for the frozen message types.

pidrei-owned (pi has no counterpart): pi's adapters mutate the live message
they are streaming, which is safe with one thread and a data race on a
free-threaded runtime. Here the message types in `types.py` are frozen
values; adapters build through these mirrors instead — same field names, so
every ported mutation line (`block.text +=`, `content.append`,
`output.usage.input = ...`, `calculate_cost(model, output.usage)`) stays
verbatim — and `AssistantMessageEventStream.push()` calls `freeze()` at the
publication seam, so each pushed event carries an independent frozen
snapshot (per-delta cadence; measured at ~3 µs per delta, see
PROPER_MT_DESIGN.md step 2a).

Builders are producer-private by contract: nothing outside the producing
adapter (and the stream's abort path, which the producer's owner task runs)
may hold one. `freeze()` rebuilds the content list and every nested value on
each call; already-frozen blocks in a builder's content pass through as-is.
User-owned dicts (`arguments`, `details`, `data`) are shared by reference —
producers rebind them per update and never mutate them in place.
"""

from dataclasses import dataclass, field, fields
from typing import Any, Literal

from pidrei_ai.types import (
    Api,
    AssistantContent,
    AssistantMessage,
    AssistantMessageDiagnostic,
    DeferredHandle,
    ProviderId,
    StopReason,
    TextContent,
    ThinkingContent,
    ToolCall,
    Usage,
    UsageCost,
)


@dataclass(slots=True)
class TextContentBuilder:
    text: str
    text_signature: str | None = None
    type: Literal["text"] = "text"

    def freeze(self) -> TextContent:
        return TextContent(text=self.text, text_signature=self.text_signature)


@dataclass(slots=True)
class ThinkingContentBuilder:
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False
    type: Literal["thinking"] = "thinking"

    def freeze(self) -> ThinkingContent:
        return ThinkingContent(
            thinking=self.thinking,
            thinking_signature=self.thinking_signature,
            redacted=self.redacted,
        )


@dataclass(slots=True)
class ToolCallBuilder:
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None
    namespace: str | None = None
    type: Literal["toolCall"] = "toolCall"

    def freeze(self) -> ToolCall:
        return ToolCall(
            id=self.id,
            name=self.name,
            arguments=self.arguments,
            thought_signature=self.thought_signature,
            namespace=self.namespace,
        )


@dataclass(slots=True)
class UsageCostBuilder:
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0

    def freeze(self) -> UsageCost:
        return UsageCost(
            input=self.input,
            output=self.output,
            cache_read=self.cache_read,
            cache_write=self.cache_write,
            total=self.total,
        )


@dataclass(slots=True)
class UsageBuilder:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    total_tokens: int = 0
    cost: UsageCostBuilder = field(default_factory=UsageCostBuilder)

    def freeze(self) -> Usage:
        cost = self.cost
        return Usage(
            input=self.input,
            output=self.output,
            cache_read=self.cache_read,
            cache_write=self.cache_write,
            cache_write_1h=self.cache_write_1h,
            reasoning=self.reasoning,
            total_tokens=self.total_tokens,
            cost=cost.freeze() if isinstance(cost, UsageCostBuilder) else cost,
        )


type AssistantContentBuilder = TextContentBuilder | ThinkingContentBuilder | ToolCallBuilder


def _freeze_block(block: AssistantContentBuilder | AssistantContent) -> AssistantContent:
    freeze = getattr(block, "freeze", None)
    return freeze() if freeze is not None else block


@dataclass(slots=True)
class AssistantMessageBuilder:
    content: list[AssistantContentBuilder | AssistantContent]
    api: Api
    provider: ProviderId
    model: str
    usage: UsageBuilder
    stop_reason: StopReason
    timestamp: int
    response_model: str | None = None
    response_id: str | None = None
    diagnostics: list[AssistantMessageDiagnostic] | None = None
    error_message: str | None = None
    raw_stop_reason: str | None = None
    end_turn: bool | None = None
    deferred: DeferredHandle | None = None
    role: Literal["assistant"] = "assistant"

    @classmethod
    def from_message(cls, message: AssistantMessage, **overrides: Any) -> AssistantMessageBuilder:
        """Builder seeded from a frozen message (content blocks stay frozen refs
        until the caller rebinds them; `freeze()` passes them through)."""
        values = {f.name: getattr(message, f.name) for f in fields(AssistantMessage) if f.name != "role"}
        values.update(overrides)
        return cls(**values)

    def freeze(self) -> AssistantMessage:
        usage = self.usage
        return AssistantMessage(
            content=[_freeze_block(block) for block in self.content],
            api=self.api,
            provider=self.provider,
            model=self.model,
            usage=usage.freeze() if isinstance(usage, UsageBuilder) else usage,
            stop_reason=self.stop_reason,
            timestamp=self.timestamp,
            response_model=self.response_model,
            response_id=self.response_id,
            diagnostics=list(self.diagnostics) if self.diagnostics is not None else None,
            error_message=self.error_message,
            raw_stop_reason=self.raw_stop_reason,
            end_turn=self.end_turn,
            deferred=self.deferred,
        )
