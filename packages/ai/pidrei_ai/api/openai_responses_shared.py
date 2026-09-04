"""Port of pi's shared Responses-API logic (packages/ai/src/api/openai-responses-shared.ts).

Message/tool conversion and the event-stream processor shared by the
openai-responses family (openai, azure, codex). Stream events arrive as plain
dicts (pidrei parses the SSE itself); pi's per-block scratch (`partialJson`,
`customInput`) lives on the slot records here instead of the content blocks.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

from pidrei_ai.api.constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    get_grammar_tool_input,
    get_json_schema_tool_parameters,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from pidrei_ai.api.transform_messages import transform_messages
from pidrei_ai.builders import (
    AssistantMessageBuilder,
    TextContentBuilder,
    ThinkingContentBuilder,
    ToolCallBuilder,
    UsageBuilder,
)
from pidrei_ai.registry import calculate_cost
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    Model,
    StopReason,
    TextDeltaEvent,
    TextEndEvent,
    TextStartEvent,
    ThinkingDeltaEvent,
    ThinkingEndEvent,
    ThinkingStartEvent,
    Tool,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
)
from pidrei_ai.utils.event_stream import AssistantMessageEventStream
from pidrei_ai.utils.hash import short_hash
from pidrei_ai.utils.json_parse import parse_streaming_json
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates


# Python has one absent value where JavaScript has two; this stands in for
# `undefined` where a caller may also pass an explicit `null` (see
# `convert_responses_tools`).
UNSET: Any = object()


# =============================================================================
# Utilities
# =============================================================================


def encode_text_signature_v1(id: str, phase: str | None = None) -> str:
    payload: dict[str, Any] = {"v": 1, "id": id}
    if phase:
        payload["phase"] = phase
    return json.dumps(payload, separators=(",", ":"))


def parse_text_signature(signature: str | None) -> dict[str, Any] | None:
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            parsed = json.loads(signature)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict) and parsed.get("v") == 1 and isinstance(parsed.get("id"), str):
            if parsed.get("phase") in ("commentary", "final_answer"):
                return {"id": parsed["id"], "phase": parsed["phase"]}
            return {"id": parsed["id"]}
    return {"id": signature}


def convert_tool_result_output(model: Model, content: list) -> str | list[dict]:
    text_result = "\n".join(block.text for block in content if block.type == "text")
    images = [block for block in content if block.type == "image"]
    has_text = len(text_result) > 0

    if not images or "image" not in model.input:
        return sanitize_surrogates(
            text_result if has_text else "(see attached image)" if images else "(no tool output)"
        )

    output: list[dict] = []
    if has_text:
        output.append({"type": "input_text", "text": sanitize_surrogates(text_result)})
    for image in images:
        output.append(
            {"type": "input_image", "detail": "auto", "image_url": f"data:{image.mime_type};base64,{image.data}"}
        )
    return output


# =============================================================================
# Message conversion
# =============================================================================


def convert_responses_messages(
    model: Model,
    context: Context,
    allowed_tool_call_providers: set[str],
    *,
    include_system_prompt: bool = True,
    grammar_tool_input_properties: dict[str, str] | None = None,
    deferred_tools: dict[str, Tool] | None = None,
    deferred_tools_mode: str | None = None,
    tool_options: dict | None = None,
) -> list[dict]:
    grammar_tool_input_properties = grammar_tool_input_properties or {}
    deferred_tools = deferred_tools or {}
    messages: list[dict] = []
    loaded_tool_names: set[str] = set()

    def normalize_id_part(part: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", part)
        normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
        return re.sub(r"_+$", "", normalized)

    def build_foreign_responses_item_id(item_id: str) -> str:
        normalized = f"fc_{short_hash(item_id)}"
        return normalized[:64] if len(normalized) > 64 else normalized

    def normalize_tool_call_id(id: str, _target_model: Model, source: AssistantMessage) -> str:
        if model.provider not in allowed_tool_call_providers:
            return normalize_id_part(id)
        if "|" not in id:
            return normalize_id_part(id)
        call_id, _, item_id = id.partition("|")
        normalized_call_id = normalize_id_part(call_id)
        is_foreign_tool_call = source.provider != model.provider or source.api != model.api
        normalized_item_id = (
            build_foreign_responses_item_id(item_id) if is_foreign_tool_call else normalize_id_part(item_id)
        )
        # OpenAI Responses API requires item ids to start with "fc".
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    transformed_messages = transform_messages(context.messages, model, normalize_tool_call_id)

    if include_system_prompt and context.system_prompt:
        compat = model.compat
        supports_developer_role = getattr(compat, "supports_developer_role", None)
        role = "developer" if model.reasoning and supports_developer_role is not False else "system"
        messages.append({"role": role, "content": sanitize_surrogates(context.system_prompt)})

    msg_index = 0
    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                messages.append(
                    {"role": "user", "content": [{"type": "input_text", "text": sanitize_surrogates(msg.content)}]}
                )
            else:
                content = []
                for item in msg.content:
                    if item.type == "text":
                        content.append({"type": "input_text", "text": sanitize_surrogates(item.text)})
                    else:
                        content.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.mime_type};base64,{item.data}",
                            }
                        )
                if not content:
                    msg_index += 1
                    continue
                messages.append({"role": "user", "content": content})
        elif msg.role == "assistant":
            output: list[dict] = []
            is_same_provider_and_api = msg.provider == model.provider and msg.api == model.api
            is_same_model = is_same_provider_and_api and msg.model == model.id
            is_different_model = is_same_provider_and_api and msg.model != model.id
            text_block_index = 0

            for block in msg.content:
                if block.type == "thinking":
                    if block.thinking_signature:
                        output.append(json.loads(block.thinking_signature))
                elif block.type == "text":
                    parsed_signature = parse_text_signature(block.text_signature)
                    fallback_message_id = (
                        f"msg_pi_{msg_index}" if text_block_index == 0 else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1
                    # OpenAI requires ids to be at most 64 characters.
                    msg_id = parsed_signature["id"] if parsed_signature else None
                    if not msg_id:
                        msg_id = fallback_message_id
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"
                    entry: dict[str, Any] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": sanitize_surrogates(block.text), "annotations": []}
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    phase = parsed_signature.get("phase") if parsed_signature else None
                    if phase is not None:
                        entry["phase"] = phase
                    output.append(entry)
                elif block.type == "toolCall":
                    call_id, _, item_id_raw = block.id.partition("|")
                    custom_input_property = grammar_tool_input_properties.get(block.name)
                    item_id: str | None = item_id_raw if "|" in block.id else ""

                    # Different-model messages omit the item id to dodge OpenAI's
                    # fc/rs pairing validation; non-fc_* ids (e.g. ctc_*) are
                    # dropped when replaying custom-tool calls as function_call.
                    if (is_different_model and item_id and item_id.startswith("fc_")) or (
                        custom_input_property is None and not (item_id or "").startswith("fc_")
                    ):
                        item_id = None

                    can_replay_namespace = is_same_model or block.name in deferred_tools

                    if custom_input_property is not None:
                        entry = {
                            "type": "custom_tool_call",
                            "call_id": call_id,
                            "name": block.name,
                            "input": sanitize_surrogates(
                                get_grammar_tool_input(block.name, block.arguments, custom_input_property)
                            ),
                        }
                        if item_id is not None:
                            entry["id"] = item_id
                        if can_replay_namespace and block.namespace is not None:
                            entry["namespace"] = block.namespace
                        output.append(entry)
                    else:
                        entry = {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": block.name,
                            "arguments": json.dumps(block.arguments),
                        }
                        if item_id is not None:
                            entry["id"] = item_id
                        if can_replay_namespace and block.namespace is not None:
                            entry["namespace"] = block.namespace
                        output.append(entry)
            if not output:
                msg_index += 1
                continue
            messages.extend(output)
        elif msg.role == "toolResult":
            call_id = msg.tool_call_id.partition("|")[0]
            result_output = convert_tool_result_output(model, msg.content)

            if msg.tool_name in grammar_tool_input_properties:
                messages.append({"type": "custom_tool_call_output", "call_id": call_id, "output": result_output})
            else:
                messages.append({"type": "function_call_output", "call_id": call_id, "output": result_output})

            newly_loaded: list[Tool] = []
            for name in msg.added_tool_names or []:
                tool = deferred_tools.get(name)
                if tool is None or name in loaded_tool_names:
                    continue
                loaded_tool_names.add(name)
                newly_loaded.append(tool)
            if newly_loaded and deferred_tools_mode == "additional-tools":
                messages.append(
                    {
                        "type": "additional_tools",
                        "role": "developer",
                        "tools": convert_responses_tools(newly_loaded, **(tool_options or {})),
                    }
                )
            elif newly_loaded and deferred_tools_mode == "tool-search":
                names = [tool.name for tool in newly_loaded]
                search_call_id = f"pi_tool_load_{short_hash(msg.tool_call_id + ':' + ','.join(names))}"
                messages.append(
                    {
                        "type": "tool_search_call",
                        "call_id": search_call_id,
                        "execution": "client",
                        "status": "completed",
                        "arguments": {"query": " ".join(names), "limit": len(names)},
                    }
                )
                messages.append(
                    {
                        "type": "tool_search_output",
                        "call_id": search_call_id,
                        "execution": "client",
                        "status": "completed",
                        "tools": convert_responses_tools(
                            newly_loaded, **{**(tool_options or {}), "defer_loading": True}
                        ),
                    }
                )
        msg_index += 1

    return messages


# =============================================================================
# Tool conversion
# =============================================================================


def convert_responses_tools(
    tools: list[Tool],
    *,
    strict: bool | None = UNSET,
    supports_strict_mode: bool = True,
    supports_openai_grammar_tools: bool = False,
    defer_loading: bool = False,
) -> list[dict]:
    # pi: `options?.strict === undefined ? false : options.strict` — an explicit
    # `null` (the Codex adapter sends one) must survive as `strict: null`, so
    # "not passed" needs its own value distinct from None.
    default_strict = False if strict is UNSET else strict

    converted: list[dict] = []
    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(tool, supports_openai_grammar_tools)
        if grammar:
            entry: dict[str, Any] = {
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "format": {"type": "grammar", "syntax": grammar.format, "definition": grammar.definition},
            }
            if defer_loading:
                entry["defer_loading"] = True
            converted.append(entry)
            continue

        constrained_strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        strict = constrained_strict if constrained_strict is not None else default_strict
        function_tool: dict[str, Any] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": get_json_schema_tool_parameters(tool, strict is True),
        }
        if defer_loading:
            function_tool["defer_loading"] = True
        if supports_strict_mode:
            function_tool["strict"] = strict
        converted.append(function_tool)
    return converted


# =============================================================================
# Stream processing
# =============================================================================


@dataclass(slots=True)
class _Slot:
    kind: str  # "thinking" | "text" | "toolCall"
    block: Any
    content_index: int
    partial_json: str | None = None
    custom_property: str | None = None
    json_buffer: GrammarToolInputJsonBuffer | None = None


def _map_stop_reason(status: str | None, incomplete_reason: str | None = None) -> tuple[StopReason, str | None]:
    """Returns ``(stop_reason, error_message)`` — pi's `{stopReason, errorMessage}`."""
    if not status:
        return "stop", None
    if status == "completed":
        return "stop", None
    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return "length", None
        return "error", (
            f"Response incomplete: {incomplete_reason}"
            if incomplete_reason
            else "Response incomplete without a provider reason"
        )
    if status in ("failed", "cancelled"):
        return "error", None
    if status in ("in_progress", "queued"):  # These two are wonky...
        return "stop", None
    raise RuntimeError(f"Unhandled stop reason: {status}")


async def process_responses_stream(  # noqa: C901 (mirrors pi's event ladder)
    events,
    output: AssistantMessageBuilder,
    stream: AssistantMessageEventStream,
    model: Model,
    *,
    service_tier: str | None = None,
    grammar_tool_input_properties: dict[str, str] | None = None,
    resolve_service_tier=None,
    apply_service_tier_pricing=None,
) -> None:
    grammar_tool_input_properties = grammar_tool_input_properties or {}
    saw_terminal_response_event = False
    output_slots: dict[int, _Slot] = {}
    reasoning_blocks_by_id: dict[str, ThinkingContentBuilder] = {}

    def get_slot(output_index: int, kind: str) -> _Slot | None:
        slot = output_slots.get(output_index)
        return slot if slot is not None and slot.kind == kind else None

    def push_tool_call_delta(slot: _Slot, delta: str | None) -> None:
        if delta is None:
            return
        stream.push(ToolCallDeltaEvent(content_index=slot.content_index, delta=delta, partial=output))

    def get_custom_input(slot: _Slot) -> str:
        if slot.custom_property is None:
            return ""
        value = slot.block.arguments.get(slot.custom_property)
        return value if isinstance(value, str) else ""

    def append_custom_input(slot: _Slot, next_input: str, close: bool) -> str | None:
        if slot.custom_property is None or slot.json_buffer is None:
            return None
        delta = append_grammar_tool_input_json_delta(slot.json_buffer, slot.custom_property, next_input, close)
        slot.block.arguments = {slot.custom_property: next_input}
        return delta

    def apply_message_phase_stop_reason(item: dict) -> None:
        if item.get("type") == "message" and item.get("phase") == "final_answer":
            output.stop_reason = "stop"

    def create_slot(output_index: int, item: dict) -> _Slot | None:
        item_type = item.get("type")
        if item_type == "reasoning":
            block = ThinkingContentBuilder(thinking="")
            output.content.append(block)
            slot = _Slot(kind="thinking", block=block, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(ThinkingStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "message":
            apply_message_phase_stop_reason(item)
            block = TextContentBuilder(text="")
            output.content.append(block)
            slot = _Slot(kind="text", block=block, content_index=len(output.content) - 1)
            output_slots[output_index] = slot
            stream.push(TextStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "function_call":
            block = ToolCallBuilder(
                id=f"{item.get('call_id')}|{item.get('id')}",
                name=item.get("name", ""),
                arguments={},
                namespace=item.get("namespace"),
            )
            output.content.append(block)
            slot = _Slot(
                kind="toolCall",
                block=block,
                content_index=len(output.content) - 1,
                partial_json=item.get("arguments") or "",
            )
            output_slots[output_index] = slot
            stream.push(ToolCallStartEvent(content_index=slot.content_index, partial=output))
            return slot
        if item_type == "custom_tool_call":
            input_property = grammar_tool_input_properties.get(item.get("name", ""), "input")
            block = ToolCallBuilder(
                id=f"{item.get('call_id')}|{item.get('id')}",
                name=item.get("name", ""),
                arguments={},
                namespace=item.get("namespace"),
            )
            output.content.append(block)
            slot = _Slot(
                kind="toolCall",
                block=block,
                content_index=len(output.content) - 1,
                custom_property=input_property,
                json_buffer=GrammarToolInputJsonBuffer(),
            )
            output_slots[output_index] = slot
            stream.push(ToolCallStartEvent(content_index=slot.content_index, partial=output))
            if item.get("input"):
                push_tool_call_delta(slot, append_custom_input(slot, item["input"], False))
            return slot
        return None

    def get_or_create_slot(output_index: int, item: dict) -> _Slot | None:
        return output_slots.get(output_index) or create_slot(output_index, item)

    def backfill_reasoning_signatures(response_output: list) -> None:
        # Azure can omit reasoning.encrypted_content from output_item.done and
        # provide it only in response.completed; backfill for stateless replay.
        for item in response_output:
            if not isinstance(item, dict) or item.get("type") != "reasoning" or not item.get("encrypted_content"):
                continue
            block = reasoning_blocks_by_id.get(item.get("id"))
            if block is None or not block.thinking_signature:
                continue
            stored_item = json.loads(block.thinking_signature)
            if stored_item.get("encrypted_content"):
                continue
            block.thinking_signature = json.dumps({**stored_item, "encrypted_content": item["encrypted_content"]})

    def finalize_response(response: dict) -> None:
        nonlocal saw_terminal_response_event
        saw_terminal_response_event = True
        backfill_reasoning_signatures(response.get("output") or [])
        if response.get("id"):
            output.response_id = response["id"]
        usage = response.get("usage")
        if usage:
            input_details = usage.get("input_tokens_details") or {}
            cached_tokens = input_details.get("cached_tokens") or 0
            cache_write_tokens = input_details.get("cache_write_tokens") or 0
            output_details = usage.get("output_tokens_details") or {}
            output.usage = UsageBuilder(
                # OpenAI includes cached and cache-write tokens in input_tokens.
                input=max(0, (usage.get("input_tokens") or 0) - cached_tokens - cache_write_tokens),
                output=usage.get("output_tokens") or 0,
                cache_read=cached_tokens,
                cache_write=cache_write_tokens,
                reasoning=output_details.get("reasoning_tokens") or 0,
                total_tokens=usage.get("total_tokens") or 0,
            )
        calculate_cost(model, output.usage)
        if apply_service_tier_pricing is not None:
            if resolve_service_tier is not None:
                resolved_tier = resolve_service_tier(response.get("service_tier"), service_tier)
            else:
                resolved_tier = (
                    response.get("service_tier") if response.get("service_tier") is not None else service_tier
                )
            apply_service_tier_pricing(output.usage, resolved_tier)
        # Map status to stop reason. For incomplete responses, retain the
        # provider's specific reason so max-output truncation and content
        # filtering stay distinct.
        status = response.get("status")
        incomplete_details = response.get("incomplete_details")
        incomplete_reason = None
        if isinstance(incomplete_details, dict) and isinstance(incomplete_details.get("reason"), str):
            incomplete_reason = incomplete_details["reason"]
        output.raw_stop_reason = f"{status}.{incomplete_reason}" if incomplete_reason else status
        output.stop_reason, output.error_message = _map_stop_reason(status, incomplete_reason)
        if any(block.type == "toolCall" for block in output.content) and output.stop_reason == "stop":
            output.stop_reason = "toolUse"

    async for event in events:
        event_type = event.get("type")
        if event_type == "response.created":
            output.response_id = (event.get("response") or {}).get("id")
        elif event_type == "response.output_item.added":
            create_slot(event.get("output_index"), event.get("item") or {})
        elif event_type in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
            slot = get_slot(event.get("output_index"), "thinking")
            if slot is None:
                continue
            slot.block.thinking += event.get("delta", "")
            stream.push(
                ThinkingDeltaEvent(content_index=slot.content_index, delta=event.get("delta", ""), partial=output)
            )
        elif event_type == "response.reasoning_summary_part.done":
            slot = get_slot(event.get("output_index"), "thinking")
            if slot is None:
                continue
            slot.block.thinking += "\n\n"
            stream.push(ThinkingDeltaEvent(content_index=slot.content_index, delta="\n\n", partial=output))
        elif event_type in ("response.output_text.delta", "response.refusal.delta"):
            slot = get_slot(event.get("output_index"), "text")
            if slot is None:
                continue
            slot.block.text += event.get("delta", "")
            stream.push(TextDeltaEvent(content_index=slot.content_index, delta=event.get("delta", ""), partial=output))
        elif event_type == "response.function_call_arguments.delta":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or slot.partial_json is None:
                continue
            slot.partial_json += event.get("delta", "")
            slot.block.arguments = parse_streaming_json(slot.partial_json)
            push_tool_call_delta(slot, event.get("delta", ""))
        elif event_type == "response.function_call_arguments.done":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or slot.partial_json is None:
                continue
            previous_partial_json = slot.partial_json
            slot.partial_json = event.get("arguments", "")
            slot.block.arguments = parse_streaming_json(slot.partial_json)
            if slot.partial_json.startswith(previous_partial_json):
                delta = slot.partial_json[len(previous_partial_json) :]
                if delta:
                    push_tool_call_delta(slot, delta)
        elif event_type == "response.custom_tool_call_input.delta":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or slot.custom_property is None:
                continue
            push_tool_call_delta(
                slot, append_custom_input(slot, get_custom_input(slot) + event.get("delta", ""), False)
            )
        elif event_type == "response.custom_tool_call_input.done":
            slot = get_slot(event.get("output_index"), "toolCall")
            if slot is None or slot.custom_property is None:
                continue
            push_tool_call_delta(slot, append_custom_input(slot, event.get("input", ""), True))
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            apply_message_phase_stop_reason(item)
            slot = get_or_create_slot(event.get("output_index"), item)
            item_type = item.get("type")

            if item_type == "reasoning" and slot is not None and slot.kind == "thinking":
                summary_text = "\n\n".join(part.get("text", "") for part in item.get("summary") or []) or ""
                content_text = "\n\n".join(part.get("text", "") for part in item.get("content") or []) or ""
                slot.block.thinking = summary_text or content_text or slot.block.thinking
                slot.block.thinking_signature = json.dumps(item)
                reasoning_blocks_by_id[item.get("id")] = slot.block
                stream.push(
                    ThinkingEndEvent(content_index=slot.content_index, content=slot.block.thinking, partial=output)
                )
                output_slots.pop(event.get("output_index"), None)
            elif item_type == "message" and slot is not None and slot.kind == "text":
                slot.block.text = "".join(
                    part.get("text", "") if part.get("type") == "output_text" else part.get("refusal", "")
                    for part in item.get("content") or []
                )
                slot.block.text_signature = encode_text_signature_v1(item.get("id"), item.get("phase"))
                stream.push(TextEndEvent(content_index=slot.content_index, content=slot.block.text, partial=output))
                output_slots.pop(event.get("output_index"), None)
            elif (
                item_type == "function_call"
                and slot is not None
                and slot.kind == "toolCall"
                and slot.partial_json is not None
            ):
                slot.block.arguments = parse_streaming_json(item.get("arguments") or slot.partial_json or "{}")
                if item.get("namespace") is not None:
                    slot.block.namespace = item["namespace"]
                slot.partial_json = None
                stream.push(ToolCallEndEvent(content_index=slot.content_index, tool_call=slot.block, partial=output))
                output_slots.pop(event.get("output_index"), None)
            elif (
                item_type == "custom_tool_call"
                and slot is not None
                and slot.kind == "toolCall"
                and slot.custom_property is not None
            ):
                next_input = item.get("input")
                push_tool_call_delta(
                    slot,
                    append_custom_input(slot, next_input if next_input is not None else get_custom_input(slot), True),
                )
                if item.get("namespace") is not None:
                    slot.block.namespace = item["namespace"]
                slot.custom_property = None
                slot.json_buffer = None
                stream.push(ToolCallEndEvent(content_index=slot.content_index, tool_call=slot.block, partial=output))
                output_slots.pop(event.get("output_index"), None)
        elif event_type in ("response.completed", "response.incomplete"):
            finalize_response(event.get("response") or {})
        elif event_type == "error":
            raise RuntimeError(f"Error Code {event.get('code')}: {event.get('message')}")
        elif event_type == "response.failed":
            saw_terminal_response_event = True
            response = event.get("response") or {}
            output.raw_stop_reason = response.get("status")
            error = response.get("error")
            details = response.get("incomplete_details")
            if error:
                msg = f"{error.get('code') or 'unknown'}: {error.get('message') or 'no message'}"
            elif details and details.get("reason"):
                msg = f"incomplete: {details['reason']}"
            else:
                msg = "Unknown error (no error details in response)"
            raise RuntimeError(msg)

    if not saw_terminal_response_event:
        raise RuntimeError("OpenAI Responses stream ended before a terminal response event")
