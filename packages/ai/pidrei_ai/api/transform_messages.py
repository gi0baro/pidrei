"""Port of pi's cross-provider message transforms (packages/ai/src/api/transform-messages.ts).

Handles unsupported-image downgrade, cross-model thinking replay (thinking →
plain text, redacted thinking dropped), thought-signature stripping, tool-call
id normalization, skipping errored/aborted assistant turns, and synthesizing
"No result provided" tool results for orphaned tool calls.
"""

import time
from collections.abc import Callable
from dataclasses import replace

from pidrei_ai.types import (
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ToolCall,
    ToolResultMessage,
)


NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = "(tool image omitted: model does not support images)"


def _replace_images_with_placeholder(content: list, placeholder: str) -> list[TextContent]:
    result: list[TextContent] = []
    previous_was_placeholder = False

    for block in content:
        if block.type == "image":
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue

        result.append(block)
        previous_was_placeholder = block.text == placeholder

    return result


def _downgrade_unsupported_images(messages: list[Message], model: Model) -> list[Message]:
    if "image" in model.input:
        return messages

    downgraded: list[Message] = []
    for message in messages:
        if message.role == "user" and isinstance(message.content, list):
            downgraded.append(
                replace(
                    message,
                    content=_replace_images_with_placeholder(message.content, NON_VISION_USER_IMAGE_PLACEHOLDER),
                )
            )
        elif message.role == "toolResult":
            downgraded.append(
                replace(
                    message,
                    content=_replace_images_with_placeholder(message.content, NON_VISION_TOOL_IMAGE_PLACEHOLDER),
                )
            )
        else:
            downgraded.append(message)
    return downgraded


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[Message]:
    """Normalize a message history for replay against `model`.

    Tool call IDs are normalized for cross-provider compatibility (e.g. OpenAI
    Responses generates 450+ char IDs with `|`; Anthropic requires
    `^[a-zA-Z0-9_-]+$` up to 64 chars).
    """
    tool_call_id_map: dict[str, str] = {}
    # Normalize null content from untyped callers (custom tools, hand-built
    # histories, old session files) so downstream code can rely on the contract.
    normalized_messages = [replace(message, content=[]) if message.content is None else message for message in messages]
    image_aware_messages = _downgrade_unsupported_images(normalized_messages, model)

    # First pass: thinking/text/toolCall transformation per assistant message.
    transformed: list[Message] = []
    for message in image_aware_messages:
        if message.role == "user":
            transformed.append(message)
            continue

        if message.role == "toolResult":
            normalized_id = tool_call_id_map.get(message.tool_call_id)
            if normalized_id is not None and normalized_id != message.tool_call_id:
                transformed.append(replace(message, tool_call_id=normalized_id))
            else:
                transformed.append(message)
            continue

        if message.role == "assistant":
            is_same_model = (
                message.provider == model.provider and message.api == model.api and message.model == model.id
            )

            transformed_content: list = []
            for block in message.content:
                if block.type == "thinking":
                    # Redacted thinking is opaque encrypted content, only valid
                    # for the same model; drop it cross-model to avoid API errors.
                    if block.redacted:
                        if is_same_model:
                            transformed_content.append(block)
                        continue
                    # Same model: keep thinking blocks with signatures (needed
                    # for replay) even if the thinking text is empty.
                    if is_same_model and block.thinking_signature:
                        transformed_content.append(block)
                        continue
                    # Skip empty thinking blocks, convert others to plain text.
                    if not block.thinking or block.thinking.strip() == "":
                        continue
                    transformed_content.append(block if is_same_model else TextContent(text=block.thinking))
                    continue

                if block.type == "text":
                    transformed_content.append(block if is_same_model else TextContent(text=block.text))
                    continue

                if block.type == "toolCall":
                    normalized_tool_call = block
                    if not is_same_model and block.thought_signature is not None:
                        normalized_tool_call = replace(normalized_tool_call, thought_signature=None)
                    if not is_same_model and normalize_tool_call_id is not None:
                        normalized_id = normalize_tool_call_id(block.id, model, message)
                        if normalized_id != block.id:
                            tool_call_id_map[block.id] = normalized_id
                            normalized_tool_call = replace(normalized_tool_call, id=normalized_id)
                    transformed_content.append(normalized_tool_call)
                    continue

                transformed_content.append(block)

            transformed.append(replace(message, content=transformed_content))
            continue

        transformed.append(message)

    # Second pass: insert synthetic empty tool results for orphaned tool calls.
    # This preserves thinking signatures and satisfies API requirements.
    result: list[Message] = []
    pending_tool_calls: list[ToolCall] = []
    existing_tool_result_ids: set[str] = set()

    def insert_synthetic_tool_results() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        if pending_tool_calls:
            for tool_call in pending_tool_calls:
                if tool_call.id not in existing_tool_result_ids:
                    result.append(
                        ToolResultMessage(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            content=[TextContent(text="No result provided")],
                            is_error=True,
                            timestamp=int(time.time() * 1000),
                        )
                    )
            pending_tool_calls = []
            existing_tool_result_ids = set()

    for message in transformed:
        if message.role == "assistant":
            # Pending orphaned tool calls from a previous assistant: insert now.
            insert_synthetic_tool_results()

            # Skip errored/aborted assistant messages entirely: incomplete
            # turns that shouldn't be replayed (partial content can cause API
            # errors; the model should retry from the last valid state).
            if message.stop_reason in ("error", "aborted"):
                continue

            tool_calls = [block for block in message.content if block.type == "toolCall"]
            if tool_calls:
                pending_tool_calls = tool_calls
                existing_tool_result_ids = set()

            result.append(message)
        elif message.role == "toolResult":
            existing_tool_result_ids.add(message.tool_call_id)
            result.append(message)
        elif message.role == "user":
            # User message interrupts tool flow: synthesize orphaned results.
            insert_synthetic_tool_results()
            result.append(message)
        else:
            result.append(message)

    # If the conversation ends with unresolved tool calls, synthesize now.
    insert_synthetic_tool_results()

    return result
