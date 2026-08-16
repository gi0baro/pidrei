"""Shared utilities for the Google Generative AI and Google Vertex providers.

Port of pi's `packages/ai/src/api/google-shared.ts`.

pi imports `Content`, `Part`, `FinishReason` and `FunctionCallingConfigMode` from
the `@google/genai` SDK. There is no equivalent dependency here (see
`api/google_client.py` for why), so contents and parts are the plain dicts that
go straight into the request body, and the two enums are their wire strings —
which is what the SDK's values are.
"""

import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pidrei_ai.api.constrained_sampling import get_json_schema_tool_parameters, resolve_json_schema_strict_sampling
from pidrei_ai.api.transform_messages import transform_messages
from pidrei_ai.types import (
    AssistantMessage,
    Context,
    ImageContent,
    Model,
    StopReason,
    StreamOptions,
    TextContent,
    Tool,
)
from pidrei_ai.utils.provider_retry import retry_provider_request
from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates


type GoogleThinkingLevel = Literal["THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"]

# `FunctionCallingConfigMode` in the SDK; these are its wire values.
FUNCTION_CALLING_MODE_AUTO = "AUTO"
FUNCTION_CALLING_MODE_NONE = "NONE"
FUNCTION_CALLING_MODE_ANY = "ANY"
FUNCTION_CALLING_MODE_VALIDATED = "VALIDATED"


def is_thinking_part(part: dict[str, Any]) -> bool:
    """Whether a streamed Gemini `Part` should be treated as "thinking".

    Protocol note (Gemini / Vertex AI thought signatures):
    - `thought: true` is the definitive marker for thinking content (thought summaries).
    - `thoughtSignature` is an encrypted representation of the model's internal thought process
      used to preserve reasoning context across multi-turn interactions.
    - `thoughtSignature` can appear on ANY part type (text, functionCall, etc.) - it does NOT
      indicate the part itself is thinking content.
    - For non-functionCall responses, the signature appears on the last part for context replay.
    - When persisting/replaying model outputs, signature-bearing parts must be preserved as-is;
      do not merge/move signatures across parts.

    See: https://ai.google.dev/gemini-api/docs/thought-signatures
    """
    return part.get("thought") is True


def retain_thought_signature(existing: str | None, incoming: str | None) -> str | None:
    """Retain thought signatures during streaming.

    Some backends only send `thoughtSignature` on the first delta for a given part/block; later
    deltas may omit it. This helper preserves the last non-empty signature for the current block.

    Note: this does NOT merge or move signatures across distinct response parts. It only prevents
    a signature from being overwritten with `undefined` within the same streamed block.
    """
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


# Thought signatures must be base64 for Google APIs (TYPE_BYTES).
_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return _BASE64_SIGNATURE_PATTERN.match(signature) is not None


def _resolve_thought_signature(is_same_provider_and_model: bool, signature: str | None) -> str | None:
    """Only keep signatures from the same provider/model and with valid base64."""
    return signature if is_same_provider_and_model and _is_valid_thought_signature(signature) else None


def requires_tool_call_id(model_id: str) -> bool:
    """Models via Google APIs that require explicit tool call IDs in function calls/responses."""
    gemini_major_version = _get_gemini_major_version(model_id)
    return model_id.startswith(("claude-", "gpt-oss-")) or (
        gemini_major_version is not None and gemini_major_version >= 3
    )


_GEMINI_MAJOR_VERSION = re.compile(r"^gemini(?:-live)?-(\d+)")


def _get_gemini_major_version(model_id: str) -> int | None:
    match = _GEMINI_MAJOR_VERSION.match(model_id.lower())
    if not match:
        return None
    return int(match.group(1), 10)


def _supports_multimodal_function_response(model_id: str) -> bool:
    gemini_major_version = _get_gemini_major_version(model_id)
    if gemini_major_version is not None:
        return gemini_major_version >= 3
    return True


_TOOL_CALL_ID_DISALLOWED = re.compile(r"[^a-zA-Z0-9_-]")


def convert_messages(model: Model, context: Context) -> list[dict[str, Any]]:
    """Convert internal messages to Gemini `Content[]` format."""
    contents: list[dict[str, Any]] = []

    def normalize_tool_call_id(id: str, _target_model: Model, _source: AssistantMessage) -> str:
        if not requires_tool_call_id(model.id):
            return id
        return _TOOL_CALL_ID_DISALLOWED.sub("_", id)[:64]

    transformed_messages = transform_messages(context.messages, model, normalize_tool_call_id)

    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                contents.append({"role": "user", "parts": [{"text": sanitize_surrogates(msg.content)}]})
            else:
                parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        parts.append({"text": sanitize_surrogates(item.text)})
                    else:
                        parts.append({"inlineData": {"mimeType": item.mime_type, "data": item.data}})
                if len(parts) == 0:
                    continue
                contents.append({"role": "user", "parts": parts})
        elif msg.role == "assistant":
            parts = []
            # Check if message is from same provider and model - only then keep thinking blocks
            is_same_provider_and_model = msg.provider == model.provider and msg.model == model.id

            for block in msg.content:
                if block.type == "text":
                    thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.text_signature)
                    # Skip empty text blocks — unless they carry a thought signature. Gemini can attach
                    # the signature to a part whose visible text is empty and requires it echoed back;
                    # dropping it breaks the reasoning chain and the model intermittently ends mid-task
                    # turns with a thought-only STOP (empty completion, no tool call).
                    if (not block.text or block.text.strip() == "") and not thought_signature:
                        continue
                    parts.append(
                        {
                            "text": sanitize_surrogates(block.text),
                            **({"thoughtSignature": thought_signature} if thought_signature else {}),
                        }
                    )
                elif block.type == "thinking":
                    # Only keep as thinking block if same provider AND same model
                    # Otherwise convert to plain text (no tags to avoid model mimicking them)
                    if is_same_provider_and_model:
                        thought_signature = _resolve_thought_signature(
                            is_same_provider_and_model, block.thinking_signature
                        )
                        # Same rule as text blocks: an empty thinking block is dropped only when it
                        # carries no signature (mirrors the anthropic converter's handling).
                        if (not block.thinking or block.thinking.strip() == "") and not thought_signature:
                            continue
                        parts.append(
                            {
                                "thought": True,
                                "text": sanitize_surrogates(block.thinking),
                                **({"thoughtSignature": thought_signature} if thought_signature else {}),
                            }
                        )
                    else:
                        # Cross-provider/model: the signature is unusable, empty blocks stay dropped.
                        if not block.thinking or block.thinking.strip() == "":
                            continue
                        parts.append({"text": sanitize_surrogates(block.thinking)})
                elif block.type == "toolCall":
                    thought_signature = _resolve_thought_signature(is_same_provider_and_model, block.thought_signature)
                    parts.append(
                        {
                            "functionCall": {
                                "name": block.name,
                                "args": block.arguments if block.arguments is not None else {},
                                **({"id": block.id} if requires_tool_call_id(model.id) else {}),
                            },
                            **({"thoughtSignature": thought_signature} if thought_signature else {}),
                        }
                    )

            if len(parts) == 0:
                continue
            contents.append({"role": "model", "parts": parts})
        elif msg.role == "toolResult":
            # Extract text and image content
            text_content = [c for c in msg.content if isinstance(c, TextContent)]
            text_result = "\n".join(c.text for c in text_content)
            image_content = [c for c in msg.content if isinstance(c, ImageContent)] if "image" in model.input else []

            has_text = len(text_result) > 0
            has_images = len(image_content) > 0

            # Gemini 3+ models support multimodal function responses with images nested inside
            # functionResponse.parts. Claude and other non-Gemini models behind Cloud Code Assist /
            # Gemini < 3 still needs a separate user image turn.
            model_supports_multimodal_function_response = _supports_multimodal_function_response(model.id)

            # Use "output" key for success, "error" key for errors as per SDK documentation
            response_value = (
                sanitize_surrogates(text_result) if has_text else ("(see attached image)" if has_images else "")
            )

            image_parts: list[dict[str, Any]] = [
                {"inlineData": {"mimeType": image_block.mime_type, "data": image_block.data}}
                for image_block in image_content
            ]

            include_id = requires_tool_call_id(model.id)
            function_response_part: dict[str, Any] = {
                "functionResponse": {
                    "name": msg.tool_name,
                    "response": {"error": response_value} if msg.is_error else {"output": response_value},
                    **({"parts": image_parts} if has_images and model_supports_multimodal_function_response else {}),
                    **({"id": msg.tool_call_id} if include_id else {}),
                }
            }

            # Cloud Code Assist API requires all function responses to be in a single user turn.
            # Check if the last content is already a user turn with function responses and merge.
            last_content = contents[-1] if contents else None
            if (
                last_content is not None
                and last_content.get("role") == "user"
                and any(p.get("functionResponse") for p in last_content.get("parts") or [])
            ):
                last_content["parts"].append(function_response_part)
            else:
                contents.append({"role": "user", "parts": [function_response_part]})

            # For Gemini < 3, add images in a separate user message
            if has_images and not model_supports_multimodal_function_response:
                contents.append({"role": "user", "parts": [{"text": "Tool result image:"}, *image_parts]})

    return contents


_JSON_SCHEMA_META_DECLARATIONS = frozenset(
    {
        "$schema",
        "$id",
        "$anchor",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        "$defs",
        "definitions",  # pre-draft-2019-09 equivalent of $defs
    }
)


def _sanitize_for_open_api(schema: Any) -> Any:
    """Strip meta-declarations from a schema obj."""
    if not isinstance(schema, dict):
        return schema

    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _JSON_SCHEMA_META_DECLARATIONS:
            continue
        result[key] = _sanitize_for_open_api(value)
    return result


def convert_tools(
    tools: list[Tool], use_parameters: bool = False, supports_strict_mode: bool = True
) -> list[dict[str, Any]] | None:
    """Convert tools to Gemini function declarations format.

    By default uses `parametersJsonSchema` which supports full JSON Schema (including
    anyOf, oneOf, const, etc.). Set `use_parameters` to True to use the legacy `parameters`
    field instead (OpenAPI 3.03 Schema). This is needed for Cloud Code Assist with Claude
    models, where the API translates `parameters` into Anthropic's `input_schema`.
    """

    def declaration(tool: Tool) -> dict[str, Any]:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        parameters = get_json_schema_tool_parameters(tool, strict)
        return {
            "name": tool.name,
            "description": tool.description,
            **(
                {"parameters": _sanitize_for_open_api(parameters)}
                if use_parameters
                else {"parametersJsonSchema": parameters}
            ),
        }

    if len(tools) == 0:
        return None
    return [{"functionDeclarations": [declaration(tool) for tool in tools]}]


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    """Gemini 3+ enforces required function parameters in validated tool-calling modes."""
    major_version = _get_gemini_major_version(model_id)
    return major_version is not None and major_version >= 3


def map_tool_choice(choice: str) -> str:
    """Map tool choice string to Gemini `FunctionCallingConfigMode`."""
    match choice:
        case "auto":
            return FUNCTION_CALLING_MODE_AUTO
        case "none":
            return FUNCTION_CALLING_MODE_NONE
        case "any":
            return FUNCTION_CALLING_MODE_ANY
        case _:
            return FUNCTION_CALLING_MODE_AUTO


def resolve_google_function_calling_mode(
    tools: list[Tool],
    tool_choice: str | None,
    supports_strict_mode: bool,
) -> str | None:
    use_strict_mode = any(resolve_json_schema_strict_sampling(tool, supports_strict_mode) is True for tool in tools)
    if tool_choice == "none" or tool_choice == "any":
        return map_tool_choice(tool_choice)
    if use_strict_mode:
        return FUNCTION_CALLING_MODE_VALIDATED
    return map_tool_choice(tool_choice) if tool_choice else None


# `FinishReason` in the SDK. pi's `mapStopReason` switch is exhaustive over the enum and its
# `never` default throws, so an unknown value is a hard error rather than a silent "error" stop —
# the same is true here, and the adapters' error handling turns it into an error event.
_STOP_REASON_ERROR = frozenset(
    {
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "SAFETY",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "RECITATION",
        "FINISH_REASON_UNSPECIFIED",
        "OTHER",
        "LANGUAGE",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "NO_IMAGE",
    }
)


def map_stop_reason(reason: str) -> StopReason:
    """Map Gemini `FinishReason` to our `StopReason`."""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    if reason in _STOP_REASON_ERROR:
        return "error"
    raise ValueError(f"Unhandled stop reason: {reason}")


def map_stop_reason_string(reason: str) -> StopReason:
    """Map string finish reason to our `StopReason` (for raw API responses)."""
    match reason:
        case "STOP":
            return "stop"
        case "MAX_TOKENS":
            return "length"
        case _:
            return "error"


async def retry_google_request[T](request: Callable[[], Awaitable[T]], options: StreamOptions | None = None) -> T:
    """Run a Google GenAI request with the shared provider retry policy
    (408/409/429/5xx with backoff, honoring retry-after), mirroring how the
    Anthropic and OpenAI adapters wrap their initial request in
    `retry_provider_request`. `GoogleApiError` has a `status` attribute but no
    `headers` attribute, and `retry_provider_request` only retries errors that
    carry both, so normalize the error by adding the missing `headers` before
    re-raising.
    """

    async def _request() -> T:
        try:
            return await request()
        except Exception as error:
            if hasattr(error, "status") and not hasattr(error, "headers"):
                try:
                    error.headers = None
                except AttributeError:
                    pass  # slots-only exception: stays non-retryable, as pi's frozen errors would
            raise

    max_retries = options.max_retries if options is not None and options.max_retries is not None else 0
    return await retry_provider_request(
        _request,
        max_retries=max_retries,
        max_retry_delay_ms=options.max_retry_delay_ms if options is not None else None,
        cancel=options.cancel if options is not None else None,
    )
