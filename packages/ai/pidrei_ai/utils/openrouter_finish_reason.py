"""Port of pi's OpenRouter finish reason helpers (packages/ai/src/utils/openrouter-finish-reason.ts).

OpenRouter's `native_finish_reason`: the upstream provider's own finish reason, which OpenRouter
sends next to the OpenAI-style `finish_reason` it normalizes it into. Recorded for diagnostics, and
used to catch upstream failures that OpenRouter reports as an empty `stop`.

The helpers take the streamed partial, which here is the producer-private `AssistantMessageBuilder`
(recipe `freeze-at-seam`).
"""

from typing import Any

from pidrei_ai.builders import AssistantMessageBuilder


def read_openrouter_native_finish_reason(choice: dict[str, Any]) -> str | None:
    """Returns OpenRouter's `native_finish_reason` from a streamed chat completion choice, when present."""
    # Wire input: an OpenRouter extension that the OpenAI SDK types do not declare.
    native = choice.get("native_finish_reason")
    return native if isinstance(native, str) and len(native) > 0 else None


def apply_openrouter_native_finish_reason(
    message: AssistantMessageBuilder,
    finish_reason: str,
    native_reason: str,
) -> None:
    """Records a native finish reason that differs from `finish_reason` on a message whose stop reason was
    already mapped from `finish_reason`: `raw_stop_reason` becomes e.g. "stop.MALFORMED_RESPONSE" (the
    format of openai-responses' "incomplete.max_output_tokens"), and an error message gets the native
    reason appended. The stop reason itself is unchanged.
    """
    if native_reason == finish_reason:
        return
    message.raw_stop_reason = f"{finish_reason}.{native_reason}"
    if message.stop_reason == "error" and message.error_message:
        # Appended, not replaced: retry matching relies on the finish_reason message prefix.
        message.error_message += f" (native_finish_reason: {native_reason})"


# Native reasons OpenRouter treats as a clean stop; see `flag_openrouter_empty_stop` for why they matter.
# Two of OpenRouter's entries are left out because their providers document them as failures:
# Gemini's "OTHER" ("stopped for a reason not otherwise specified"; `map_stop_reason` in
# google_shared.py maps it to an error) and DeepSeek's "insufficient_system_resource" ("interrupted
# due to insufficient resource").
#
# OpenRouter does not publish this list, so it can drift. It is an allowlist on purpose: a missing
# clean reason costs a few retries of an empty turn, while a missing failure reason would end the turn
# silently. Lowercase, compared case-insensitively, because upstreams differ in case ("STOP", "stop").
_CLEAN_STOP_NATIVE_REASONS = frozenset(
    {
        "stop",
        "end_turn",
        "stop_sequence",
        "eos",
        "complete",
        "completed",
        "pause_turn",
        "compaction",
        "end_of_stream",
    }
)


def flag_openrouter_empty_stop(message: AssistantMessageBuilder, native_reason: str) -> None:
    """Turns an OpenRouter `stop` that is really an upstream failure into a retryable error.

    OpenRouter reports any upstream finish reason it has no failure case for as `finish_reason: "stop"`,
    so some upstream failures arrive as `stop` and only `native_finish_reason` still names them.
    Without this, pidrei would end such a turn silently, often with nothing but a thinking block. Example
    as of November 2026: Gemini's transient MALFORMED_RESPONSE after streamed reasoning arrives as `stop`
    (OpenRouter already reports it as `finish_reason: "error"` when there is no output at all).

    A legitimately empty turn also arrives as `stop`, e.g. Claude ending with an empty `end_turn` after a
    tool result, and should still end quietly. Only the native reason tells the two apart: a turn without
    visible output (text or tool calls; reasoning does not count) is flagged unless its native reason is
    a clean stop. The message matches the retry pattern for OpenRouter's own "Provider returned an
    empty response".
    """
    if message.stop_reason != "stop" or native_reason.lower() in _CLEAN_STOP_NATIVE_REASONS:
        return
    has_visible_output = any(
        block.type == "toolCall" or (block.type == "text" and len(block.text.strip()) > 0) for block in message.content
    )
    if has_visible_output:
        return
    message.stop_reason = "error"
    message.error_message = f"OpenRouter returned an empty response (native_finish_reason: {native_reason})"
