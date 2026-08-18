"""Custom compaction.

Replaces the default compaction behavior with a full summary of the entire
context. Instead of keeping the last 20k tokens of conversation turns, this
extension:
1. Summarizes ALL messages (messages_to_summarize + turn_prefix_messages)
2. Discards all old turns completely, keeping only the summary

This example also demonstrates using a different model (Gemini Flash) for
summarization, which can be cheaper/faster than the main conversation model.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/custom_compaction.py
"""

import time

from pidrei.core.compaction import CompactionResult, serialize_conversation
from pidrei.core.messages import convert_to_llm
from pidrei_ai.types import Context, StreamOptions, TextContent, UserMessage
from pidrei_ai.utils.uuid import uuidv7


SUMMARY_INSTRUCTIONS = """\
1. The main goals and objectives discussed
2. Key decisions made and their rationale
3. Important code changes, file modifications, or technical details
4. Current state of any ongoing work
5. Any blockers, issues, or open questions
6. Next steps that were planned or suggested

Be thorough but concise. The summary will replace the ENTIRE conversation \
history, so include all information needed to continue the work effectively.

Format the summary as structured markdown with clear sections."""


def extension(pi):
    async def on_before_compact(event, ctx):
        ctx.ui.notify("Custom compaction extension triggered", "info")

        preparation = event["preparation"]
        signal = event["signal"]

        # Use Gemini Flash for summarization (cheaper/faster than most
        # conversation models)
        model = ctx.model_registry.find("google", "gemini-2.5-flash")
        if model is None:
            ctx.ui.notify("Could not find Gemini Flash model, using default compaction", "warning")
            return None

        # Combine all messages for full summary
        all_messages = [*preparation.messages_to_summarize, *preparation.turn_prefix_messages]

        ctx.ui.notify(
            f"Custom compaction: summarizing {len(all_messages)} messages "
            f"({preparation.tokens_before:,} tokens) with {model.id}...",
            "info",
        )

        # Convert messages to readable text format
        conversation_text = serialize_conversation(convert_to_llm(all_messages))

        # Include previous summary context if available
        previous_context = (
            f"\n\nPrevious session summary for context:\n{preparation.previous_summary}"
            if preparation.previous_summary
            else ""
        )

        # Build messages that ask for a comprehensive summary
        prompt = (
            f"You are a conversation summarizer. Create a comprehensive summary of this "
            f"conversation that captures:{previous_context}\n\n"
            f"{SUMMARY_INSTRUCTIONS}\n\n"
            f"<conversation>\n{conversation_text}\n</conversation>"
        )
        summary_messages = [UserMessage(content=[TextContent(text=prompt)], timestamp=int(time.time() * 1000))]

        try:
            # Pass the cancel token to honor abort requests (e.g. the user
            # cancels compaction)
            response = await ctx.model_registry.complete(
                model,
                Context(messages=summary_messages),
                StreamOptions(
                    max_tokens=8192,
                    cancel=signal,
                    cache_retention="none",
                    session_id=uuidv7(),
                ),
            )

            summary = "\n".join(c.text for c in response.content if c.type == "text")

            if not summary.strip():
                if signal is None or not signal.cancelled:
                    ctx.ui.notify("Compaction summary was empty, using default compaction", "warning")
                return None

            # Return compaction content - SessionManager adds id/parentId.
            # Use first_kept_entry_id from the preparation to keep recent
            # messages.
            return {
                "compaction": CompactionResult(
                    summary=summary,
                    first_kept_entry_id=preparation.first_kept_entry_id,
                    tokens_before=preparation.tokens_before,
                    usage=response.usage,
                )
            }
        except Exception as error:
            ctx.ui.notify(f"Compaction failed: {error}", "error")
            # Fall back to default compaction on error
            return None

    pi.on("session_before_compact", on_before_compact)
