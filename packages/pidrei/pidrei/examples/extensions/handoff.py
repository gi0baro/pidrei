"""Handoff - transfer context to a new focused session.

Instead of compacting (which is lossy), handoff extracts what matters
for your next task and creates a new session with a generated prompt.

Usage:
    /handoff now implement this for teams as well
    /handoff execute phase one of the plan
    /handoff check other places that need this fix

The generated prompt appears as a draft in the editor for review/editing.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/handoff.py
"""

import time

import tonio.colored as tonio

from pidrei.core.compaction.utils import serialize_conversation
from pidrei.core.messages import convert_to_llm, create_compaction_summary_message
from pidrei.modes.interactive.components import BorderedLoader
from pidrei_ai.types import Context, StreamOptions, TextContent, UserMessage
from pidrei_ai.utils.uuid import uuidv7


SYSTEM_PROMPT = """You are a context transfer assistant. Given a conversation history and the user's goal for a new thread, generate a focused prompt that:

1. Summarizes relevant context from the conversation (decisions made, approaches taken, key findings)
2. Lists any relevant files that were discussed or modified
3. Clearly states the next task based on the user's goal
4. Is self-contained - the new thread should be able to proceed without the old conversation

Format your response as a prompt the user can send to start the new thread. Be concise but include all necessary context. Do not include any preamble like "Here's the prompt" - just output the prompt itself.

Example output format:
## Context
We've been working on X. Key decisions:
- Decision 1
- Decision 2

Files involved:
- path/to/file1.py
- path/to/file2.py

## Task
[Clear description of what to do next based on user's goal]"""


def entry_to_message(entry):
    if entry.get("type") == "message":
        return entry["message"]
    if entry.get("type") == "compaction":
        return create_compaction_summary_message(entry["summary"], entry["tokensBefore"], entry["timestamp"])
    return None


def get_handoff_messages(branch):
    """Gather the branch's messages. If the branch was compacted, include the
    compaction summary plus entries from firstKeptEntryId onward."""
    compaction_index = -1
    for i in range(len(branch) - 1, -1, -1):
        if branch[i].get("type") == "compaction":
            compaction_index = i
            break
    if compaction_index < 0:
        return [message for message in map(entry_to_message, branch) if message is not None]

    compaction = branch[compaction_index]
    first_kept_index = next(
        (i for i, entry in enumerate(branch) if entry["id"] == compaction.get("firstKeptEntryId")),
        -1,
    )
    compacted_branch = [
        compaction,
        *(branch[first_kept_index:compaction_index] if first_kept_index >= 0 else []),
        *branch[compaction_index + 1 :],
    ]
    return [message for message in map(entry_to_message, compacted_branch) if message is not None]


def extension(pi):
    async def handoff(args, ctx):
        if ctx.mode != "tui":
            ctx.ui.notify("handoff requires interactive mode", "error")
            return

        model = ctx.model
        if model is None:
            ctx.ui.notify("No model selected", "error")
            return

        goal = args.strip()
        if not goal:
            ctx.ui.notify("Usage: /handoff <goal for new thread>", "error")
            return

        # Gather conversation context from the current branch.
        messages = get_handoff_messages(ctx.session_manager.get_branch())

        if not messages:
            ctx.ui.notify("No conversation to hand off", "error")
            return

        # Convert to LLM format and serialize
        llm_messages = convert_to_llm(messages)
        conversation_text = serialize_conversation(llm_messages)
        current_session_file = ctx.session_manager.get_session_file()

        # Generate the handoff prompt with loader UI
        def factory(tui, theme, _kb, done):
            loader = BorderedLoader(tui, theme, "Generating handoff prompt...")
            loader.on_abort = lambda: done(None)

            async def generate():
                try:
                    user_message = UserMessage(
                        content=[
                            TextContent(
                                text=(
                                    f"## Conversation History\n\n{conversation_text}\n\n"
                                    f"## User's Goal for New Thread\n\n{goal}"
                                )
                            )
                        ],
                        timestamp=int(time.time() * 1000),
                    )

                    response = await ctx.model_registry.complete(
                        model,
                        Context(system_prompt=SYSTEM_PROMPT, messages=[user_message]),
                        StreamOptions(
                            cancel=loader.signal,
                            cache_retention="none",
                            session_id=uuidv7(),
                        ),
                    )

                    if response.stop_reason == "aborted":
                        done(None)
                        return

                    done("\n".join(c.text for c in response.content if getattr(c, "type", None) == "text"))
                except Exception as error:
                    ctx.ui.notify(f"Handoff generation failed: {error}", "error")
                    done(None)

            tonio.spawn.without_tracking(generate())
            return loader

        result = await ctx.ui.custom(factory)

        if result is None:
            ctx.ui.notify("Cancelled", "info")
            return

        # Let the user edit the generated prompt
        edited_prompt = await ctx.ui.editor("Edit handoff prompt", result)

        if edited_prompt is None:
            ctx.ui.notify("Cancelled", "info")
            return

        # Create a new session with parent tracking. Use the replacement-session
        # context for post-switch UI work; the original ctx is stale after a
        # successful session replacement.
        async def with_session(replacement_ctx):
            replacement_ctx.ui.set_editor_text(edited_prompt)
            replacement_ctx.ui.notify("Handoff ready. Submit when ready.", "info")

        new_session_result = await ctx.new_session(
            {
                "parent_session": current_session_file,
                "with_session": with_session,
            }
        )

        if new_session_result["cancelled"]:
            ctx.ui.notify("New session cancelled", "info")

    pi.register_command(
        "handoff",
        handler=handoff,
        description="Transfer context to a new focused session",
    )
