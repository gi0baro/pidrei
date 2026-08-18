"""Q&A extraction - extracts questions from assistant responses.

Demonstrates the "prompt generator" pattern:
1. /qna command gets the last assistant message
2. Shows a spinner while extracting (hides editor)
3. Loads the result into the editor for the user to fill in answers

Start pidrei with this extension:
    pidrei -e ./examples/extensions/qna.py
"""

import time

import tonio.colored as tonio

from pidrei.modes.interactive.components import BorderedLoader
from pidrei_ai.types import Context, StreamOptions, TextContent, UserMessage


SYSTEM_PROMPT = """You are a question extractor. Given text from a conversation, extract any questions that need answering and format them for the user to fill in.

Output format:
- List each question on its own line, prefixed with "Q: "
- After each question, add a blank line for the answer prefixed with "A: "
- If no questions are found, output "No questions found in the last message."

Example output:
Q: What is your preferred database?
A:

Q: Should we use TypeScript or JavaScript?
A:

Keep questions in the order they appeared. Be concise."""


def extension(pi):
    async def qna(_args, ctx):
        if ctx.mode != "tui":
            ctx.ui.notify("qna requires interactive mode", "error")
            return

        model = ctx.model
        if model is None:
            ctx.ui.notify("No model selected", "error")
            return

        # Find the last assistant message on the current branch
        last_assistant_text = None
        for entry in reversed(ctx.session_manager.get_branch()):
            if entry.get("type") != "message":
                continue
            message = entry.get("message")
            if getattr(message, "role", None) != "assistant":
                continue
            if message.stop_reason != "stop":
                ctx.ui.notify(f"Last assistant message incomplete ({message.stop_reason})", "error")
                return
            text_parts = [c.text for c in message.content if getattr(c, "type", None) == "text"]
            if text_parts:
                last_assistant_text = "\n".join(text_parts)
                break

        if last_assistant_text is None:
            ctx.ui.notify("No assistant messages found", "error")
            return

        # Run extraction with loader UI
        def factory(tui, theme, _kb, done):
            loader = BorderedLoader(tui, theme, f"Extracting questions using {model.id}...")
            loader.on_abort = lambda: done(None)

            # Do the work
            async def extract():
                try:
                    user_message = UserMessage(
                        content=[TextContent(text=last_assistant_text)],
                        timestamp=int(time.time() * 1000),
                    )

                    response = await ctx.model_registry.complete(
                        model,
                        Context(system_prompt=SYSTEM_PROMPT, messages=[user_message]),
                        StreamOptions(cancel=loader.signal),
                    )

                    if response.stop_reason == "aborted":
                        done(None)
                        return

                    done("\n".join(c.text for c in response.content if getattr(c, "type", None) == "text"))
                except Exception:
                    done(None)

            tonio.spawn.without_tracking(extract())
            return loader

        result = await ctx.ui.custom(factory)

        if result is None:
            ctx.ui.notify("Cancelled", "info")
            return

        ctx.ui.set_editor_text(result)
        ctx.ui.notify("Questions loaded. Edit and submit when ready.", "info")

    pi.register_command(
        "qna",
        handler=qna,
        description="Extract questions from last assistant message into editor",
    )
