"""Conversation summary in a custom UI.

`/summarize` serializes the current branch to plain text, asks a fixed
model (openai/gpt-5.2) for a structured summary, and shows the result in a
custom bordered Markdown component that closes on Enter/Escape.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/summarize.py
"""

import json
import time

from pidrei.modes.interactive.components import DynamicBorder
from pidrei.modes.interactive.theme import get_markdown_theme
from pidrei_ai.types import Context, SimpleStreamOptions, TextContent, UserMessage
from pidrei_ai.utils.uuid import uuidv7
from pidrei_tui import Container, Markdown, Text, matches_key


def extract_text_parts(content):
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    return [part.text for part in content if getattr(part, "type", None) == "text"]


def extract_tool_call_lines(content):
    if not isinstance(content, list):
        return []
    return [
        f"Tool {part.name} was called with args {json.dumps(part.arguments, default=str)}"
        for part in content
        if getattr(part, "type", None) == "toolCall"
    ]


def build_conversation_text(entries):
    sections = []

    for entry in entries:
        if entry.get("type") != "message":
            continue
        message = entry.get("message")
        role = getattr(message, "role", None)
        if role not in ("user", "assistant"):
            continue

        entry_lines = []
        text_parts = extract_text_parts(message.content)
        if text_parts:
            role_label = "User" if role == "user" else "Assistant"
            message_text = "\n".join(text_parts).strip()
            if message_text:
                entry_lines.append(f"{role_label}: {message_text}")

        if role == "assistant":
            entry_lines.extend(extract_tool_call_lines(message.content))

        if entry_lines:
            sections.append("\n".join(entry_lines))

    return "\n\n".join(sections)


def build_summary_prompt(conversation_text):
    return f"""\
Summarize this conversation so I can resume it later.
Include goals, key decisions, progress, open questions, and next steps.
Keep it concise and structured with headings.

<conversation>
{conversation_text}
</conversation>"""


class SummaryView:
    """Bordered Markdown view that closes on Enter/Escape."""

    def __init__(self, theme, summary, done):
        self._done = done
        self._container = Container()
        border = DynamicBorder(lambda s: theme.fg("accent", s))
        self._container.add_child(border)
        self._container.add_child(Text(theme.fg("accent", theme.bold("Conversation Summary")), 1, 0))
        self._container.add_child(Markdown(summary, 1, 1, get_markdown_theme()))
        self._container.add_child(Text(theme.fg("dim", "Press Enter or Esc to close"), 1, 0))
        self._container.add_child(border)

    def render(self, width):
        return self._container.render(width)

    def invalidate(self):
        self._container.invalidate()

    async def handle_input(self, data):
        if matches_key(data, "enter") or matches_key(data, "escape"):
            self._done(None)


async def show_summary_ui(summary, ctx):
    if ctx.mode != "tui":
        return

    await ctx.ui.custom(lambda _tui, theme, _kb, done: SummaryView(theme, summary, done))


def extension(pi):
    async def summarize(_args, ctx):
        branch = ctx.session_manager.get_branch()
        conversation_text = build_conversation_text(branch)

        if not conversation_text.strip():
            if ctx.has_ui:
                ctx.ui.notify("No conversation text found", "warning")
            return

        if ctx.has_ui:
            ctx.ui.notify("Preparing summary...", "info")

        model = ctx.model_registry.get_model("openai", "gpt-5.2")
        if model is None:
            if ctx.has_ui:
                ctx.ui.notify("Model openai/gpt-5.2 not found", "warning")
            return
        if not ctx.model_registry.has_configured_auth(model.provider):
            if ctx.has_ui:
                ctx.ui.notify("No authentication configured for openai/gpt-5.2", "warning")
            return

        summary_messages = [
            UserMessage(
                content=[TextContent(text=build_summary_prompt(conversation_text))],
                timestamp=int(time.time() * 1000),
            )
        ]

        # complete_simple is pi's `complete` with reasoningEffort: the unified
        # entry point that accepts a reasoning level.
        response = await ctx.model_registry.complete_simple(
            model,
            Context(messages=summary_messages),
            SimpleStreamOptions(
                reasoning="high",
                cache_retention="none",
                session_id=uuidv7(),
            ),
        )

        summary = "\n".join(c.text for c in response.content if getattr(c, "type", None) == "text")

        await show_summary_ui(summary, ctx)

    pi.register_command(
        "summarize",
        handler=summarize,
        description="Summarize the current conversation in a custom UI",
    )
