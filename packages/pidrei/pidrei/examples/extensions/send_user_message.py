"""Send User Message

Demonstrates `pi.send_user_message()` for sending user messages from
extensions. Unlike `pi.send_message()` which sends custom messages,
`send_user_message()` sends actual user messages that appear in the
conversation as if typed by the user.

Usage:
    /ask What is 2+2?     - Sends a user message (always triggers a turn)
    /steer Focus on X     - Sends while streaming with steer delivery
    /followup And then?   - Sends while streaming with followUp delivery

Start pidrei with this extension:
    pidrei -e ./examples/extensions/send_user_message.py
"""

from pidrei_ai.types import TextContent


def extension(pi):
    # Simple command that sends a user message
    async def ask(args, ctx):
        if not args.strip():
            ctx.ui.notify("Usage: /ask <message>", "warning")
            return

        # send_user_message always triggers a turn when not streaming.
        # If streaming, it raises (no deliverAs specified).
        if not ctx.is_idle():
            ctx.ui.notify("Agent is busy. Use /steer or /followup instead.", "warning")
            return

        pi.send_user_message(args)

    # Command that steers the agent mid-conversation
    async def steer(args, ctx):
        if not args.strip():
            ctx.ui.notify("Usage: /steer <message>", "warning")
            return

        if ctx.is_idle():
            # Not streaming, just send normally
            pi.send_user_message(args)
        else:
            # Streaming - use steer to interrupt
            pi.send_user_message(args, {"deliverAs": "steer"})

    # Command that queues a follow-up message
    async def followup(args, ctx):
        if not args.strip():
            ctx.ui.notify("Usage: /followup <message>", "warning")
            return

        if ctx.is_idle():
            # Not streaming, just send normally
            pi.send_user_message(args)
        else:
            # Streaming - queue as follow-up
            pi.send_user_message(args, {"deliverAs": "followUp"})
            ctx.ui.notify("Follow-up queued", "info")

    # Example with content list (text + images would go here)
    async def askwith(args, ctx):
        if not args.strip():
            ctx.ui.notify("Usage: /askwith <message>", "warning")
            return

        if not ctx.is_idle():
            ctx.ui.notify("Agent is busy", "warning")
            return

        # send_user_message accepts a str or a list of TextContent/ImageContent
        pi.send_user_message(
            [
                TextContent(text=f"User request: {args}"),
                TextContent(text="Please respond concisely."),
            ]
        )

    pi.register_command("ask", handler=ask, description="Send a user message to the agent")
    pi.register_command(
        "steer",
        handler=steer,
        description="Send a steering message (interrupts current processing)",
    )
    pi.register_command(
        "followup",
        handler=followup,
        description="Queue a follow-up message (waits for current processing)",
    )
    pi.register_command(
        "askwith",
        handler=askwith,
        description="Send a user message with structured content",
    )
