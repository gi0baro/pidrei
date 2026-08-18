"""Pirate Mode

Demonstrates modifying the system prompt in `before_agent_start` to
dynamically change agent behavior based on extension state.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/pirate.py

Use /pirate to toggle pirate mode. When enabled, the agent responds like a
pirate.
"""

PIRATE_PROMPT = """

IMPORTANT: You are now in PIRATE MODE. You must:
- Speak like a stereotypical pirate in all responses
- Use phrases like "Arrr!", "Ahoy!", "Shiver me timbers!", "Avast!", "Ye scurvy dog!"
- Replace "my" with "me", "you" with "ye", "your" with "yer"
- Refer to the user as "matey" or "landlubber"
- End sentences with nautical expressions
- Still complete the actual task correctly, just in pirate speak
"""


def extension(pi):
    state = {"pirate_mode": False}

    # Register /pirate command to toggle pirate mode
    async def toggle(_args, ctx):
        state["pirate_mode"] = not state["pirate_mode"]
        ctx.ui.notify(
            "Arrr! Pirate mode enabled!" if state["pirate_mode"] else "Pirate mode disabled",
            "info",
        )

    pi.register_command("pirate", description="Toggle pirate mode (agent speaks like a pirate)", handler=toggle)

    # Append to system prompt when pirate mode is enabled
    async def on_before_agent_start(event, _ctx):
        if state["pirate_mode"]:
            return {"systemPrompt": event["systemPrompt"] + PIRATE_PROMPT}
        return None

    pi.on("before_agent_start", on_before_agent_start)
