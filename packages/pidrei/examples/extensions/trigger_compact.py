"""Compaction trigger.

Auto-compacts the session the first turn context usage crosses a threshold,
and exposes `/trigger-compact` to do it on demand.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/trigger_compact.py
"""

COMPACT_THRESHOLD_TOKENS = 100_000


def extension(pi):
    # pi distinguishes undefined (no turn seen yet) from null; here `None`
    # covers both, which is the same thing: there is no previous reading to
    # have crossed from.
    state: dict[str, int | None] = {"previous_tokens": None}

    def trigger_compaction(ctx, custom_instructions: str | None = None) -> None:
        if ctx.has_ui:
            ctx.ui.notify("Compaction started", "info")

        def on_complete(_result) -> None:
            if ctx.has_ui:
                ctx.ui.notify("Compaction completed", "info")

        def on_error(error) -> None:
            if ctx.has_ui:
                ctx.ui.notify(f"Compaction failed: {error}", "error")

        ctx.compact(
            {
                "custom_instructions": custom_instructions,
                "on_complete": on_complete,
                "on_error": on_error,
            }
        )

    def on_turn_end(_event, ctx) -> None:
        usage = ctx.get_context_usage()
        current_tokens = getattr(usage, "tokens", None) if usage is not None else None
        if current_tokens is None:
            return

        previous_tokens = state["previous_tokens"]
        crossed_threshold = previous_tokens is not None and previous_tokens <= COMPACT_THRESHOLD_TOKENS
        state["previous_tokens"] = current_tokens
        if not crossed_threshold or current_tokens <= COMPACT_THRESHOLD_TOKENS:
            return
        trigger_compaction(ctx)

    async def run_command(args: str, ctx) -> None:
        trigger_compaction(ctx, args.strip() or None)

    pi.on("turn_end", on_turn_end)
    pi.register_command("trigger-compact", description="Trigger compaction immediately", handler=run_command)
