"""Model Status

Shows model changes in the status bar.

Demonstrates the `model_select` event, which fires when the model changes.
The payload carries `model`, `previousModel` (None on the first selection)
and `source` — "set" for /model or `pi.set_model()`, "cycle" for keyboard
cycling.

Deviation from pi: pi puts the `prev → next (source)` line on `console.log`
and notifies with just `Model: {id}` (suppressed when `source` is
"restore"). pidrei's TUI runtime has no console, so that line is folded
into the notification — and its agent session never emits a "restore"
source ("set"/"cycle" only), so the suppression branch has nothing to
guard.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/model_status.py
"""


def extension(pi):
    async def on_model_select(event, ctx) -> None:
        model = event["model"]
        previous = event["previousModel"]
        source = event["source"]

        # Format model identifiers
        next_id = f"{model.provider}/{model.id}"
        prev_id = f"{previous.provider}/{previous.id}" if previous is not None else "none"

        # Show notification on change, including where it came from
        ctx.ui.notify(f"Model: {prev_id} → {next_id} ({source})", "info")

        # Update status bar with current model
        ctx.ui.set_status("model", f"🤖 {model.id}")

    pi.on("model_select", on_model_select)
