"""Inter-Extension Event Bus

Shows pi.events for communication between extensions. One extension can emit
events that other extensions listen to.

Usage: /emit [message] - emit an event on the bus

Start pidrei with this extension:
    pidrei -e ./examples/extensions/event_bus.py
"""


def extension(pi):
    # Store ctx for use in the bus handler, which only receives the data.
    state = {"ctx": None}

    # Listen for events from other extensions. Bus handlers are async and
    # fire-and-forget: emit() detaches them onto the runtime.
    async def on_notification(data) -> None:
        ctx = state["ctx"]
        if ctx is not None:
            ctx.ui.notify(f"Event from {data['from']}: {data['message']}", "info")

    pi.events.on("my:notification", on_notification)

    # Command to emit events (emits "my:notification" which the listener
    # above receives).
    async def handle_emit(args, _ctx) -> None:
        message = args.strip() or "hello"
        pi.events.emit("my:notification", {"message": message, "from": "/emit command"})
        # The listener above will show the notification.

    pi.register_command("emit", handler=handle_emit, description="Emit my:notification event (usage: /emit message)")

    # Example: emit on session start.
    async def on_session_start(_event, ctx) -> None:
        state["ctx"] = ctx
        pi.events.emit("my:notification", {"message": "Session started", "from": "event-bus-example"})

    pi.on("session_start", on_session_start)
