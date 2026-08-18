"""Widget Placement

Demonstrates the `placement` option of `ctx.ui.set_widget()`: widgets sit
above the editor by default, or below it with
`{"placement": "belowEditor"}`. A plain list of strings is wrapped in a
widget component automatically.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/widget_placement.py
"""


def extension(pi):
    async def on_session_start(_event, ctx) -> None:
        if not ctx.has_ui:
            return
        ctx.ui.set_widget("widget-above", ["Above editor widget"])
        ctx.ui.set_widget("widget-below", ["Below editor widget"], {"placement": "belowEditor"})

    pi.on("session_start", on_session_start)
