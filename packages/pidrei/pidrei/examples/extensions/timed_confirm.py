"""Timed Confirm

Demonstrates timed dialogs with live countdown.

Commands:
- /timed - Shows confirm dialog that auto-cancels after 5 seconds with countdown
- /timed-select - Shows select dialog that auto-cancels after 10 seconds with countdown
- /timed-signal - Shows confirm using a CancelToken (manual approach)

Start pidrei with this extension:
    pidrei -e ./examples/extensions/timed_confirm.py
"""

import tonio.colored as tonio

from pidrei_ai.utils.cancel import CancelToken


def extension(pi):
    # Simple approach: use the timeout option (recommended). Timeouts are in
    # milliseconds, mirroring pi.
    async def run_timed(_args, ctx):
        confirmed = await ctx.ui.confirm(
            "Timed Confirmation",
            "This dialog will auto-cancel in 5 seconds. Confirm?",
            {"timeout": 5000},
        )

        if confirmed:
            ctx.ui.notify("Confirmed by user!", "info")
        else:
            ctx.ui.notify("Cancelled or timed out", "info")

    async def run_timed_select(_args, ctx):
        choice = await ctx.ui.select("Pick an option", ["Option A", "Option B", "Option C"], {"timeout": 10000})

        if choice:
            ctx.ui.notify(f"Selected: {choice}", "info")
        else:
            ctx.ui.notify("Selection cancelled or timed out", "info")

    # Manual approach: use a CancelToken (pidrei's AbortSignal) for more
    # control. A watchdog task cancels the token when the deadline passes;
    # settling the event first is what stops it after a user answer.
    async def run_timed_signal(_args, ctx):
        token = CancelToken()
        settled = tonio.Event()

        async def watchdog():
            await settled.wait(5)
            if not settled.is_set():
                token.cancel()

        watchdog_join = tonio.spawn(watchdog())

        ctx.ui.notify("Dialog will auto-cancel in 5 seconds...", "info")

        try:
            confirmed = await ctx.ui.confirm(
                "Timed Confirmation",
                "This dialog will auto-cancel in 5 seconds. Confirm?",
                {"signal": token},
            )
        finally:
            settled.set()
            await watchdog_join

        if confirmed:
            ctx.ui.notify("Confirmed by user!", "info")
        elif token.cancelled:
            ctx.ui.notify("Dialog timed out (auto-cancelled)", "warning")
        else:
            ctx.ui.notify("Cancelled by user", "info")

    pi.register_command(
        "timed",
        handler=run_timed,
        description="Show a timed confirmation dialog (auto-cancels in 5s with countdown)",
    )
    pi.register_command(
        "timed-select",
        handler=run_timed_select,
        description="Show a timed select dialog (auto-cancels in 10s with countdown)",
    )
    pi.register_command(
        "timed-signal",
        handler=run_timed_signal,
        description="Show a timed confirm using a CancelToken (manual approach)",
    )
