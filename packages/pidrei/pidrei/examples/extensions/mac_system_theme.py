"""macOS System Theme Sync

Syncs the pidrei theme with the macOS system appearance (dark/light mode) by
polling `osascript` every two seconds. pi leans on JS `setInterval`; here the
poll is a tonio background task that ends cooperatively through an Event when
the session shuts down.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/mac_system_theme.py
"""

import tonio.colored as tonio


POLL_SECONDS = 2.0
_APPEARANCE_SCRIPT = 'tell application "System Events" to tell appearance preferences to return dark mode'


async def extension(pi):
    stopped = tonio.Event()

    async def is_dark_mode() -> bool:
        result = await pi.exec("osascript", ["-e", _APPEARANCE_SCRIPT])
        return result.code == 0 and result.stdout.strip() == "true"

    async def on_session_start(_event, ctx) -> None:
        current_theme = "dark" if await is_dark_mode() else "light"
        await ctx.ui.set_theme(current_theme)

        # One poller per extension instance: a new or resumed session loads
        # the extensions again (and this one's session_shutdown stops it).
        async def poll() -> None:
            nonlocal current_theme
            while True:
                await stopped.wait(POLL_SECONDS)
                if stopped.is_set():
                    return
                new_theme = "dark" if await is_dark_mode() else "light"
                if stopped.is_set():
                    return  # shut down during the check: this ctx is stale
                if new_theme != current_theme:
                    current_theme = new_theme
                    await ctx.ui.set_theme(current_theme)

        tonio.spawn.without_tracking(poll())

    async def on_session_shutdown(_event, _ctx) -> None:
        stopped.set()

    pi.on("session_start", on_session_start)
    pi.on("session_shutdown", on_session_shutdown)
