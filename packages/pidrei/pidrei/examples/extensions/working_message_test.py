"""Working Message Persistence Test

Sets a custom working message and indicator on session start so you can
verify they survive across loader recreations (e.g. between agent turns).

Send a few messages in interactive mode: the working message should stay
"Working... (custom)" with a brown dot indicator every time the loader
appears, not revert to the default gray "Working...".

Start pidrei with this extension:
    pidrei -e ./examples/extensions/working_message_test.py
"""

CUSTOM_MESSAGE = "\x1b[38;2;155;86;63mWorking... (custom)\x1b[39m"
CUSTOM_INDICATOR = {"frames": ["\x1b[38;2;155;86;63m●\x1b[39m"]}


def extension(pi):
    async def on_session_start(_event, ctx) -> None:
        ctx.ui.set_working_message(CUSTOM_MESSAGE)
        ctx.ui.set_working_indicator(CUSTOM_INDICATOR)

    pi.on("session_start", on_session_start)
