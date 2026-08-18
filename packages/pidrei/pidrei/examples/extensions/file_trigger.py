"""File Trigger

Watches a trigger file and injects its contents into the conversation.
Useful for external systems to send messages to the agent.

pi uses Node's fs.watch; pidrei ships no fs-watch primitive, so a background
tonio task polls the file's mtime instead.

Usage:
    echo "Run the tests" > /tmp/agent-trigger.txt

Start pidrei with this extension:
    pidrei -e ./examples/extensions/file_trigger.py
"""

import tonio.colored as tonio
from tonio.colored import fs


TRIGGER_FILE = "/tmp/agent-trigger.txt"  # noqa: S108 - a well-known path is the point of the example
POLL_INTERVAL_SECONDS = 1.0


def extension(pi):
    state = {"watching": False}

    async def watch_loop() -> None:
        # Baseline mtime: content already present at startup does not trigger,
        # matching fs.watch, which only reports changes after the watch starts.
        last_mtime: float | None = None
        try:
            last_mtime = (await fs.Path(TRIGGER_FILE).stat()).st_mtime
        except OSError:
            pass  # File might not exist yet

        while True:
            await tonio.sleep(POLL_INTERVAL_SECONDS)
            try:
                mtime = (await fs.Path(TRIGGER_FILE).stat()).st_mtime
            except OSError:
                continue  # File might not exist yet
            if mtime == last_mtime:
                continue
            last_mtime = mtime

            try:
                content = (await fs.Path(TRIGGER_FILE).read_text(encoding="utf-8")).strip()
            except OSError:
                continue
            if not content:
                continue

            try:
                pi.send_message(
                    {
                        "customType": "file-trigger",
                        "content": f"External trigger: {content}",
                        "display": True,
                    },
                    {"triggerTurn": True},  # triggerTurn - get the LLM to respond
                )
            except RuntimeError:
                # The runtime went stale (reload or session replacement); the
                # watcher dies with it.
                return
            # Clear after reading. The write bumps mtime, but the now-empty
            # content is skipped on the next poll.
            await fs.Path(TRIGGER_FILE).write_text("", encoding="utf-8")

    async def on_session_start(_event, ctx) -> None:
        # session_start also fires on new and switched sessions; keep one watcher.
        if not state["watching"]:
            state["watching"] = True
            tonio.spawn.without_tracking(watch_loop())

        if ctx.has_ui:
            ctx.ui.notify(f"Watching {TRIGGER_FILE}", "info")

    pi.on("session_start", on_session_start)
