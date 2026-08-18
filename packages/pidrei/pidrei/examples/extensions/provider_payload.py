"""Provider payload log.

Appends every provider request payload, and each response's status and
headers, to `.pidrei/provider-payload.log` in the session's working directory
— a way to see exactly what goes over the wire. `before_provider_request` can
also replace the payload; the commented return shows how.

Start pidrei with this extension:
    pidrei -e ./examples/extensions/provider_payload.py
"""

import json
import os

import tonio.colored as tonio

from pidrei.config import CONFIG_DIR_NAME


def _append(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def extension(pi):
    async def log(ctx, text: str) -> None:
        log_file = os.path.join(ctx.cwd, CONFIG_DIR_NAME, "provider-payload.log")
        # pi appends with appendFileSync; blocking the event loop is not an
        # option here, so the write runs on the blocking pool.
        await tonio.spawn_blocking(_append, log_file, text)

    async def on_request(event, ctx):
        await log(ctx, f"{json.dumps(event['payload'], indent=2, default=str)}\n\n")

        # Optional: replace the payload instead of only logging it.
        # return {**event["payload"], "temperature": 0}

    async def on_response(event, ctx) -> None:
        headers = json.dumps(dict(event["headers"] or {}), default=str)
        await log(ctx, f"[{event['status']}] {headers}\n\n")

    pi.on("before_provider_request", on_request)
    pi.on("after_provider_response", on_response)
