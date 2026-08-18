"""Smoke test for the GitLab Duo provider, run outside the agent.

Run: python -m pidrei.examples.extensions.custom_provider_gitlab_duo.test [model-id] [--thinking]

With no arguments it streams from claude-sonnet-4-5-20250929; pass another
model id from MODELS, and --thinking to request reasoning.

Reads the gitlab-duo credential from auth.json; run /login gitlab-duo inside
pidrei (with the extension loaded) first.
"""

import json
import sys
import time

import tonio.colored as tonio

from pidrei.config import get_auth_path
from pidrei_ai.types import Context, SimpleStreamOptions, UserMessage

from . import MODELS, gitlab_duo_provider


def _read_auth() -> dict:
    with open(get_auth_path(), encoding="utf-8") as handle:
        return json.load(handle)


async def main() -> int:
    args = [arg for arg in sys.argv[1:] if arg != "--thinking"]
    model_id = args[0] if args else "claude-sonnet-4-5-20250929"
    use_thinking = "--thinking" in sys.argv[1:]

    model = next((entry for entry in MODELS if entry.id == model_id), None)
    if model is None:
        print(f"Unknown model: {model_id}", file=sys.stderr)
        print("Available:", ", ".join(entry.id for entry in MODELS), file=sys.stderr)
        return 1

    auth_data = await tonio.spawn_blocking(_read_auth)
    gitlab_cred = auth_data.get("gitlab-duo")
    if not isinstance(gitlab_cred, dict) or not gitlab_cred.get("access"):
        print("No gitlab-duo credentials. Run /login gitlab-duo first.", file=sys.stderr)
        return 1

    provider = gitlab_duo_provider()
    context = Context(
        messages=[UserMessage(content="Say hello in exactly 3 words.", timestamp=int(time.time() * 1000))]
    )

    print(f"Model: {model.id}, API: {model.api}, Thinking: {use_thinking}")

    stream = provider.stream_simple(
        model,
        context,
        SimpleStreamOptions(
            api_key=gitlab_cred["access"],
            max_tokens=100,
            reasoning="low" if use_thinking else None,
        ),
    )

    async for event in stream:
        if event.type == "thinking_start":
            print("[Thinking]")
        elif event.type == "thinking_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "thinking_end":
            print("\n[/Thinking]\n")
        elif event.type == "text_delta":
            print(event.delta, end="", flush=True)
        elif event.type == "error":
            print(f"\nError: {event.error.error_message}", file=sys.stderr)
        elif event.type == "done":
            print(f"\n\nDone! {event.reason} {event.message.usage}")
    return 0


if __name__ == "__main__":
    sys.exit(tonio.run(main()))
