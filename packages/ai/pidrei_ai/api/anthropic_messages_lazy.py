"""Port of pi's anthropic-messages.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def anthropic_messages_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import anthropic_messages

        return anthropic_messages

    return lazy_api(load)
