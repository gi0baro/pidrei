"""Port of pi's mistral-conversations.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def mistral_conversations_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import mistral_conversations

        return mistral_conversations

    return lazy_api(load)
