"""Port of pi's openai-responses.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def openai_responses_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import openai_responses

        return openai_responses

    return lazy_api(load)
