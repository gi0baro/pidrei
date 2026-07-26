"""Port of pi's openai-codex-responses.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def openai_codex_responses_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import openai_codex_responses

        return openai_codex_responses

    return lazy_api(load)
