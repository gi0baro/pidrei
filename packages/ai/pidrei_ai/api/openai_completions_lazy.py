"""Port of pi's openai-completions.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def openai_completions_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import openai_completions

        return openai_completions

    return lazy_api(load)
