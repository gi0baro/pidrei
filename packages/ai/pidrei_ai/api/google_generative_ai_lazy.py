"""Port of pi's google-generative-ai.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def google_generative_ai_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import google_generative_ai

        return google_generative_ai

    return lazy_api(load)
