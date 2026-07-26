"""Port of pi's azure-openai-responses.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def azure_openai_responses_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import azure_openai_responses

        return azure_openai_responses

    return lazy_api(load)
