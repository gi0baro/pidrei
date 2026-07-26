"""Port of pi's bedrock-converse-stream.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def bedrock_converse_stream_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import bedrock_converse_stream

        return bedrock_converse_stream

    return lazy_api(load)
