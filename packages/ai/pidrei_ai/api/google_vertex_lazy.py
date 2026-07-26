"""Port of pi's google-vertex.lazy.ts."""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def google_vertex_api() -> LazyApi:
    async def load():
        from pidrei_ai.api import google_vertex

        return google_vertex

    return lazy_api(load)
