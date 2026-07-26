"""Port of pi's google-generative-ai.lazy.ts.

The adapter itself lands in PLAN.md Phase 5d. Until then the loader raises, so a
provider that advertises this API (only `opencode`, for 5 of its 58 models) fails
loudly on the affected models instead of silently dropping them from the catalog.
`lazy_api` turns the raise into a stream error event, like any setup failure.
"""

from pidrei_ai.api.lazy import LazyApi, lazy_api


def google_generative_ai_api() -> LazyApi:
    async def load():
        raise NotImplementedError("google-generative-ai adapter is not wired yet")

    return lazy_api(load)
