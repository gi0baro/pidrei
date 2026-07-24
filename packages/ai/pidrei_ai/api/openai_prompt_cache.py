"""Port of pi's OpenAI prompt-cache-key clamp (packages/ai/src/api/openai-prompt-cache.ts)."""

OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def clamp_openai_prompt_cache_key(key: str | None) -> str | None:
    if key is None:
        return None
    if len(key) <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        return key
    return key[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH]
