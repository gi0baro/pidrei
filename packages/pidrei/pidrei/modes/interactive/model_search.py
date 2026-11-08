"""Mirror of pi coding-agent src/modes/interactive/model-search.ts.

Items are ``{"id", "provider", "name"?}`` records.
"""

import re


def get_model_search_text(item: dict) -> str:
    model_id = item["id"]
    provider = item["provider"]
    name = f" {item['name']}" if item.get("name") else ""
    return f"{model_id} {provider} {provider}/{model_id} {provider} {model_id}{name}"


def get_model_selector_search_text(item: dict) -> str:
    """Search text for the /model selector.

    Exact provider-prefixed queries should rank before proxy-provider IDs
    like openrouter/openai/gpt-5, so keep the bare model ID out of the
    leading position.
    """
    model_id = item["id"]
    provider = item["provider"]
    name = f" {item['name']}" if item.get("name") else ""
    return f"{provider} {provider}/{model_id} {provider} {model_id}{name}"


# Python's `\w` is letters, numbers and `_`, so this splits on everything but `\p{L}\p{N}`, as pi.
_NON_WORD_RE = re.compile(r"[\W_]+")


def _split_model_label(value: str) -> list[str]:
    return [word for word in _NON_WORD_RE.split(value.lower()) if word]


def get_distinct_model_name(model_id: str, name: str | None) -> str | None:
    """Returns the model name when it has words the ID does not.

    For example a renamed override like "MiniMax M3 [rasmus@openrouter, parasail]" for
    "minimax/minimax-m3". Names that only restate the ID with different casing, separators, or a
    dropped namespace/version suffix ("Claude Sonnet 4.5 (EU)" for
    "eu.anthropic.claude-sonnet-4-5-20250929-v1:0") return None.
    """
    name = name.strip() if name else ""
    if not name:
        return None
    id_words = _split_model_label(model_id)
    id_tokens = set(id_words)
    compact_id = "".join(id_words)

    # Single-character words must match a whole ID token: "4" is not covered by "mistral-small-2603".
    def is_covered(word: str) -> bool:
        return word in id_tokens or (len(word) > 1 and word in compact_id)

    return None if all(is_covered(word) for word in _split_model_label(name)) else name
