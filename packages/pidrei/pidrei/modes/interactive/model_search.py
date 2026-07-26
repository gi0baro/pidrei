"""Mirror of pi coding-agent src/modes/interactive/model-search.ts.

Items are ``{"id", "provider", "name"?}`` records.
"""


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
