"""Mirror of pi coding-agent test/model-search.test.ts."""

from pidrei.modes.interactive.model_search import get_distinct_model_name, get_model_selector_search_text
from pidrei_tui import fuzzy_filter


# #5916
class TestGetDistinctModelName:
    def test_hides_names_that_restate_the_id(self):
        assert get_distinct_model_name("claude-sonnet-4-5", "Claude Sonnet 4.5") is None
        assert get_distinct_model_name("openai/gpt-oss-120b", "OpenAI: gpt-oss-120b") is None
        assert get_distinct_model_name("eu.anthropic.claude-sonnet-4-5-20250929-v1:0", "Claude Sonnet 4.5 (EU)") is None
        assert get_distinct_model_name("gpt-5", "") is None
        assert get_distinct_model_name("gpt-5", None) is None

    def test_shows_names_with_words_the_id_lacks(self):
        name = "MiniMax M3 [rasmus@openrouter, parasail]"
        assert get_distinct_model_name("minimax/minimax-m3", name) == name
        assert get_distinct_model_name("gemini-3.1-flash-lite-image", "Nano Banana 2 Lite") == "Nano Banana 2 Lite"

    def test_requires_single_character_words_to_match_a_whole_id_token(self):
        assert get_distinct_model_name("mistral-small-2603", "Mistral Small 4") == "Mistral Small 4"


class TestModelSelectorSearchRanking:
    # #11220
    def test_ranks_models_containing_the_query_above_scattered_matches(self):
        models = [
            {"provider": "openrouter", "id": "openai/gpt-oss-120b", "name": "OpenAI: gpt-oss-120b"},
            {"provider": "openrouter", "id": "anthropic/claude-sonnet-4.5", "name": "Anthropic: Claude Sonnet 4.5"},
            {"provider": "openrouter", "id": "anthropic/claude-opus-4.1", "name": "Anthropic: Claude Opus 4.1"},
            {"provider": "anthropic", "id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5"},
            {"provider": "anthropic", "id": "claude-opus-4-1", "name": "Claude Opus 4.1"},
        ]

        result = [m["id"] for m in fuzzy_filter(models, "opus", get_model_selector_search_text)]

        assert sorted(result[:2]) == ["anthropic/claude-opus-4.1", "claude-opus-4-1"]

    # #10065
    def test_ranks_the_typed_model_above_models_sharing_a_long_provider_prefix(self):
        models = [
            {"provider": "fireworks", "id": "accounts/fireworks/models/deepseek-v4-flash-0731"},
            {"provider": "fireworks", "id": "accounts/fireworks/models/kimi-k3"},
            {"provider": "fireworks", "id": "accounts/fireworks/routers/firerouter", "name": "FireRouter"},
        ]

        result = fuzzy_filter(models, "firerouter", get_model_selector_search_text)

        assert result[0]["id"] == "accounts/fireworks/routers/firerouter"
