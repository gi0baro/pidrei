"""Mirror of pi coding-agent test/model-resolver.test.ts (plus max-thinking.test.ts's
CLI/settings assertions; its theme half is Phase 4)."""

import io
import sys

import pytest

from pidrei.cli.args import is_valid_thinking_level
from pidrei.core.model_resolver import (
    DEFAULT_MODEL_PER_PROVIDER,
    ModelScopeDiagnostic,
    find_initial_model,
    parse_model_pattern,
    resolve_cli_model,
    resolve_model_scope,
    resolve_model_scope_with_diagnostics,
)
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.providers.all import get_builtin_models, get_builtin_providers
from tests.model_runtime_helpers import make_model


MOCK_MODELS = [
    make_model(
        "anthropic",
        "claude-sonnet-4-5",
        api="anthropic-messages",
        base_url="https://api.anthropic.com",
        name="Claude Sonnet 4.5",
        reasoning=True,
        input=["text", "image"],
        context_window=200000,
        max_tokens=8192,
    ),
    make_model(
        "openai",
        "gpt-4o",
        api="anthropic-messages",
        base_url="https://api.openai.com",
        name="GPT-4o",
        input=["text", "image"],
        context_window=128000,
        max_tokens=4096,
    ),
]

MOCK_OPENROUTER_MODELS = [
    make_model(
        "openrouter",
        "qwen/qwen3-coder:exacto",
        api="anthropic-messages",
        base_url="https://openrouter.ai/api/v1",
        name="Qwen3 Coder Exacto",
        reasoning=True,
        context_window=128000,
        max_tokens=8192,
    ),
    make_model(
        "openrouter",
        "openai/gpt-4o:extended",
        api="anthropic-messages",
        base_url="https://openrouter.ai/api/v1",
        name="GPT-4o Extended",
        input=["text", "image"],
        context_window=128000,
        max_tokens=4096,
    ),
]

ALL_MODELS = [*MOCK_MODELS, *MOCK_OPENROUTER_MODELS]


def _ambiguous_models() -> list:
    """The same bare id under two providers (pi's gpt-5.6-sol case)."""
    return [
        make_model(provider, "gpt-5.6-sol", api="anthropic-messages", name="GPT 5.6 Sol")
        for provider in ("azure-openai-responses", "openai-codex")
    ]


class MockRuntime:
    def __init__(self, models, *, configured=None):
        self._models = models
        self._configured = configured

    def get_models(self):
        return list(self._models)

    async def get_available(self, provider_id=None, options=None):
        return list(self._models)

    def get_available_snapshot(self):
        return list(self._models)

    def get_model(self, provider, model_id):
        return next((m for m in self._models if m.provider == provider and m.id == model_id), None)

    def has_configured_auth(self, provider):
        if self._configured is None:
            return True
        return self._configured(provider)


class TestParseModelPattern:
    def test_exact_match_returns_model_with_none_thinking_level(self):
        result = parse_model_pattern("claude-sonnet-4-5", ALL_MODELS)
        assert result.model.id == "claude-sonnet-4-5"
        assert result.thinking_level is None
        assert result.warning is None

    def test_partial_match_returns_best_model_with_none_thinking_level(self):
        result = parse_model_pattern("sonnet", ALL_MODELS)
        assert result.model.id == "claude-sonnet-4-5"
        assert result.thinking_level is None
        assert result.warning is None

    def test_no_match_returns_none_model_and_thinking_level(self):
        result = parse_model_pattern("nonexistent", ALL_MODELS)
        assert result.model is None
        assert result.thinking_level is None
        assert result.warning is None

    def test_sonnet_high_returns_sonnet_with_high_thinking_level(self):
        result = parse_model_pattern("sonnet:high", ALL_MODELS)
        assert result.model.id == "claude-sonnet-4-5"
        assert result.thinking_level == "high"
        assert result.warning is None

    def test_gpt_4o_medium_returns_gpt_4o_with_medium_thinking_level(self):
        result = parse_model_pattern("gpt-4o:medium", ALL_MODELS)
        assert result.model.id == "gpt-4o"
        assert result.thinking_level == "medium"
        assert result.warning is None

    def test_all_valid_thinking_levels_work(self):
        for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
            result = parse_model_pattern(f"sonnet:{level}", ALL_MODELS)
            assert result.model.id == "claude-sonnet-4-5"
            assert result.thinking_level == level
            assert result.warning is None

    def test_sonnet_random_returns_sonnet_with_none_thinking_level_and_warning(self):
        result = parse_model_pattern("sonnet:random", ALL_MODELS)
        assert result.model.id == "claude-sonnet-4-5"
        assert result.thinking_level is None
        assert "Invalid thinking level" in result.warning
        assert "random" in result.warning

    def test_gpt_4o_invalid_returns_gpt_4o_with_none_thinking_level_and_warning(self):
        result = parse_model_pattern("gpt-4o:invalid", ALL_MODELS)
        assert result.model.id == "gpt-4o"
        assert result.thinking_level is None
        assert "Invalid thinking level" in result.warning

    def test_qwen3_coder_exacto_matches_the_model_with_none_thinking_level(self):
        result = parse_model_pattern("qwen/qwen3-coder:exacto", ALL_MODELS)
        assert result.model.id == "qwen/qwen3-coder:exacto"
        assert result.thinking_level is None
        assert result.warning is None

    def test_openrouter_qwen_qwen3_coder_exacto_matches_with_provider_prefix(self):
        result = parse_model_pattern("openrouter/qwen/qwen3-coder:exacto", ALL_MODELS)
        assert result.model.id == "qwen/qwen3-coder:exacto"
        assert result.model.provider == "openrouter"
        assert result.thinking_level is None
        assert result.warning is None

    def test_qwen3_coder_exacto_high_matches_model_with_high_thinking_level(self):
        result = parse_model_pattern("qwen/qwen3-coder:exacto:high", ALL_MODELS)
        assert result.model.id == "qwen/qwen3-coder:exacto"
        assert result.thinking_level == "high"
        assert result.warning is None

    def test_openrouter_qwen3_coder_exacto_high_matches_with_provider_and_thinking_level(self):
        result = parse_model_pattern("openrouter/qwen/qwen3-coder:exacto:high", ALL_MODELS)
        assert result.model.id == "qwen/qwen3-coder:exacto"
        assert result.model.provider == "openrouter"
        assert result.thinking_level == "high"
        assert result.warning is None

    def test_gpt_4o_extended_matches_the_extended_model_with_none_thinking_level(self):
        result = parse_model_pattern("openai/gpt-4o:extended", ALL_MODELS)
        assert result.model.id == "openai/gpt-4o:extended"
        assert result.thinking_level is None
        assert result.warning is None

    def test_qwen3_coder_exacto_random_returns_model_with_none_thinking_level_and_warning(self):
        result = parse_model_pattern("qwen/qwen3-coder:exacto:random", ALL_MODELS)
        assert result.model.id == "qwen/qwen3-coder:exacto"
        assert result.thinking_level is None
        assert "Invalid thinking level" in result.warning
        assert "random" in result.warning

    def test_qwen3_coder_exacto_high_random_returns_model_with_none_thinking_level_and_warning(self):
        result = parse_model_pattern("qwen/qwen3-coder:exacto:high:random", ALL_MODELS)
        assert result.model.id == "qwen/qwen3-coder:exacto"
        assert result.thinking_level is None
        assert "Invalid thinking level" in result.warning
        assert "random" in result.warning

    def test_empty_pattern_matches_via_partial_matching(self):
        result = parse_model_pattern("", ALL_MODELS)
        assert result.model is not None
        assert result.thinking_level is None

    def test_pattern_ending_with_colon_treats_empty_suffix_as_invalid(self):
        result = parse_model_pattern("sonnet:", ALL_MODELS)
        assert result.model.id == "claude-sonnet-4-5"
        assert "Invalid thinking level" in result.warning


class TestResolveModelScopeWithDiagnostics:
    @pytest.mark.tonio
    async def test_returns_scoped_models_and_structured_diagnostics(self):
        runtime = MockRuntime(ALL_MODELS)

        result = await resolve_model_scope_with_diagnostics(["sonnet:high", "gpt-4o:invalid", "missing"], runtime)

        assert [scoped.model.id for scoped in result.scoped_models] == ["claude-sonnet-4-5", "gpt-4o"]
        assert result.scoped_models[0].thinking_level == "high"
        assert result.scoped_models[1].thinking_level is None
        assert result.diagnostics == [
            ModelScopeDiagnostic(
                type="warning",
                code="invalid-thinking-level",
                message='Invalid thinking level "invalid" in pattern "gpt-4o:invalid". Using default instead.',
                pattern="gpt-4o:invalid",
            ),
            ModelScopeDiagnostic(
                type="warning", code="no-match", message='No models match pattern "missing"', pattern="missing"
            ),
        ]

    @pytest.mark.tonio
    async def test_resolve_model_scope_preserves_cli_warning_output(self):
        runtime = MockRuntime(ALL_MODELS)
        captured = io.StringIO()
        original_stderr = sys.stderr
        sys.stderr = captured
        try:
            scoped_models = await resolve_model_scope(["missing"], runtime)
        finally:
            sys.stderr = original_stderr

        assert scoped_models == []
        assert 'Warning: No models match pattern "missing"' in captured.getvalue()

    @pytest.mark.tonio
    async def test_resolves_bracketed_model_ids_as_exact_references_before_glob_matching(self):
        bracketed = make_model(
            "custom",
            "bracketed-model[1m]",
            api="anthropic-messages",
            base_url="https://example.invalid",
            name="Bracketed Model",
            reasoning=True,
            context_window=128000,
            max_tokens=8192,
        )
        runtime = MockRuntime([*ALL_MODELS, bracketed])

        result = await resolve_model_scope_with_diagnostics(["custom/bracketed-model[1m]"], runtime)

        assert [scoped.model.id for scoped in result.scoped_models] == ["bracketed-model[1m]"]
        assert result.diagnostics == []

    @pytest.mark.tonio
    async def test_resolves_bracketed_model_ids_with_thinking_levels_as_exact_references(self):
        bracketed = make_model(
            "custom",
            "bracketed-model[1m]",
            api="anthropic-messages",
            base_url="https://example.invalid",
            name="Bracketed Model",
            reasoning=True,
            context_window=128000,
            max_tokens=8192,
        )
        runtime = MockRuntime([*ALL_MODELS, bracketed])

        result = await resolve_model_scope_with_diagnostics(["custom/bracketed-model[1m]:high"], runtime)

        assert [scoped.model.id for scoped in result.scoped_models] == ["bracketed-model[1m]"]
        assert result.scoped_models[0].thinking_level == "high"
        assert result.diagnostics == []


class TestResolveCliModel:
    def test_resolves_model_provider_id_without_provider_flag(self):
        result = resolve_cli_model(cli_model="openai/gpt-4o", model_runtime=MockRuntime(ALL_MODELS))
        assert result.error is None
        assert result.model.provider == "openai"
        assert result.model.id == "gpt-4o"

    def test_resolves_fuzzy_patterns_within_an_explicit_provider(self):
        result = resolve_cli_model(cli_provider="openai", cli_model="4o", model_runtime=MockRuntime(ALL_MODELS))
        assert result.error is None
        assert result.model.provider == "openai"
        assert result.model.id == "gpt-4o"

    def test_supports_model_pattern_thinking_without_explicit_thinking_flag(self):
        result = resolve_cli_model(cli_model="sonnet:high", model_runtime=MockRuntime(ALL_MODELS))
        assert result.error is None
        assert result.model.id == "claude-sonnet-4-5"
        assert result.thinking_level == "high"

    def test_prefers_exact_model_id_match_over_provider_inference(self):
        result = resolve_cli_model(cli_model="openai/gpt-4o:extended", model_runtime=MockRuntime(ALL_MODELS))
        assert result.error is None
        assert result.model.provider == "openrouter"
        assert result.model.id == "openai/gpt-4o:extended"

    def test_does_not_strip_invalid_suffix_as_thinking_level_in_model_flag(self):
        result = resolve_cli_model(
            cli_provider="openai", cli_model="gpt-4o:extended", model_runtime=MockRuntime(ALL_MODELS)
        )
        assert result.error is None
        assert result.model.provider == "openai"
        assert result.model.id == "gpt-4o:extended"

    def test_allows_custom_model_ids_for_explicit_providers_without_double_prefixing(self):
        result = resolve_cli_model(
            cli_provider="openrouter", cli_model="openrouter/openai/ghost-model", model_runtime=MockRuntime(ALL_MODELS)
        )
        assert result.error is None
        assert result.model.provider == "openrouter"
        assert result.model.id == "openai/ghost-model"

    def test_returns_a_clear_error_when_there_are_no_models(self):
        result = resolve_cli_model(cli_provider="openai", cli_model="gpt-4o", model_runtime=MockRuntime([]))
        assert result.model is None
        assert "No models available" in result.error

    def test_prefers_the_sole_authenticated_provider_for_an_ambiguous_bare_exact_model_id(self):
        result = resolve_cli_model(
            cli_model="gpt-5.6-sol",
            model_runtime=MockRuntime(_ambiguous_models(), configured=lambda provider: provider == "openai-codex"),
        )
        assert result.error is None
        assert result.model.provider == "openai-codex"
        assert result.model.id == "gpt-5.6-sol"

    def test_requires_an_explicit_provider_for_an_ambiguous_bare_exact_model_id(self):
        result = resolve_cli_model(
            cli_model="gpt-5.6-sol",
            model_runtime=MockRuntime(_ambiguous_models(), configured=lambda _provider: False),
        )
        assert result.model is None
        assert 'Model "gpt-5.6-sol" is ambiguous across providers' in result.error
        assert "azure-openai-responses/gpt-5.6-sol" in result.error
        assert "openai-codex/gpt-5.6-sol" in result.error
        assert "Use --provider or provider/model" in result.error

    def test_prefers_provider_model_split_over_gateway_model_with_matching_id(self):
        zai_model = make_model(
            "zai",
            "glm-5",
            api="anthropic-messages",
            base_url="https://open.bigmodel.cn/api/paas/v4",
            name="GLM-5",
            reasoning=True,
            context_window=128000,
            max_tokens=8192,
        )
        gateway_model = make_model(
            "vercel-ai-gateway",
            "zai/glm-5",
            api="anthropic-messages",
            base_url="https://ai-gateway.vercel.sh",
            name="GLM-5",
            reasoning=True,
            context_window=128000,
            max_tokens=8192,
        )
        runtime = MockRuntime([*ALL_MODELS, zai_model, gateway_model])

        result = resolve_cli_model(cli_model="zai/glm-5", model_runtime=runtime)

        assert result.error is None
        assert result.model.provider == "zai"
        assert result.model.id == "glm-5"

    def test_prefers_an_authenticated_exact_raw_model_id_over_an_unauthenticated_inferred_provider(self):
        commandcode_model = make_model(
            "commandcode",
            "xiaomi/mimo-v2.5-pro",
            api="anthropic-messages",
            base_url="https://example.invalid",
            name="Xiaomi MiMo via Commandcode",
            context_window=128000,
            max_tokens=8192,
        )
        xiaomi_model = make_model(
            "xiaomi",
            "mimo-v2.5-pro",
            api="anthropic-messages",
            base_url="https://api.xiaomimimo.com",
            name="Xiaomi MiMo",
            context_window=128000,
            max_tokens=8192,
        )
        runtime = MockRuntime(
            [*ALL_MODELS, commandcode_model, xiaomi_model], configured=lambda provider: provider == "commandcode"
        )

        result = resolve_cli_model(cli_model="xiaomi/mimo-v2.5-pro", model_runtime=runtime)

        assert result.error is None
        assert result.model.provider == "commandcode"
        assert result.model.id == "xiaomi/mimo-v2.5-pro"

    def test_resolves_provider_prefixed_fuzzy_patterns(self):
        result = resolve_cli_model(cli_model="openrouter/qwen", model_runtime=MockRuntime(ALL_MODELS))
        assert result.error is None
        assert result.model.provider == "openrouter"
        assert result.model.id == "qwen/qwen3-coder:exacto"


class TestCustomModelFallbackWithThinkingSuffix:
    """Mirror of the #5552 suite: provider exists but the model id doesn't."""

    def _runtime(self):
        neuralwatt_model = make_model(
            "neuralwatt",
            "some-base-model",
            api="anthropic-messages",
            base_url="https://api.neuralwatt.com",
            name="Some Base Model",
            context_window=128000,
            max_tokens=8192,
        )
        return MockRuntime([*ALL_MODELS, neuralwatt_model])

    def test_strips_thinking_suffix_from_custom_model_id_in_fallback_path(self):
        result = resolve_cli_model(cli_model="neuralwatt/zai-org/GLM-5.1-FP8:high", model_runtime=self._runtime())
        assert result.error is None
        assert result.model.provider == "neuralwatt"
        # The :high suffix must NOT leak into the model id sent to the API
        assert result.model.id == "zai-org/GLM-5.1-FP8"
        assert result.model.reasoning is True
        assert result.thinking_level == "high"

    def test_custom_model_without_thinking_suffix_works_normally_in_fallback_path(self):
        result = resolve_cli_model(cli_model="neuralwatt/zai-org/GLM-5.1-FP8", model_runtime=self._runtime())
        assert result.error is None
        assert result.model.provider == "neuralwatt"
        assert result.model.id == "zai-org/GLM-5.1-FP8"
        assert result.thinking_level is None

    def test_all_valid_thinking_levels_work_in_fallback_path(self):
        for level in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
            result = resolve_cli_model(
                cli_model=f"neuralwatt/zai-org/GLM-5.1-FP8:{level}", model_runtime=self._runtime()
            )
            assert result.error is None
            assert result.model.id == "zai-org/GLM-5.1-FP8"
            assert result.thinking_level == level

    def test_invalid_thinking_suffix_on_custom_model_is_treated_as_part_of_model_id(self):
        result = resolve_cli_model(cli_model="neuralwatt/zai-org/GLM-5.1-FP8:banana", model_runtime=self._runtime())
        assert result.error is None
        assert result.model.provider == "neuralwatt"
        # Invalid suffix stays in the id (it's not a thinking level)
        assert result.model.id == "zai-org/GLM-5.1-FP8:banana"
        assert result.thinking_level is None

    def test_explicit_provider_with_custom_model_thinking_strips_suffix_correctly(self):
        result = resolve_cli_model(
            cli_provider="neuralwatt", cli_model="zai-org/GLM-5.1-FP8:high", model_runtime=self._runtime()
        )
        assert result.error is None
        assert result.model.provider == "neuralwatt"
        assert result.model.id == "zai-org/GLM-5.1-FP8"
        assert result.thinking_level == "high"

    def test_with_explicit_thinking_flag_suffix_is_kept_as_part_of_model_id(self):
        result = resolve_cli_model(
            cli_model="neuralwatt/zai-org/GLM-5.1-FP8:high", cli_thinking="medium", model_runtime=self._runtime()
        )
        assert result.error is None
        assert result.model.provider == "neuralwatt"
        # :high is kept as part of the model id since --thinking was explicit
        assert result.model.id == "zai-org/GLM-5.1-FP8:high"
        assert result.thinking_level is None


class TestDefaultModelSelection:
    def test_openai_defaults_track_current_models(self):
        assert DEFAULT_MODEL_PER_PROVIDER["openai"] == "gpt-5.5"
        assert DEFAULT_MODEL_PER_PROVIDER["openai-codex"] == "gpt-5.5"

    def test_zai_minimax_cerebras_and_ant_ling_defaults_track_current_models(self):
        assert DEFAULT_MODEL_PER_PROVIDER["zai"] == "glm-5.3"
        assert DEFAULT_MODEL_PER_PROVIDER["zai-coding-cn"] == "glm-5.3"
        assert DEFAULT_MODEL_PER_PROVIDER["minimax"] == "MiniMax-M2.7"
        assert DEFAULT_MODEL_PER_PROVIDER["minimax-cn"] == "MiniMax-M2.7"
        assert DEFAULT_MODEL_PER_PROVIDER["cerebras"] == "gpt-oss-120b"
        assert DEFAULT_MODEL_PER_PROVIDER["ant-ling"] == "Ring-2.6-1T"

    def test_builtin_defaults_exist_in_generated_provider_catalogs(self):
        for provider in get_builtin_providers():
            default_id = DEFAULT_MODEL_PER_PROVIDER[provider]
            assert any(model.id == default_id for model in get_builtin_models(provider)), (
                f"{provider} default {default_id} should exist in its generated catalog"
            )

    def test_ai_gateway_default_tracks_current_model(self):
        assert DEFAULT_MODEL_PER_PROVIDER["vercel-ai-gateway"] == "zai/glm-5.1"

    def test_xai_default_tracks_current_model(self):
        assert DEFAULT_MODEL_PER_PROVIDER["xai"] == "grok-4.6"

    def test_qwen_token_plan_individual_default_tracks_current_model(self):
        assert DEFAULT_MODEL_PER_PROVIDER["qwen-token-plan-individual"] == "qwen3.8-max"

    @pytest.mark.tonio
    async def test_find_initial_model_accepts_explicit_provider_custom_model_ids(self):
        result = await find_initial_model(
            cli_provider="openrouter",
            cli_model="openrouter/openai/ghost-model",
            scoped_models=[],
            is_continuing=False,
            model_runtime=MockRuntime(ALL_MODELS),
        )
        assert result.model.provider == "openrouter"
        assert result.model.id == "openai/ghost-model"

    @pytest.mark.tonio
    async def test_find_initial_model_selects_ai_gateway_default_when_available(self):
        ai_gateway_model = make_model(
            "vercel-ai-gateway",
            "zai/glm-5.1",
            api="anthropic-messages",
            base_url="https://ai-gateway.vercel.sh",
            name="GLM 5.1",
            reasoning=True,
            input=["text", "image"],
            context_window=200000,
            max_tokens=8192,
        )
        result = await find_initial_model(
            scoped_models=[],
            is_continuing=False,
            model_runtime=MockRuntime([ai_gateway_model]),
        )
        assert result.model.provider == "vercel-ai-gateway"
        assert result.model.id == "zai/glm-5.1"

    @pytest.mark.tonio
    async def test_find_initial_model_ignores_an_unauthenticated_saved_default(self):
        saved_deepseek = make_model(
            "deepseek",
            "deepseek-v4-flash",
            api="anthropic-messages",
            base_url="https://api.deepseek.com",
            name="DeepSeek V4 Flash",
            reasoning=True,
            context_window=128000,
            max_tokens=8192,
        )
        local_deepseek = make_model(
            "spark-two",
            "deepseek-v4-flash",
            api="anthropic-messages",
            base_url="http://spark-two:8000/v1",
            name="DeepSeek V4 Flash",
            reasoning=True,
            context_window=128000,
            max_tokens=8192,
        )

        class Runtime:
            def get_model(self, provider, model_id):
                if provider == "deepseek" and model_id == "deepseek-v4-flash":
                    return saved_deepseek
                return None

            def has_configured_auth(self, provider):
                return provider == "spark-two"

            def get_available_snapshot(self):
                return [local_deepseek]

        result = await find_initial_model(
            scoped_models=[],
            is_continuing=False,
            default_provider="deepseek",
            default_model_id="deepseek-v4-flash",
            model_runtime=Runtime(),
        )

        assert result.model.provider == "spark-two"
        assert result.model.id == "deepseek-v4-flash"


class TestMaxThinkingLevel:
    """CLI/settings half of pi's max-thinking.test.ts (theme half is Phase 4)."""

    def test_is_accepted_by_cli_and_settings(self):
        assert is_valid_thinking_level("max") is True

        settings = SettingsManager.in_memory()
        settings.set_default_thinking_level("max")
        # No flush: the in-memory backend writes inline, so there is
        # nothing queued to wait for.
        assert settings.get_default_thinking_level() == "max"
