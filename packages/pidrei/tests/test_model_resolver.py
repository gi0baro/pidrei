"""Mirror of pi coding-agent test/model-resolver.test.ts (plus max-thinking.test.ts's
CLI/settings assertions; its theme half is Phase 4)."""

from dataclasses import dataclass, replace

import pytest

from pidrei.cli.args import is_valid_thinking_level
from pidrei.core import model_resolver
from pidrei.core.model_resolver import (
    DEFAULT_MODEL_PER_PROVIDER,
    ModelReferenceRequest,
    ModelScopeDiagnostic,
    find_initial_model,
    resolve_cli_model,
    resolve_model_reference,
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

    def configured_auth_check(self):
        return self.has_configured_auth


def _reference_model(provider: str, id: str):
    return make_model(
        provider, id, base_url="https://example.invalid", reasoning=True, context_window=128000, max_tokens=8192
    )


REFERENCE_CATALOG = [
    _reference_model("anthropic", "claude-sonnet-4-5"),
    _reference_model("anthropic", "claude-sonnet-4-5-20250929"),
    _reference_model("anthropic", "claude-opus-4-8"),
    _reference_model("anthropic", "exact"),
    # Fuzzy-matches "shared"; an ambiguous exact "shared" must not fall through to it.
    _reference_model("anthropic", "shared-pro"),
    _reference_model("openai", "gpt-4o"),
    _reference_model("openrouter", "qwen/qwen3-coder:exacto"),
    _reference_model("openrouter", "openai/gpt-4o:extended"),
    _reference_model("gw-a", "shared"),
    _reference_model("gw-a", "anthropic/future"),
    _reference_model("gw-a", "anthropic/exact"),
    _reference_model("gw-a", "anthropic/claude-opus-4.8"),
    _reference_model("gw-a", "anthropic/claude-opus"),
    _reference_model("gw-b", "shared"),
    _reference_model("gw-b", "anthropic/future"),
    _reference_model("gw-b", "claude-opus-4-8"),
    _reference_model("gw-b", "gw-b/free"),
    _reference_model("gw-b", "zz:free"),
]


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    name: str
    text: str
    # {"model", "thinking_level"?, "warning"?} | {"ambiguous", "usable_count"} | "not-found"
    expected: dict | str
    usable: tuple[str, ...] = ()
    provider: str | None = None
    thinking_suffix: str = "strict"
    fuzzy: bool = True


REFERENCE_CASES = [
    # Reference syntax
    ReferenceCase("exact id", "claude-sonnet-4-5", {"model": "anthropic/claude-sonnet-4-5"}),
    ReferenceCase("exact id is case-insensitive", "GPT-4O", {"model": "openai/gpt-4o"}),
    ReferenceCase("fuzzy prefers aliases over dated ids", "sonnet", {"model": "anthropic/claude-sonnet-4-5"}),
    ReferenceCase("thinking suffix", "sonnet:high", {"model": "anthropic/claude-sonnet-4-5", "thinking_level": "high"}),
    ReferenceCase(
        "colon in a literal id is not a thinking suffix",
        "qwen/qwen3-coder:exacto",
        {"model": "openrouter/qwen/qwen3-coder:exacto"},
    ),
    ReferenceCase(
        "provider prefix, colon id, and thinking suffix",
        "openrouter/qwen/qwen3-coder:exacto:high",
        {"model": "openrouter/qwen/qwen3-coder:exacto", "thinking_level": "high"},
    ),
    ReferenceCase(
        "raw id wins when the provider prefix names no matching model",
        "openai/gpt-4o:extended",
        {"model": "openrouter/openai/gpt-4o:extended"},
    ),
    ReferenceCase(
        "fuzzy searches the prefixed provider first", "openrouter/qwen", {"model": "openrouter/qwen/qwen3-coder:exacto"}
    ),
    ReferenceCase("strict keeps an invalid suffix in the id", "gpt-4o:random", "not-found"),
    ReferenceCase(
        "lenient drops an invalid suffix with a warning",
        "gpt-4o:random",
        {"model": "openai/gpt-4o", "warning": 'Invalid thinking level "random"'},
        thinking_suffix="lenient",
    ),
    ReferenceCase(
        "lenient invalid outer suffix discards the inner thinking level",
        "qwen/qwen3-coder:exacto:high:random",
        {"model": "openrouter/qwen/qwen3-coder:exacto", "warning": 'Invalid thinking level "random"'},
        thinking_suffix="lenient",
    ),
    ReferenceCase("none keeps the suffix in the id", "gpt-4o:high", "not-found", thinking_suffix="none"),
    ReferenceCase("exact-only skips fuzzy matches", "sonnet", "not-found", fuzzy=False),
    ReferenceCase("empty text", "  ", "not-found"),
    # Exact matches across providers
    # #7327
    ReferenceCase("sole usable provider wins", "shared", {"model": "gw-b/shared"}, usable=("gw-b",)),
    ReferenceCase(
        "no usable provider is ambiguous instead of falling through to fuzzy",
        "shared",
        {"ambiguous": ["gw-a/shared", "gw-b/shared"], "usable_count": 0},
    ),
    ReferenceCase(
        "several usable providers are ambiguous",
        "shared",
        {"ambiguous": ["gw-a/shared", "gw-b/shared"], "usable_count": 2},
        usable=("gw-a", "gw-b"),
    ),
    ReferenceCase(
        "thinking suffix does not change provider selection",
        "shared:high",
        {"model": "gw-b/shared", "thinking_level": "high"},
        usable=("gw-b",),
    ),
    ReferenceCase(
        "thinking suffix does not bypass ambiguity",
        "shared:high",
        {"ambiguous": ["gw-a/shared", "gw-b/shared"], "usable_count": 2},
        usable=("gw-a", "gw-b"),
    ),
    # #10160
    ReferenceCase(
        "provider prefix without a direct model selects the usable raw id",
        "anthropic/future",
        {"model": "gw-b/anthropic/future"},
        usable=("gw-b",),
    ),
    # #5823
    ReferenceCase(
        "usable qualified match beats a usable raw id",
        "anthropic/exact",
        {"model": "anthropic/exact"},
        usable=("anthropic", "gw-a"),
    ),
    # #5643
    ReferenceCase(
        "usable raw id beats an unusable qualified match",
        "anthropic/exact",
        {"model": "gw-a/anthropic/exact"},
        usable=("gw-a",),
    ),
    ReferenceCase(
        "unusable qualified match is kept when nothing is usable", "anthropic/exact", {"model": "anthropic/exact"}
    ),
    # Like `mistral/codestral` -> mistral/codestral-latest despite gateway id `mistral/codestral`
    ReferenceCase(
        "named provider's usable fuzzy match beats a raw id elsewhere",
        "anthropic/claude-opus",
        {"model": "anthropic/claude-opus-4-8"},
        usable=("anthropic",),
    ),
    ReferenceCase(
        "usable raw id beats the named provider's unusable fuzzy match",
        "anthropic/claude-opus",
        {"model": "gw-a/anthropic/claude-opus"},
        usable=("gw-a",),
    ),
    ReferenceCase(
        "named provider's fuzzy match is kept when nothing is usable",
        "anthropic/claude-opus",
        {"model": "anthropic/claude-opus-4-8"},
    ),
    ReferenceCase("named provider's literal id beats its fuzzy matches", "gw-b/free", {"model": "gw-b/gw-b/free"}),
    # Fuzzy matches across providers
    ReferenceCase(
        "fuzzy prefers usable providers", "opus", {"model": "gw-a/anthropic/claude-opus-4.8"}, usable=("gw-a",)
    ),
    ReferenceCase(
        "fuzzy id tie across usable providers is ambiguous",
        "opus",
        {"ambiguous": ["anthropic/claude-opus-4-8", "gw-b/claude-opus-4-8"], "usable_count": 2},
        usable=("anthropic", "gw-a", "gw-b"),
    ),
    ReferenceCase(
        "fuzzy id tie resolves to the sole usable provider",
        "opus",
        {"model": "anthropic/claude-opus-4-8"},
        usable=("anthropic",),
    ),
    # Explicit provider constraint
    ReferenceCase(
        "explicit provider ignores usable providers elsewhere",
        "shared",
        {"model": "gw-a/shared"},
        provider="gw-a",
        usable=("gw-b",),
    ),
    ReferenceCase("explicit provider accepts its own prefix", "GW-B/shared", {"model": "gw-b/shared"}, provider="gw-b"),
    ReferenceCase("explicit provider allows fuzzy matches", "4o", {"model": "openai/gpt-4o"}, provider="openai"),
    ReferenceCase("explicit provider never leaves the provider", "gpt-4o", "not-found", provider="gw-a"),
]


def _model_ref(model) -> str:
    return f"{model.provider}/{model.id}"


@pytest.mark.parametrize("case", REFERENCE_CASES, ids=[case.name for case in REFERENCE_CASES])
def test_resolve_model_reference(case: ReferenceCase):
    request = ModelReferenceRequest(
        text=case.text, provider=case.provider, thinking_suffix=case.thinking_suffix, fuzzy=case.fuzzy
    )
    # Catalog order must never decide the result.
    for models in (REFERENCE_CATALOG, list(reversed(REFERENCE_CATALOG))):
        result = resolve_model_reference(request, models, lambda provider: provider in case.usable)
        expected = case.expected
        if expected == "not-found":
            assert result.status == "not-found"
        elif "ambiguous" in expected:
            assert result.status == "ambiguous"
            assert sorted(_model_ref(model) for model in result.candidates) == expected["ambiguous"]
            assert result.usable_count == expected["usable_count"]
        else:
            assert result.status == "resolved"
            assert _model_ref(result.model) == expected["model"]
            assert result.thinking_level == expected.get("thinking_level")
            if "warning" in expected:
                assert expected["warning"] in result.warning
            else:
                assert result.warning is None


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
        captured: list[str] = []
        original_write_stderr = model_resolver.write_stderr
        model_resolver.write_stderr = captured.append
        try:
            scoped_models = await resolve_model_scope(["missing"], runtime)
        finally:
            model_resolver.write_stderr = original_write_stderr

        assert scoped_models == []
        assert 'Warning: No models match pattern "missing"' in "".join(captured)

    @pytest.mark.tonio
    async def test_adds_every_provider_for_a_pattern_matching_the_same_id_on_several_providers(self):
        runtime = MockRuntime(
            [
                *ALL_MODELS,
                replace(MOCK_MODELS[1], provider="azure"),
                replace(MOCK_MODELS[1], provider="openai-codex", id="gpt-4o-mini"),
            ]
        )

        for pattern, thinking_level in (("gpt-4o:high", "high"), ("GPT-4o", None)):
            result = await resolve_model_scope_with_diagnostics([pattern], runtime)

            assert sorted(
                (f"{scoped.model.provider}/{scoped.model.id}", scoped.thinking_level) for scoped in result.scoped_models
            ) == [("azure/gpt-4o", thinking_level), ("openai/gpt-4o", thinking_level)]
            assert result.diagnostics == [
                ModelScopeDiagnostic(
                    type="warning",
                    code="ambiguous",
                    message=f'Pattern "{pattern}" matches several providers: azure, openai. Use provider/id to choose one.',
                    pattern=pattern,
                )
            ]

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


class TestDuplicateRawIdsBehindAKnownProviderPrefix:
    """pi's "duplicate raw ids behind a known provider prefix (#10160)" describe."""

    @staticmethod
    def _runtime(configured: list[str]):
        direct = replace(MOCK_MODELS[0], id="claude-sonnet-4-5", provider="anthropic")
        openrouter = replace(MOCK_MODELS[0], id="anthropic/claude-opus-5.5", provider="openrouter")
        vercel = replace(MOCK_MODELS[0], id="anthropic/claude-opus-5.5", provider="vercel-ai-gateway")
        return MockRuntime([direct, openrouter, vercel], configured=lambda provider: provider in configured)

    def test_selects_the_authenticated_gateway_with_or_without_a_thinking_suffix(self):
        for cli_model, thinking_level in (
            ("anthropic/claude-opus-5.5", None),
            ("anthropic/claude-opus-5.5:high", "high"),
        ):
            result = resolve_cli_model(cli_model=cli_model, model_runtime=self._runtime(["vercel-ai-gateway"]))

            assert result.error is None
            assert result.model.provider == "vercel-ai-gateway"
            assert result.thinking_level == thinking_level

    def test_reports_ambiguity_instead_of_choosing_by_catalog_order(self):
        result = resolve_cli_model(cli_model="anthropic/claude-opus-5.5", model_runtime=self._runtime([]))

        assert result.model is None
        assert "openrouter/anthropic/claude-opus-5.5, vercel-ai-gateway/anthropic/claude-opus-5.5" in result.error
        assert "No matching provider is authenticated. Use --provider or provider/model." in result.error

    def test_does_not_let_stored_credentials_choose_the_provider_for_api_key(self):
        result = resolve_cli_model(
            cli_model="anthropic/claude-opus-5.5",
            inline_api_key=True,
            model_runtime=self._runtime(["vercel-ai-gateway"]),
        )

        assert result.model is None
        assert "--api-key does not choose a provider." in result.error


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
        assert DEFAULT_MODEL_PER_PROVIDER["openai-codex"] == "gpt-6.1-sol"

    def test_zai_minimax_cerebras_and_ant_ling_defaults_track_current_models(self):
        assert DEFAULT_MODEL_PER_PROVIDER["zai"] == "glm-5.3"
        assert DEFAULT_MODEL_PER_PROVIDER["zai-coding-cn"] == "glm-5.3"
        assert DEFAULT_MODEL_PER_PROVIDER["minimax"] == "MiniMax-M2.7"
        assert DEFAULT_MODEL_PER_PROVIDER["minimax-cn"] == "MiniMax-M2.7"
        assert DEFAULT_MODEL_PER_PROVIDER["cerebras"] == "gpt-oss-120b"
        assert DEFAULT_MODEL_PER_PROVIDER["ant-ling"] == "Ring-2.6-1T"

    def test_builtin_chat_providers_have_defaults_in_their_generated_catalogs(self):
        for provider in get_builtin_providers():
            chat_models = get_builtin_models(provider)
            default_id = DEFAULT_MODEL_PER_PROVIDER.get(provider)
            if not chat_models:
                assert default_id is None, f"{provider} has no chat models and should have no chat default"
                continue
            assert any(model.id == default_id for model in chat_models), (
                f"{provider} default {default_id} should exist in its generated catalog"
            )

    def test_ai_gateway_default_tracks_current_model(self):
        assert DEFAULT_MODEL_PER_PROVIDER["vercel-ai-gateway"] == "zai/glm-5.1"

    def test_xai_default_tracks_current_model(self):
        assert DEFAULT_MODEL_PER_PROVIDER["xai"] == "grok-4.7"

    def test_qwen_token_plan_individual_default_tracks_current_model(self):
        assert DEFAULT_MODEL_PER_PROVIDER["qwen-token-plan-individual"] == "qwen3.8-max"

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


class TestPersistedDefaultModelScoping:
    """pi's "persisted default model scoping" describe. pi passes scopedModels
    to the AgentSession constructor; the helper session gets the same starting
    state via the public `set_scoped_models`."""

    async def _create_session(self, tmp_path, *, scoped: bool, persisted_scope: list[str] | None = None):
        from pidrei.core.agent_session import ScopedModel

        from .agent_session_helpers import create_agent_session

        sonnet = _sonnet()

        async def stream_fn(*_args, **_kwargs):
            raise AssertionError("no model request expected")

        session = await create_agent_session(str(tmp_path), stream_fn=stream_fn, model=sonnet)
        if persisted_scope:
            session.settings_manager.set_enabled_models(persisted_scope)
        if scoped:
            session.set_scoped_models([ScopedModel(model=sonnet)])
        return session

    @pytest.mark.tonio
    async def test_adds_a_persisted_default_to_an_existing_scoped_model_list(self, tmp_path):
        sonnet, opus = _sonnet(), _opus()
        session = await self._create_session(tmp_path, scoped=True, persisted_scope=[f"{sonnet.provider}/{sonnet.id}"])

        await session.set_model(opus, persist=True)

        assert session.settings_manager.get_default_provider() == opus.provider
        assert session.settings_manager.get_default_model() == opus.id
        assert [f"{scoped.model.provider}/{scoped.model.id}" for scoped in session.scoped_models] == [
            f"{sonnet.provider}/{sonnet.id}",
            f"{opus.provider}/{opus.id}",
        ]
        assert session.settings_manager.get_enabled_models() == [
            f"{sonnet.provider}/{sonnet.id}",
            f"{opus.provider}/{opus.id}",
        ]

    @pytest.mark.tonio
    async def test_does_not_create_a_scoped_model_list_when_all_models_are_available(self, tmp_path):
        session = await self._create_session(tmp_path, scoped=False)

        await session.set_model(_opus(), persist=True)

        assert session.scoped_models == []
        assert session.settings_manager.get_enabled_models() is None

    @pytest.mark.tonio
    async def test_keeps_session_only_model_changes_out_of_scope(self, tmp_path):
        sonnet = _sonnet()
        session = await self._create_session(tmp_path, scoped=True, persisted_scope=[f"{sonnet.provider}/{sonnet.id}"])

        await session.set_model(_opus(), persist=False)

        assert [f"{scoped.model.provider}/{scoped.model.id}" for scoped in session.scoped_models] == [
            f"{sonnet.provider}/{sonnet.id}"
        ]
        assert session.settings_manager.get_enabled_models() == [f"{sonnet.provider}/{sonnet.id}"]


def _sonnet():
    from pidrei_ai.providers.all import get_builtin_model

    return get_builtin_model("anthropic", "claude-sonnet-4-5")


def _opus():
    from pidrei_ai.providers.all import get_builtin_model

    return get_builtin_model("anthropic", "claude-opus-4-8")
