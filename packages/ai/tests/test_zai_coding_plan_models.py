"""Mirror of pi's zai-coding-plan-models.test.ts.

Every case reads the vendored catalog.
"""

from pidrei_ai.providers.all import get_builtin_model


def test_exposes_glm_4_6v_on_the_china_coding_plan_catalog():
    model = get_builtin_model("zai-coding-cn", "glm-4.6v")

    assert model is not None
    assert model.id == "glm-4.6v"
    assert model.provider == "zai-coding-cn"
    assert model.api == "openai-completions"
    assert model.base_url == "https://open.bigmodel.cn/api/coding/paas/v4"
    assert model.reasoning is True
    assert model.input == ["text", "image"]
    assert (model.cost.input, model.cost.output, model.cost.cache_read, model.cost.cache_write) == (0.3, 0.9, 0, 0)
    assert model.context_window == 128000
    assert model.max_tokens == 32768
    assert model.compat.max_tokens_field == "max_tokens"
    assert model.compat.thinking_format == "zai"
    assert model.compat.zai_tool_stream is True


def test_uses_api_equivalent_reference_costs_for_coding_plan_models():
    def cost_of(provider: str, model_id: str) -> tuple:
        cost = get_builtin_model(provider, model_id).cost
        return (cost.input, cost.output, cost.cache_read, cost.cache_write)

    assert cost_of("zai", "glm-5.2") == (1.4, 4.4, 0.26, 0)
    assert cost_of("zai-coding-cn", "glm-5.1") == (1.4, 4.4, 0.26, 0)
    assert cost_of("zai-coding-cn", "glm-5v-turbo") == (1.2, 4, 0.24, 0)


def test_keeps_zero_costs_for_coding_plan_models_without_a_matching_api_price():
    for provider in ("zai", "zai-coding-cn"):
        for model_id in ("glm-5.2-highspeed", "glm-5.3"):
            cost = get_builtin_model(provider, model_id).cost
            assert (cost.input, cost.output, cost.cache_read, cost.cache_write) == (0, 0, 0, 0)
