"""Mirror of pi ai test/openrouter-cost.test.ts."""

from dataclasses import replace

import pytest

from pidrei_ai.builders import UsageBuilder, UsageCostBuilder
from pidrei_ai.types import Model, ModelCost
from pidrei_ai.utils.openrouter_cost import apply_openrouter_reported_cost, is_openrouter_model


def usage(total: float = 1) -> UsageBuilder:
    return UsageBuilder(
        input=1,
        output=1,
        total_tokens=2,
        cost=UsageCostBuilder(input=0.25, output=0.75, total=total),
    )


_MODEL = Model(
    id="m",
    name="m",
    api="openai-completions",
    provider="custom",
    base_url="https://example.com/v1",
    reasoning=False,
    input=["text"],
    cost=ModelCost(),
    context_window=1,
    max_tokens=1,
)


class TestIsOpenRouterModel:
    def test_recognizes_both_the_built_in_provider_and_custom_providers_using_openrouters_endpoint(self):
        assert is_openrouter_model(replace(_MODEL, provider="openrouter", base_url="https://example.com")) is True
        assert (
            is_openrouter_model(replace(_MODEL, provider="openrouter-work", base_url="https://openrouter.ai/api/v1"))
            is True
        )
        assert is_openrouter_model(replace(_MODEL, provider="custom", base_url="https://example.com/v1")) is False


class TestApplyOpenRouterReportedCost:
    def test_uses_openrouters_reported_cost_for_non_byok_requests(self):
        value = usage()
        apply_openrouter_reported_cost(
            value,
            {
                "cost": 0,
                "is_byok": False,
                "cost_details": {
                    "upstream_inference_cost": 0,
                    "upstream_inference_prompt_cost": 0,
                    "upstream_inference_completions_cost": 0,
                },
            },
        )
        assert value.cost.total == 0
        assert value.cost.total_from_provider is True

    def test_adds_the_upstream_inference_cost_for_byok_requests(self):
        value = usage()
        apply_openrouter_reported_cost(
            value,
            {
                "cost": 0.1,
                "is_byok": True,
                "cost_details": {
                    "upstream_inference_cost": 0.9,
                    "upstream_inference_prompt_cost": 0.6,
                    "upstream_inference_completions_cost": 0.3,
                },
            },
        )
        assert value.cost.total == 1
        assert value.cost.total_from_provider is True

    def test_uses_upstream_cost_components_when_openrouter_omits_their_total(self):
        value = usage()
        apply_openrouter_reported_cost(
            value,
            {
                "cost": 0.1,
                "is_byok": True,
                "cost_details": {
                    "upstream_inference_cost": None,
                    "upstream_inference_prompt_cost": 0.6,
                    "upstream_inference_completions_cost": 0.3,
                },
            },
        )
        assert value.cost.total == pytest.approx(1)
        assert value.cost.total_from_provider is True

    def test_keeps_the_catalog_total_when_openrouters_accounting_is_incomplete(self):
        value = usage()
        value.cost.total_from_provider = False
        apply_openrouter_reported_cost(value, {"cost": None, "is_byok": False, "cost_details": None})
        apply_openrouter_reported_cost(value, {"cost": 0.1, "is_byok": True, "cost_details": None})
        assert value.cost.total == 1
        assert value.cost.total_from_provider is False
