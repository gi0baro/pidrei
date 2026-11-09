"""Mirror of pi's model-cost-tiers.test.ts.

pi's helpers live in catalog/sources/vercel-ai-gateway.ts and
catalog/sources/openrouter.ts, as do PiDrei's (scripts/catalog/sources/).
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from catalog.sources.openrouter import build_openrouter_catalog
from catalog.sources.vercel_ai_gateway import get_ai_gateway_cost


def open_router_chat_cost(pricing):
    catalog = build_openrouter_catalog(
        [
            {
                "id": "anthropic/claude-haiku-5.5",
                "name": "Claude Haiku 5.5",
                "supported_parameters": ["tools"],
                "pricing": pricing,
            }
        ],
        [],
        [],
    )
    return catalog["chat"][0]["cost"] if catalog["chat"] else None


# -- OpenRouter pricing overrides ------------------------------------------------


def test_turns_prompt_length_overrides_into_tiers():
    assert open_router_chat_cost(
        {
            "prompt": "0.0000001",
            "completion": "0.0000005",
            "input_cache_read": "0.00000001",
            "input_cache_write": "0.000000125",
            "overrides": [
                {
                    "min_prompt_tokens": 100000,
                    "prompt": "0.0000005",
                    "completion": "0.0000025",
                    "input_cache_read": "0.00000005",
                    "input_cache_write": "0.000000625",
                }
            ],
        }
    ) == {
        "input": 0.1,
        "output": 0.5,
        "cacheRead": 0.01,
        "cacheWrite": 0.125,
        "tiers": [{"inputTokensAbove": 100000, "input": 0.5, "output": 2.5, "cacheRead": 0.05, "cacheWrite": 0.625}],
    }


def test_keeps_base_rates_missing_from_an_override():
    cost = open_router_chat_cost(
        {
            "prompt": "0.000002",
            "completion": "0.000012",
            "input_cache_read": "0.0000002",
            "input_cache_write": "0.000000375",
            "overrides": [{"min_prompt_tokens": 200000, "prompt": "0.000004", "completion": "0.000018"}],
        }
    )
    assert cost["tiers"] == [
        {"inputTokensAbove": 200000, "input": 4, "output": 18, "cacheRead": 0.2, "cacheWrite": 0.375}
    ]


def test_skips_time_of_day_overrides():
    assert open_router_chat_cost(
        {
            "prompt": "0.000000132",
            "completion": "0.000000528",
            "overrides": [
                {"utc_start": 0, "utc_end": 1600, "prompt": "0.000000132", "completion": "0.000000528"},
                {"utc_days": ["saturday"], "min_prompt_tokens": 1000, "prompt": "0.0000001"},
            ],
        }
    ) == {"input": 0.132, "output": 0.528, "cacheRead": 0, "cacheWrite": 0}


# -- Vercel AI Gateway pricing tiers ---------------------------------------------


def test_turns_bracket_starts_into_tiers_with_the_rates_in_effect_there():
    assert get_ai_gateway_cost(
        {
            "input": "0.000001",
            "output": "0.000005",
            "input_cache_read": "0.0000002",
            "input_tiers": [
                {"cost": "0.000001", "min": 0, "max": 32001},
                {"cost": "0.0000018", "min": 32001, "max": 128001},
                {"cost": "0.000003", "min": 128001},
            ],
            "output_tiers": [
                {"cost": "0.000005", "min": 0, "max": 32001},
                {"cost": "0.000009", "min": 32001, "max": 128001},
                {"cost": "0.000015", "min": 128001},
            ],
        }
    ) == {
        "input": 1,
        "output": 5,
        "cacheRead": 0.2,
        "cacheWrite": 0,
        "tiers": [
            {"inputTokensAbove": 32000, "input": 1.8, "output": 9, "cacheRead": 0.2, "cacheWrite": 0},
            {"inputTokensAbove": 128000, "input": 3, "output": 15, "cacheRead": 0.2, "cacheWrite": 0},
        ],
    }


def test_returns_base_rates_without_tiers():
    assert get_ai_gateway_cost({"input": "0.000003", "output": 0.000015}) == {
        "input": 3,
        "output": 15,
        "cacheRead": 0,
        "cacheWrite": 0,
    }
