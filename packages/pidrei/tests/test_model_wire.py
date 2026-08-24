"""Wire round-trip for compat fields whose value is a nested object.

pi's compat objects are plain JS values, so `JSON.stringify` and `mergeCompat`
handle them without help; pidrei parses them into dataclasses, which have to be
converted back on the way out. `allowedFallbackModels` is the only such field
today — it reached the vendored catalog with the 0.84.3 regen and RPC's
`get_available_models` could not serialize it.
"""

import json

from pidrei.core.model_wire import compat_to_dict, merge_compat, model_to_dict, parse_compat
from pidrei_ai.types import AnthropicAllowedFallbackModel, AnthropicMessagesCompat, Model, ModelCost


FALLBACK_WIRE = {
    "provider": "anthropic",
    "model": "claude-opus-4-8",
    "cost": {"input": 5, "output": 25, "cacheRead": 0.5, "cacheWrite": 6.25},
}


def _model(compat: AnthropicMessagesCompat) -> Model:
    return Model(
        id="claude-fable-5",
        name="Fable 5",
        api="anthropic-messages",
        provider="anthropic",
        base_url="https://api.anthropic.com/v1",
        reasoning=True,
        input=["text"],
        cost=ModelCost(input=1, output=5, cache_read=0.1, cache_write=1.25),
        context_window=200000,
        max_tokens=64000,
        compat=compat,
    )


def test_parses_fallback_targets_into_dataclasses():
    compat = parse_compat("anthropic-messages", {"allowedFallbackModels": [FALLBACK_WIRE]})

    assert compat is not None
    (target,) = compat.allowed_fallback_models
    assert isinstance(target, AnthropicAllowedFallbackModel)
    assert target.model == "claude-opus-4-8"
    assert target.cost.cache_write == 6.25


def test_serializes_fallback_targets_back_to_json():
    compat = parse_compat("anthropic-messages", {"allowedFallbackModels": [FALLBACK_WIRE]})

    assert compat_to_dict(compat) == {"allowedFallbackModels": [FALLBACK_WIRE]}
    # The RPC path is a bare json.dumps over this, so it has to be JSON-native.
    assert json.loads(json.dumps(model_to_dict(_model(compat))))["compat"] == {"allowedFallbackModels": [FALLBACK_WIRE]}


def test_keeps_fallback_targets_typed_across_a_compat_merge():
    base = parse_compat("anthropic-messages", {"allowedFallbackModels": [FALLBACK_WIRE]})

    merged = merge_compat("anthropic-messages", base, {"supportsStrictTools": True})

    assert merged is not None
    assert merged.supports_strict_tools is True
    (target,) = merged.allowed_fallback_models
    assert isinstance(target, AnthropicAllowedFallbackModel)
    assert target.provider == "anthropic"
