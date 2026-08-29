"""`oauth` in models.json is no longer a recognised key.

pidrei-only. pi lets a models.json provider declare `oauth: "radius"` to swap in
its radius builtin, and had three code paths honouring it: a schema `const`, a
"baseUrl is required when oauth is set" guard, and a base-url branch that kept
the model's own URL instead of the config's. radius was a documented drop, so
all three were unreachable except through a hand-written key — where the last
one silently changed base-url resolution for a provider with no radius auth.

Phase 6 removed them. These cases pin what replaced them, because the schema
does not set `additionalProperties: false` — so an unknown key is *ignored*
rather than rejected, which keeps a pi-authored models.json loadable here.
"""

import json
import os

import pytest

from pidrei.core.model_config import ModelConfig
from pidrei.core.provider_composer import apply_models_json
from tests.model_runtime_helpers import make_model


@pytest.fixture
def models_json(tmp_path):
    """Standard `tmp_path` (the pre-tonio-0.9.14 yield-fixture ban is gone)."""

    def write(config: dict) -> str:
        path = os.path.join(str(tmp_path), "models.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)
        return path

    return write


@pytest.mark.tonio
async def test_oauth_is_accepted_but_carries_no_meaning(models_json):
    """Ignored, not rejected: forward compatibility with pi's own configs."""
    path = models_json({"providers": {"custom": {"oauth": "radius", "baseUrl": "https://gw.test/v1"}}})

    config = await ModelConfig.load(path)

    assert config.get_error() is None
    assert config.get_provider("custom") == {"oauth": "radius", "baseUrl": "https://gw.test/v1"}


@pytest.mark.tonio
async def test_an_unknown_oauth_value_is_equally_meaningless(models_json):
    """pi's schema pinned the one legal value; there is nothing to pin now."""
    path = models_json({"providers": {"custom": {"oauth": "something-else", "baseUrl": "https://gw.test/v1"}}})

    config = await ModelConfig.load(path)

    assert config.get_error() is None


def test_base_url_comes_from_the_config_even_when_oauth_is_present():
    """The removed branch kept `model.base_url` here, ignoring the config."""
    base = make_model("custom", "m")

    models = apply_models_json(
        "custom",
        [base],
        {"oauth": "radius", "baseUrl": "https://gw.test/v1"},
    )

    assert models[0].base_url == "https://gw.test/v1"


def test_oauth_alone_no_longer_satisfies_the_must_specify_check():
    """pi raised a bespoke "baseUrl is required when oauth is set"; a config
    carrying only a meaningless key now falls to the general error instead."""
    with pytest.raises(Exception, match='must specify "baseUrl"'):
        apply_models_json("custom", [make_model("custom", "m")], {"oauth": "radius"})
