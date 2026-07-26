"""Mirror of pi's image-model-data.test.ts.

Covers `parse_openrouter_image_models` from the catalog generator, the same
function pi's spec imports from `scripts/generate-image-models.ts`.
"""

import sys
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_image_models import parse_openrouter_image_models


VALID_IMAGE_MODEL = {
    "id": "example/image-model",
    "name": "Example Image Model",
    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image"]},
    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
}


@pytest.mark.parametrize("payload", [{}, {"data": []}, {"data": "invalid"}])
def test_rejects_a_missing_or_empty_strict_catalog(payload):
    with pytest.raises(ValueError, match="missing or empty image model list"):
        parse_openrouter_image_models(payload, True)


def test_rejects_a_strict_catalog_with_no_usable_image_models():
    payload = {
        "data": [
            {
                **VALID_IMAGE_MODEL,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            }
        ]
    }

    with pytest.raises(ValueError, match="no usable image models"):
        parse_openrouter_image_models(payload, True)


def test_parses_a_non_empty_image_model_catalog():
    models = parse_openrouter_image_models({"data": [VALID_IMAGE_MODEL]}, True)

    assert len(models) == 1
    assert models[0]["id"] == "example/image-model"
    assert models[0]["input"] == ["text", "image"]
    assert models[0]["output"] == ["image"]


def test_a_non_strict_empty_catalog_returns_no_models_instead_of_raising():
    assert parse_openrouter_image_models({}, False) == []


def test_pricing_is_scaled_to_dollars_per_million_tokens():
    models = parse_openrouter_image_models({"data": [VALID_IMAGE_MODEL]}, True)

    assert models[0]["cost"]["input"] == pytest.approx(1.0)
    assert models[0]["cost"]["output"] == pytest.approx(2.0)


def test_a_model_without_input_modalities_defaults_to_text():
    payload = {"data": [{**VALID_IMAGE_MODEL, "architecture": {"output_modalities": ["image"]}}]}

    assert parse_openrouter_image_models(payload, True)[0]["input"] == ["text"]
