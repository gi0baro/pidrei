"""Mirror of pi's image-model-data.test.ts.

Covers `build_openrouter_catalog` from the catalog generator, the function pi's
spec imports from `scripts/openrouter-catalog.ts`.
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_models import build_openrouter_catalog


IMAGE_ONLY = {
    "id": "example/image-model",
    "name": "Example Image Model",
    "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["image"]},
    "pricing": {"prompt": "0.000001", "completion": "0.000002"},
}

CHAT_WITH_IMAGES = {
    "id": "example/multimodal",
    "name": "Example Multimodal",
    "supported_parameters": ["tools"],
    "architecture": {
        "modality": "text+image->text+image",
        "input_modalities": ["text", "image"],
        "output_modalities": ["text", "image"],
    },
    "context_length": 32000,
}

CHAT_ONLY = {
    "id": "example/chat",
    "name": "Example Chat",
    "supported_parameters": ["tools"],
    "architecture": {"modality": "text->text", "output_modalities": ["text"]},
}

DECISION_MODEL = {
    "id": "typesafe/jev-1.13",
    "name": "TypeSafe: Jev 1.13",
    "supported_parameters": [],
    "architecture": {"modality": "text->decisions", "input_modalities": ["text"], "output_modalities": ["decisions"]},
    "pricing": {"prompt": "0.000000042", "completion": "0"},
    "context_length": 32000,
    "top_provider": {"context_length": 32000, "max_completion_tokens": 28800},
}


def test_emits_image_only_models_from_the_image_listing_as_image_models():
    catalog = build_openrouter_catalog([], [IMAGE_ONLY], [])
    assert catalog["chat"] == []
    [image] = catalog["images"]
    assert image["type"] == "image"
    assert image["id"] == "example/image-model"
    assert image["api"] == "openrouter-images"
    assert image["input"] == ["text", "image"]
    assert image["output"] == ["image"]
    assert image["cost"]["input"] == 1
    assert image["cost"]["output"] == 2


def test_emits_separate_chat_and_image_entries_for_an_id_that_supports_both_operations():
    catalog = build_openrouter_catalog([CHAT_WITH_IMAGES, CHAT_ONLY], [CHAT_WITH_IMAGES, IMAGE_ONLY], [])
    assert [model["id"] for model in catalog["chat"]] == ["example/multimodal", "example/chat"]
    assert [model["type"] for model in catalog["chat"]] == ["chat", "chat"]
    assert [model["id"] for model in catalog["images"]] == ["example/multimodal", "example/image-model"]
    assert all(model["type"] == "image" for model in catalog["images"])
    assert [model["output"] for model in catalog["images"]] == [["text", "image"], ["image"]]


def test_ignores_listed_models_that_neither_support_tools_nor_emit_images():
    catalog = build_openrouter_catalog(
        [{**CHAT_ONLY, "supported_parameters": []}],
        [{**IMAGE_ONLY, "architecture": {"output_modalities": ["text"]}}],
        [{**DECISION_MODEL, "architecture": {"output_modalities": ["text"]}}],
    )
    assert catalog["chat"] == []
    assert catalog["images"] == []
    assert catalog["classifiers"] == []


def test_emits_decision_models_as_system_one_classifier_models():
    catalog = build_openrouter_catalog([], [], [DECISION_MODEL, DECISION_MODEL])
    assert catalog["chat"] == []
    assert catalog["classifiers"] == [
        {
            "type": "classifier",
            "id": "typesafe/jev-1.13",
            "name": "TypeSafe: Jev 1.13",
            "api": "typesafe-system-one",
            "provider": "openrouter",
            "baseUrl": "https://openrouter.ai/api/v1",
            "input": ["text"],
            "cost": {"input": 0.042, "output": 0, "cacheRead": 0, "cacheWrite": 0},
            "contextWindow": 32000,
        }
    ]
