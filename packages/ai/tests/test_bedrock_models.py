"""Mirror of pi's bedrock-models.test.ts.

pi's file is a live catalog sweep gated on AWS credentials and
`BEDROCK_EXTENSIVE_MODEL_TEST`; only its ungated case is mirrored, as the rest
would need real Bedrock access.
"""

from pidrei_ai.providers.all import get_builtin_models


def test_should_get_all_available_bedrock_models():
    assert len(get_builtin_models("amazon-bedrock")) > 0


def test_exposes_claude_opus_5_through_an_inference_profile_only():
    models = get_builtin_models("amazon-bedrock")
    assert any(model.id == "global.anthropic.claude-opus-5" for model in models)
    assert not any(model.id == "anthropic.claude-opus-5" for model in models)
