"""Mirror of pi's google-thinking-signature.test.ts."""

from pidrei_ai.api.google_shared import is_thinking_part, retain_thought_signature


def test_treats_part_thought_true_as_thinking():
    assert is_thinking_part({"thought": True}) is True
    assert is_thinking_part({"thought": True, "thoughtSignature": "opaque-signature"}) is True


def test_does_not_treat_thought_signature_alone_as_thinking():
    # Per Google docs, thoughtSignature is for context replay and can appear on any part type.
    # Only thought === true indicates thinking content.
    # See: https://ai.google.dev/gemini-api/docs/thought-signatures
    assert is_thinking_part({"thoughtSignature": "opaque-signature"}) is False
    assert is_thinking_part({"thought": False, "thoughtSignature": "opaque-signature"}) is False


def test_does_not_treat_empty_or_missing_signatures_as_thinking_if_thought_is_not_set():
    assert is_thinking_part({}) is False
    assert is_thinking_part({"thought": False, "thoughtSignature": ""}) is False


def test_preserves_the_existing_signature_when_subsequent_deltas_omit_thought_signature():
    first = retain_thought_signature(None, "sig-1")
    assert first == "sig-1"

    second = retain_thought_signature(first, None)
    assert second == "sig-1"

    third = retain_thought_signature(second, "")
    assert third == "sig-1"


def test_updates_the_signature_when_a_new_non_empty_signature_arrives():
    assert retain_thought_signature("sig-1", "sig-2") == "sig-2"
