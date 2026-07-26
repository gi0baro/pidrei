from pidrei_ai.utils.sanitize_unicode import sanitize_surrogates


def test_preserves_valid_emoji_and_text():
    text = "Hello 🙈 World äußerst こんにちは ∑∫∂√"
    assert sanitize_surrogates(text) == text


def test_removes_unpaired_high_surrogate():
    assert sanitize_surrogates("Text \ud83d here") == "Text  here"


def test_removes_unpaired_low_surrogate():
    assert sanitize_surrogates("Text \ude48 here") == "Text  here"


def test_removes_surrogates_from_lone_json_escapes():
    # json.loads of a lone escape produces a surrogate code point in the str.
    import json

    decoded = json.loads('"broken \\ud83d emoji"')
    assert sanitize_surrogates(decoded) == "broken  emoji"
