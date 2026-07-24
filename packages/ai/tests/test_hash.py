from pppi_ai.utils.hash import short_hash


def test_deterministic():
    assert short_hash("hello") == short_hash("hello")


def test_distinct_inputs_distinct_outputs():
    values = {short_hash(text) for text in ("", "a", "b", "hello", "hello!", "https://api.anthropic.com")}
    assert len(values) == 6


def test_base36_format():
    digest = short_hash("https://api.anthropic.com/v1/messages")
    assert digest
    assert all(char in "0123456789abcdefghijklmnopqrstuvwxyz" for char in digest)


def test_handles_astral_and_non_ascii_input():
    # Iterates UTF-16 code units like JS charCodeAt: astral chars hash as two units.
    assert short_hash("emoji 🙈 test") != short_hash("emoji ?? test")
    assert short_hash("äußerst") != short_hash("ausserst")
