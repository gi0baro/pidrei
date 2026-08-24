"""Mirror of pi's suite/regressions/7269-cli-end-of-options.test.ts.

pi's `it.each` becomes `@pytest.mark.parametrize`; the tonio marker only sits
on the case that drives a session, so the pure parser case stays synchronous.
"""

import pytest

from pidrei.cli.args import parse_args
from pidrei_ai.providers.faux import faux_assistant_message

from .harness import create_harness, get_user_texts


@pytest.mark.tonio
@pytest.mark.parametrize(
    "prompt",
    ["- summarize the following points for me", "--answer my question briefly"],
)
async def test_passes_a_dash_prefixed_prompt_after_end_of_options(prompt):
    parsed = parse_args(["-ne", "--no-session", "-p", "--", prompt])
    assert parsed.messages == [prompt]
    assert len(parsed.unknown_flags) == 0
    assert parsed.diagnostics == []

    harness = await create_harness()
    try:
        harness.set_responses([faux_assistant_message("ok")])
        await harness.session.prompt(parsed.messages[0])
        assert get_user_texts(harness) == [prompt]
    finally:
        harness.cleanup()


def test_stops_parsing_options_while_retaining_file_handling():
    parsed = parse_args(["--unknown-flag", "value", "--", "--provider", "openai", "-c", "@prompt.md"])

    assert parsed.unknown_flags["unknown-flag"] == "value"
    assert parsed.provider is None
    assert parsed.continue_ is None
    assert parsed.messages == ["--provider", "openai", "-c"]
    assert parsed.file_args == ["prompt.md"]
    assert parsed.diagnostics == []
