"""Mirror of pi's regressions/5433-extension-oauth-prompt-input.test.ts.

The login dialog an extension's OAuth flow drives must keep every earlier
prompt's rendered value stable once a later prompt is active — a second prompt
must not re-render the first one's answer, or echo its own twice.

pi mocks `openBrowser`; nothing here reaches `show_auth`'s browser path, so
there is nothing to mock.
"""

import pytest

from pidrei.modes.interactive.components.login_dialog import LoginDialogComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei.utils.ansi import strip_ansi


class _FakeTui:
    def request_render(self) -> None:
        pass


@pytest.mark.tonio
async def create_dialog() -> LoginDialogComponent:
    await init_theme("dark")
    return LoginDialogComponent(_FakeTui(), "prompt-repro", lambda *_args: None, "Prompt Repro")


def render_dialog(dialog: LoginDialogComponent) -> list[str]:
    return [line.rstrip() for line in strip_ansi("\n".join(dialog.render(120))).split("\n")]


def count_rendered_value(lines: list[str], value: str) -> int:
    return sum(1 for line in lines if line.strip() == f"> {value}")


@pytest.mark.tonio
async def test_keeps_previous_prompt_input_stable_when_a_later_prompt_is_active():
    dialog = await create_dialog()

    first_prompt = dialog.show_prompt("First prompt:", "first-value")
    await dialog.handle_input("first-value")
    await dialog.handle_input("\n")
    assert await first_prompt == "first-value"

    second_prompt = dialog.show_prompt("Second prompt:")
    await dialog.handle_input("second-secret-demo")

    lines = render_dialog(dialog)
    output = "\n".join(lines)
    assert "First prompt:" in output
    assert "Second prompt:" in output
    assert count_rendered_value(lines, "first-value") == 1
    assert count_rendered_value(lines, "second-secret-demo") == 1

    await dialog.handle_input("\n")
    assert await second_prompt == "second-secret-demo"


@pytest.mark.tonio
async def test_preserves_auth_instructions_when_showing_a_prompt():
    dialog = await create_dialog()

    dialog.show_auth("https://example.invalid/login", "Authorize the extension")
    dialog.show_prompt("First prompt:").close()

    output = "\n".join(render_dialog(dialog))
    assert "https://example.invalid/login" in output
    assert "Authorize the extension" in output
    assert "First prompt:" in output


@pytest.mark.tonio
async def test_preserves_neutral_information_and_links_when_showing_a_prompt():
    dialog = await create_dialog()

    dialog.show_info(
        "Configure credentials outside pidrei.",
        [{"label": "Provider documentation", "url": "https://example.invalid/docs"}],
    )
    dialog.show_prompt("Press Enter to continue:").close()

    output = "\n".join(render_dialog(dialog))
    assert "Configure credentials outside pidrei." in output
    assert "Provider documentation: https://example.invalid/docs" in output
    assert "Press Enter to continue:" in output


@pytest.mark.tonio
async def test_preserves_setup_details_when_showing_a_prompt():
    dialog = await create_dialog()

    dialog.show_details(["AWS credential setup:", "providers.md"])
    dialog.show_prompt("Enter API key:").close()

    output = "\n".join(render_dialog(dialog))
    assert "AWS credential setup:" in output
    assert "providers.md" in output
    assert "Enter API key:" in output


@pytest.mark.tonio
async def test_keeps_previous_manual_input_stable_when_a_later_prompt_is_active():
    dialog = await create_dialog()

    manual_input = dialog.show_manual_input("Paste callback URL:")
    await dialog.handle_input("callback-value")
    await dialog.handle_input("\n")
    assert await manual_input == "callback-value"

    prompt = dialog.show_prompt("Second prompt:")
    await dialog.handle_input("second-secret-demo")

    lines = render_dialog(dialog)
    output = "\n".join(lines)
    assert "Paste callback URL:" in output
    assert "Second prompt:" in output
    assert count_rendered_value(lines, "callback-value") == 1
    assert count_rendered_value(lines, "second-secret-demo") == 1

    await dialog.handle_input("\n")
    assert await prompt == "second-secret-demo"
