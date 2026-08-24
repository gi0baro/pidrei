"""Mirror of pi coding-agent test/suite/regressions/7829-invalid-settings-warning.test.ts.

pi builds a fake `this` for `InteractiveMode.prototype.run` and drives it with
`vi.waitFor`; here the same partial context is a `SimpleNamespace`, and the fake
`_get_user_input` ends the run by raising — the interactive loop is reached only
after the startup diagnostics have rendered. pi's fake carries a real harness
session because its `run` reaches `session.prompt`; that never happens here
either, so a stub session with the one attribute `run` reads is enough.
"""

import os
from types import SimpleNamespace

import pytest

from pidrei.core.agent_session_services import AgentSessionRuntimeDiagnostic
from pidrei.modes.interactive.interactive_mode import InteractiveMode
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_tui import Container


@pytest.mark.tonio
async def test_renders_startup_diagnostics_inside_the_transcript():
    init_theme_sync("dark")
    previous_skip = os.environ.get("PIDREI_SKIP_VERSION_CHECK")
    os.environ["PIDREI_SKIP_VERSION_CHECK"] = "1"

    class _ReachedInput(Exception):
        """Ends `run` at the interactive loop, once the startup output is rendered."""

    async def noop():
        return None

    async def no_updates():
        return []

    async def get_user_input():
        raise _ReachedInput

    chat_container = Container()
    fake = SimpleNamespace(
        init=noop,
        _options={
            "startupDiagnostics": [
                AgentSessionRuntimeDiagnostic(
                    type="warning", message="Invalid settings file /tmp/settings.json: malformed JSON"
                )
            ]
        },
        _chat_container=chat_container,
        _output_pad=1,
        _version="test",
        ui=SimpleNamespace(request_render=lambda force=False: None),
        session=SimpleNamespace(model_runtime=SimpleNamespace(get_error=lambda: None)),
        _check_for_package_updates=no_updates,
        _check_tmux_keyboard_setup=noop,
        _maybe_warn_about_anthropic_subscription_auth=noop,
        _get_user_input=get_user_input,
    )
    fake.show_warning = lambda message: InteractiveMode.show_warning(fake, message)
    fake.show_error = lambda message: InteractiveMode.show_error(fake, message)
    fake.show_status = lambda message: InteractiveMode.show_status(fake, message)

    try:
        with pytest.raises(_ReachedInput):
            await InteractiveMode.run(fake)
        output = strip_ansi("\n".join(chat_container.render(120)))
    finally:
        if previous_skip is None:
            os.environ.pop("PIDREI_SKIP_VERSION_CHECK", None)
        else:
            os.environ["PIDREI_SKIP_VERSION_CHECK"] = previous_skip

    assert "Warning: Invalid settings file /tmp/settings.json: malformed JSON" in output
