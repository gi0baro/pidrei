"""pidrei-specific: the login dialog cancels with a reason whose message is
"Login cancelled", which the interactive mode suppresses instead of showing a
login failure.

Regression for the /login crash: `LoginDialogComponent.signal` reaches
`Models.login`, whose first act is `cancel.raise_if_cancelled()`.
"""

import pytest

from pidrei.modes.interactive.components.login_dialog import LoginDialogComponent
from pidrei.modes.interactive.theme import init_theme


class _FakeTui:
    def request_render(self) -> None:
        pass


@pytest.mark.tonio
async def test_login_dialog_escape_cancels_with_login_cancelled_reason():
    await init_theme("dark")
    dialog = LoginDialogComponent(_FakeTui(), "anthropic", lambda *_args: None, "Anthropic")

    dialog._cancel()

    assert dialog.signal.cancelled
    with pytest.raises(Exception, match="Login cancelled"):
        dialog.signal.raise_if_cancelled()
