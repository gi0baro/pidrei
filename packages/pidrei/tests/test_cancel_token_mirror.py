"""The tui CancelToken mirror must stay usable as an ai-layer cancel token.

Regression for the /login crash: `LoginDialogComponent.signal` reaches
`Models.login`, whose first act is `cancel.raise_if_cancelled()` — missing on
the tui mirror at the time ("'CancelToken' object has no attribute
'raise_if_cancelled'"). Three guards:

- surface parity: every public member of the ai token exists on the mirror
  with the same parameters, so contract drift fails here before it fails in
  a login or streaming flow;
- behavior through a real consumer: `run_cancellable` treats a tui token
  exactly like an ai token, including `on_cancel` callbacks receiving the
  abort reason;
- the login dialog cancels with a reason whose message is "Login cancelled",
  which the interactive mode suppresses instead of showing a login failure.
"""

import inspect

import pytest
import tonio.colored as tonio

from pidrei.modes.interactive.components.login_dialog import LoginDialogComponent
from pidrei.modes.interactive.theme import init_theme
from pidrei_ai.utils.abort import run_cancellable
from pidrei_ai.utils.cancel import CancelToken as AiCancelToken
from pidrei_tui.components.cancellable_loader import CancelToken as TuiCancelToken


class _FakeTui:
    def request_render(self) -> None:
        pass


def _parameters(fn) -> list[tuple]:
    return [(p.name, p.kind, p.default) for p in inspect.signature(fn).parameters.values()]


def test_mirror_surface_matches_ai_token():
    for name, member in inspect.getmembers(AiCancelToken):
        if name.startswith("_"):
            continue
        assert hasattr(TuiCancelToken, name), f"tui CancelToken mirror is missing `{name}`"
        if callable(member):
            mirror = getattr(TuiCancelToken, name)
            assert _parameters(mirror) == _parameters(member), f"`{name}` parameters drifted from the ai token"


def test_mirror_raise_if_cancelled_raises_the_reason():
    token = TuiCancelToken()
    token.raise_if_cancelled()  # not cancelled: no-op

    token.cancel(ValueError("stop"))
    with pytest.raises(ValueError, match="stop"):
        token.raise_if_cancelled()


@pytest.mark.tonio
async def test_run_cancellable_unwinds_on_tui_token():
    token = TuiCancelToken()
    started = tonio.Event()

    async def parked() -> None:
        started.set()
        await tonio.Event().wait()

    async def cancel_once_started() -> None:
        await started.wait()
        token.cancel(RuntimeError("stop"))

    async def run() -> None:
        with pytest.raises(RuntimeError, match="stop"):
            await run_cancellable(parked(), token)

    await tonio.spawn(run(), cancel_once_started())


@pytest.mark.tonio
async def test_login_dialog_escape_cancels_with_login_cancelled_reason():
    await init_theme("dark")
    dialog = LoginDialogComponent(_FakeTui(), "anthropic", lambda *_args: None, "Anthropic")

    dialog._cancel()

    assert dialog.signal.cancelled
    with pytest.raises(Exception, match="Login cancelled"):
        dialog.signal.raise_if_cancelled()
