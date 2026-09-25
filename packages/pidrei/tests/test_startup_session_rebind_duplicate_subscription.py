"""Mirror of pi's suite/regressions/startup-session-rebind-duplicate-subscription.test.ts.

pi calls `rebindCurrentSession` on a hand-built context through the class
prototype; here the unbound method runs against a `SimpleNamespace` stub the
same way. The two binds are gated on events so the startup rebind is still
in flight when the replacement session takes over.
"""

from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.modes.interactive.interactive_mode import InteractiveMode


async def _async_noop() -> None:
    pass


@pytest.mark.tonio
async def test_does_not_subscribe_from_the_stale_startup_rebind():
    startup_session = object()
    replacement_session = object()
    startup_bind = tonio.Event()
    replacement_bind = tonio.Event()

    subscribe_calls: list[bool] = []
    title_calls: list[bool] = []
    bind_count = 0
    # Set as each bind parks on its gate, so the test waits for it instead of polling.
    startup_bind_reached = tonio.Event()
    replacement_bind_reached = tonio.Event()

    async def bind_current_session_extensions() -> None:
        nonlocal bind_count
        bind_count += 1
        if bind_count == 1:
            startup_bind_reached.set()
            await startup_bind.wait()
        else:
            replacement_bind_reached.set()
            await replacement_bind.wait()

    context = SimpleNamespace(
        session=startup_session,
        _unsubscribe=None,
        _apply_runtime_settings=_async_noop,
        render_current_session_state=lambda: None,
        _bind_current_session_extensions=bind_current_session_extensions,
        _subscribe_to_agent=lambda: subscribe_calls.append(True),
        _update_available_provider_count=lambda: None,
        _update_editor_border_color=lambda: None,
        _update_terminal_title=lambda: title_calls.append(True),
    )

    startup_rebind = tonio.spawn(InteractiveMode._rebind_current_session(context))
    await startup_bind_reached.wait(5)
    assert bind_count == 1

    context.session = replacement_session
    replacement_rebind = tonio.spawn(InteractiveMode._rebind_current_session(context, {"renderBeforeBind": True}))
    await replacement_bind_reached.wait(5)

    assert bind_count == 2
    assert len(subscribe_calls) == 1

    startup_bind.set()
    await startup_rebind

    assert len(subscribe_calls) == 1
    assert len(title_calls) == 0

    replacement_bind.set()
    await replacement_rebind

    assert len(subscribe_calls) == 1
    assert len(title_calls) == 1
