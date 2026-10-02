"""pidrei-only: `lazy_oauth` loads its flow once for concurrent first calls.

pi memoizes the load's promise on its one thread; here two first calls can run
in parallel, and the load is serialized instead.
"""

import threading

import pytest
import tonio.colored as tonio

from pidrei_ai.auth.helpers import lazy_oauth
from pidrei_ai.auth.types import ModelAuth, OAuthAuth, OAuthCredential


CREDENTIAL = OAuthCredential(refresh="refresh", access="access", expires=0)


@pytest.mark.tonio
async def test_concurrent_first_calls_share_one_load():
    async def login(_interaction, _options=None):
        raise AssertionError("not called")

    async def refresh(_credential, _cancel):
        raise AssertionError("not called")

    async def to_auth(credential):
        return ModelAuth(api_key=credential.access)

    flow = OAuthAuth(name="Flow", login=login, refresh=refresh, to_auth=to_auth)
    loads = 0
    loads_guard = threading.Lock()
    first_load_started = tonio.Event()
    second_load_started = tonio.Event()

    async def load():
        nonlocal loads
        with loads_guard:
            loads += 1
            index = loads
        if index == 1:
            first_load_started.set()
            # The window for the second call to start a load of its own; with
            # the load serialized it waits for this one instead, and the
            # window runs out.
            await second_load_started.wait(0.2)
        else:
            second_load_started.set()
        return flow

    lazy = lazy_oauth(name="Lazy", load=load)

    first = tonio.spawn(lazy.to_auth(CREDENTIAL))
    await first_load_started.wait(5)
    assert first_load_started.is_set()
    second = tonio.spawn(lazy.to_auth(CREDENTIAL))

    assert await first == ModelAuth(api_key="access")
    assert await second == ModelAuth(api_key="access")
    assert loads == 1
