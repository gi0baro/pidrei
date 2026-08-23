import pytest
import tonio.colored as tonio

from pidrei_ai.utils.cancel import NEVER_CANCELLED, AbortError, CancelToken, combine_cancel_tokens


def test_initial_state():
    token = CancelToken()

    assert not token.cancelled
    assert token.reason is None


def test_cancel_sets_default_reason():
    token = CancelToken()
    token.cancel()

    assert token.cancelled
    assert isinstance(token.reason, AbortError)


def test_first_cancel_wins():
    token = CancelToken()
    first = ValueError("first")
    token.cancel(first)
    token.cancel(ValueError("second"))

    assert token.reason is first


def test_raise_if_cancelled():
    token = CancelToken()
    token.raise_if_cancelled()

    token.cancel(ValueError("stop"))
    with pytest.raises(ValueError, match="stop"):
        token.raise_if_cancelled()


def test_on_cancel_fires_once_with_reason():
    token = CancelToken()
    seen = []
    token.on_cancel(seen.append)
    token.cancel(ValueError("x"))
    token.cancel(ValueError("y"))

    assert len(seen) == 1
    assert str(seen[0]) == "x"


def test_on_cancel_after_cancellation_fires_immediately():
    token = CancelToken()
    token.cancel()
    seen = []
    token.on_cancel(seen.append)

    assert len(seen) == 1


def test_on_cancel_unsubscribe():
    token = CancelToken()
    seen = []
    unsubscribe = token.on_cancel(seen.append)
    unsubscribe()
    token.cancel()

    assert seen == []


def test_combine_empty_and_single():
    assert combine_cancel_tokens().token is None
    assert combine_cancel_tokens(None, None).token is None

    token = CancelToken()
    combined = combine_cancel_tokens(None, token)
    assert combined.token is token


def test_placeholder_is_transparent_to_combine_and_cannot_fire():
    # The optional-token placeholder must behave like "no token": otherwise
    # every combine would allocate a token and subscribe to one that never fires.
    token = CancelToken()
    assert combine_cancel_tokens(NEVER_CANCELLED, token).token is token
    assert combine_cancel_tokens(NEVER_CANCELLED, None).token is None

    with pytest.raises(RuntimeError):
        NEVER_CANCELLED.cancel()
    assert not NEVER_CANCELLED.cancelled


def test_combine_propagates_first_cancellation():
    a, b = CancelToken(), CancelToken()
    combined = combine_cancel_tokens(a, b)
    assert combined.token is not None
    assert not combined.token.cancelled

    reason = ValueError("from a")
    a.cancel(reason)

    assert combined.token.cancelled
    assert combined.token.reason is reason


def test_combine_with_already_cancelled_input():
    a, b = CancelToken(), CancelToken()
    reason = ValueError("pre")
    a.cancel(reason)
    combined = combine_cancel_tokens(a, b)

    assert combined.token is not None
    assert combined.token.cancelled
    assert combined.token.reason is reason


def test_combine_cleanup_detaches():
    a, b = CancelToken(), CancelToken()
    combined = combine_cancel_tokens(a, b)
    combined.cleanup()
    a.cancel()

    assert combined.token is not None
    assert not combined.token.cancelled


@pytest.mark.tonio
async def test_wait_wakes_on_cancel():
    token = CancelToken()

    async def canceller():
        await tonio.yield_now()
        token.cancel()

    handle = tonio.spawn(canceller())
    await token.wait()
    await handle

    assert token.cancelled
