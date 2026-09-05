"""pidrei-only: `finish_body` on the cancel path.

pi hands its AbortSignal to fetch and the signal tears the body down; here the
adapter's `finally` is the close path, and a `finally` unwinding from a scope
cancel cannot await anything (tonio serves no suspension of a cancelled chain).
The close must therefore leave on its own task and still happen.
"""

import pytest
import tonio.colored as tonio
from tonio.exceptions import CancelledError

from pidrei_ai.utils import http
from pidrei_ai.utils.cancel import CancelToken


class _Response:
    def __init__(self) -> None:
        self.closed = False
        self.closed_event = tonio.Event()

    async def close(self) -> None:
        self.closed = True
        self.closed_event.set()


async def _body():
    yield b"data: {}\n\n"


@pytest.mark.tonio
async def test_closes_from_its_own_task_when_the_token_has_fired():
    response = _Response()
    cancel = CancelToken()
    cancel.cancel()

    await http.finish_body(_body(), response, drain=False, cancel=cancel)

    await response.closed_event.wait(1)
    assert response.closed


@pytest.mark.tonio
async def test_closes_from_its_own_task_when_unwinding_a_cancelled_error():
    response = _Response()

    async def unwind() -> None:
        try:
            raise CancelledError()
        finally:
            await http.finish_body(_body(), response, drain=False)

    with pytest.raises(CancelledError):
        await unwind()

    await response.closed_event.wait(1)
    assert response.closed


@pytest.mark.tonio
async def test_closes_inline_when_nothing_is_cancelled():
    response = _Response()

    await http.finish_body(_body(), response, drain=False, cancel=CancelToken())

    assert response.closed
