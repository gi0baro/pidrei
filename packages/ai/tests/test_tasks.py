import pytest
import tonio.colored as tonio

from pidrei_ai.utils.tasks import gather


async def _value(x):
    await tonio.yield_now()
    return x


async def _fail(message: str):
    await tonio.yield_now()
    raise ValueError(message)


@pytest.mark.tonio
async def test_gather_preserves_order_and_handles_empty():
    assert await gather() == []
    assert await gather(_value(1)) == [1]
    assert await gather(_value(1), _value(2), _value(3)) == [1, 2, 3]


@pytest.mark.tonio
async def test_gather_raises_the_child_error_bare():
    # Callers report str(error) like pi's Promise.all rejection; a
    # SpawnExceptionGroup would surface as "SpawnExceptionGroup (1 sub-exception)".
    with pytest.raises(ValueError, match="^boom$") as info:
        await gather(_value(1), _fail("boom"))
    assert isinstance(info.value.__cause__, ExceptionGroup)
