"""Mirror of pi's input-transform-streaming-example.test.ts.

pi builds a fake `ExtensionAPI` object literal; the same shape here is a
`SimpleNamespace` with `on` and `exec`. pidrei's `pi.exec()` returns the
coroutine rather than awaiting, so the stub does too.
"""

from types import SimpleNamespace

import pytest

from pidrei.core.exec import ExecResult

from .example_extensions import load_example


DIFF_OUTPUT = " src/index.py | 5 ++---\n 1 file changed, 2 insertions(+), 3 deletions(-)"
GIT_SUCCESS = ExecResult(stdout=DIFF_OUTPUT, stderr="", code=0, killed=False)
GIT_EMPTY = ExecResult(stdout="", stderr="", code=0, killed=False)
GIT_FAIL = ExecResult(stdout="", stderr="not a git repo", code=128, killed=False)


async def setup(exec_result: ExecResult):
    handler: dict = {}
    calls: list[tuple] = []

    async def fake_exec(command, args, **kwargs):
        calls.append((command, args))
        return exec_result

    def on(event, callback):
        if event == "input":
            handler["input"] = callback

    api = SimpleNamespace(on=on, exec=fake_exec)
    load_example("input_transform_streaming").extension(api)

    async def emit(text: str, streaming_behavior: str | None = None):
        event = {
            "type": "input",
            "text": text,
            "source": "interactive",
            "streamingBehavior": streaming_behavior,
        }
        return await handler["input"](event, SimpleNamespace())

    return emit, calls


@pytest.mark.tonio
async def test_skips_exec_during_steering():
    emit, calls = await setup(GIT_SUCCESS)

    result = await emit("what changes did I make?", "steer")

    assert result == {"action": "continue"}
    assert calls == []


@pytest.mark.tonio
async def test_transforms_when_idle_and_the_text_matches_the_trigger():
    emit, calls = await setup(GIT_SUCCESS)

    result = await emit("review my changes")

    assert calls == [("git", ["diff", "--stat"])]
    assert result["action"] == "transform"
    assert "review my changes" in result["text"]
    assert "src/index.py" in result["text"]


@pytest.mark.tonio
async def test_transforms_when_queued_as_a_follow_up():
    emit, calls = await setup(GIT_SUCCESS)

    result = await emit("show me the diff", "followUp")

    assert calls != []
    assert result["action"] == "transform"


@pytest.mark.tonio
async def test_continues_when_the_text_does_not_match_the_trigger():
    emit, calls = await setup(GIT_SUCCESS)

    result = await emit("explain this function")

    assert result == {"action": "continue"}
    assert calls == []


@pytest.mark.tonio
async def test_continues_when_git_diff_is_empty():
    emit, _calls = await setup(GIT_EMPTY)

    assert await emit("any changes?") == {"action": "continue"}


@pytest.mark.tonio
async def test_continues_when_git_fails():
    emit, _calls = await setup(GIT_FAIL)

    assert await emit("show modified files") == {"action": "continue"}
