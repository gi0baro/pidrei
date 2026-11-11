"""Mirror of pi's suite/regressions/9154-prompt-during-tree-navigation.test.ts.

JS deferreds become `tonio.Event`s, pending prompts and navigations spawned
coroutines, `Promise.race` a `tonio.Waiter.any` over Events the racers set, and
`vi.fn` spies plain recording lists. Handlers are async here: where pi's
session_tree handler starts a navigation whose synchronous prefix claims the
navigation before the handler returns, the handler here waits until the new
navigation has reached session_before_tree, which is past that claim.
"""

import pytest
import tonio.colored as tonio

from pidrei.core.agent_session import ExtensionBindings, PromptOptions
from pidrei_ai.providers.faux import faux_assistant_message

from .coding_session_helpers import assistant_msg, user_msg
from .harness import create_harness, get_message_text, get_user_texts


_NAVIGATION_BLOCKED = "Wait for the current response to finish before navigating the session tree."
# Waiter.any takes microseconds.
_FIVE_SECONDS_US = 5_000_000


@pytest.fixture
def harnesses(request):
    created: list = []
    request.addfinalizer(lambda: [harness.cleanup() for harness in created])
    return created


async def _seed_history(harness) -> str:
    await harness.session_manager.append_message(user_msg("first prompt"))
    target_id = await harness.session_manager.append_message(assistant_msg("first response"))
    await harness.session_manager.append_message(user_msg("second prompt"))
    await harness.session_manager.append_message(assistant_msg("second response"))
    harness.session.refresh_context()
    return target_id


class _Settled:
    """A spawned call's outcome: pi's `.then(() => undefined, (error) => error)`."""

    def __init__(self, coro) -> None:
        self.done = tonio.Event()
        self.error: Exception | None = None
        self.handle = tonio.spawn(self._run(coro))

    async def _run(self, coro) -> None:
        try:
            await coro
        except Exception as error:
            self.error = error
        finally:
            self.done.set()


@pytest.mark.tonio
async def test_rejects_a_prompt_while_branch_summarization_is_in_progress(harnesses):
    summary_started = tonio.Event()
    summary_released = tonio.Event()

    async def factory(pi) -> None:
        async def on_before_tree(_event, _ctx):
            summary_started.set()
            await summary_released.wait()
            return {"summary": {"summary": "abandoned branch summary"}}

        pi.on("session_before_tree", on_before_tree)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    target_id = await _seed_history(harness)
    harness.set_responses([faux_assistant_message("unexpected response")])

    navigation = tonio.spawn(harness.session.navigate_tree(target_id, {"summarize": True}))
    await summary_started.wait(5)
    assert summary_started.is_set(), "session_before_tree never ran"

    preflight_results: list[str] = []
    prompt_error: Exception | None = None
    try:
        await harness.session.prompt(
            "prompt during summary", PromptOptions(source="rpc", preflight_result=preflight_results.append)
        )
    except Exception as error:
        prompt_error = error
    finally:
        summary_released.set()
    await navigation

    persisted_user_texts = [
        get_message_text(entry["message"])
        for entry in harness.session_manager.get_entries()
        if entry["type"] == "message" and entry["message"].role == "user"
    ]
    assert preflight_results == []
    assert prompt_error is not None and "compaction is in progress" in str(prompt_error)
    assert "prompt during summary" not in get_user_texts(harness)
    assert "prompt during summary" not in persisted_user_texts
    assert harness.events_of_type("agent_start") == []
    assert harness.events_of_type("agent_settled") == []


# The inverse order also loses the completed prompt when navigation rebuilds the context.
@pytest.mark.tonio
@pytest.mark.parametrize("hook", ["input", "before_agent_start"])
async def test_rejects_navigation_during_an_awaited_hook(harnesses, hook):
    prompt_started = tonio.Event()
    prompt_released = tonio.Event()
    tree_started = tonio.Event()
    tree_released = tonio.Event()

    async def factory(pi) -> None:
        async def pause(_event, _ctx):
            prompt_started.set()
            await prompt_released.wait()

        async def on_before_tree(_event, _ctx):
            tree_started.set()
            await tree_released.wait()

        pi.on(hook, pause)
        pi.on("session_before_tree", on_before_tree)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    target_id = await _seed_history(harness)
    harness.set_responses([faux_assistant_message("response to racing prompt")])

    prompt = tonio.spawn(harness.session.prompt("racing prompt", PromptOptions(source="rpc")))
    await prompt_started.wait(5)
    assert prompt_started.is_set(), f"the {hook} hook never ran"
    navigation = _Settled(harness.session.navigate_tree(target_id))
    await tonio.Waiter.any(tree_started, navigation.done, timeout=_FIVE_SECONDS_US)
    try:
        prompt_released.set()
        await prompt
    finally:
        prompt_released.set()
        tree_released.set()
    await navigation.done.wait(5)
    assert navigation.done.is_set()

    assert navigation.error is not None and str(navigation.error) == _NAVIGATION_BLOCKED
    assert "racing prompt" in get_user_texts(harness)
    assert "response to racing prompt" in [get_message_text(message) for message in harness.session.messages]
    assert (await harness.session.navigate_tree(target_id)).cancelled is False


@pytest.mark.tonio
@pytest.mark.parametrize("outcome", ["handled", "rejected"])
async def test_allows_navigation_after_a_prompt_is_settled_during_preflight(harnesses, outcome):
    async def factory(pi) -> None:
        if outcome == "handled":

            async def on_input(_event, _ctx):
                return {"action": "handled"}

            pi.on("input", on_input)

    harness = await create_harness(with_configured_auth=outcome != "rejected", extension_factories=[factory])
    harnesses.append(harness)
    target_id = await _seed_history(harness)

    if outcome == "handled":
        await harness.session.prompt("not sent to the model")
    else:
        with pytest.raises(Exception, match="No API key"):
            await harness.session.prompt("not sent to the model")
    assert (await harness.session.navigate_tree(target_id)).cancelled is False


@pytest.mark.tonio
async def test_keeps_navigation_blocked_when_another_prompt_finishes_preflight_first(harnesses):
    started = tonio.Event()
    released = tonio.Event()

    async def factory(pi) -> None:
        async def on_input(event, _ctx):
            if event["text"] == "waiting prompt":
                started.set()
                await released.wait()
            return {"action": "handled"}

        pi.on("input", on_input)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    target_id = await _seed_history(harness)

    waiting = tonio.spawn(harness.session.prompt("waiting prompt"))
    await started.wait(5)
    assert started.is_set(), "the input hook never ran"
    try:
        await harness.session.prompt("handled immediately")
        with pytest.raises(RuntimeError, match=_NAVIGATION_BLOCKED):
            await harness.session.navigate_tree(target_id)
    finally:
        released.set()
    await waiting
    assert (await harness.session.navigate_tree(target_id)).cancelled is False


@pytest.mark.tonio
async def test_allows_a_prompt_from_session_tree_after_the_selected_context_is_installed(harnesses):
    errors: list[str] = []
    sessions: list = []

    async def factory(pi) -> None:
        async def on_tree(_event, _ctx):
            await sessions[0].send_user_message("continue on the selected branch")

        pi.on("session_tree", on_tree)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    sessions.append(harness.session)
    await harness.session.bind_extensions(ExtensionBindings(on_error=lambda error: errors.append(error.error)))
    target_id = await _seed_history(harness)
    harness.set_responses([faux_assistant_message("continued")])

    await harness.session.navigate_tree(target_id)

    assert errors == []
    assert get_user_texts(harness) == ["first prompt", "continue on the selected branch"]
    assert harness.session.get_last_assistant_text() == "continued"


@pytest.mark.tonio
async def test_does_not_clear_a_new_navigation_started_from_session_tree(harnesses):
    started = tonio.Event()
    released = tonio.Event()
    next_target: list[str] = []
    continuation: list[_Settled] = []
    sessions: list = []

    async def factory(pi) -> None:
        async def on_before_tree(event, _ctx):
            if next_target and event["preparation"]["targetId"] == next_target[0]:
                started.set()
                await released.wait()

        async def on_tree(_event, _ctx):
            if not continuation:
                continuation.append(_Settled(sessions[0].navigate_tree(next_target[0])))
                # pi's navigation claims synchronously, before this handler returns.
                await tonio.Waiter.any(started, continuation[0].done, timeout=_FIVE_SECONDS_US)

        pi.on("session_before_tree", on_before_tree)
        pi.on("session_tree", on_tree)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    sessions.append(harness.session)
    target_id = await _seed_history(harness)
    next_target.append(harness.session_manager.get_leaf_id())
    try:
        await harness.session.navigate_tree(target_id)
        assert continuation, "session_tree never started the next navigation"
        await tonio.Waiter.any(started, continuation[0].done, timeout=_FIVE_SECONDS_US)
        assert harness.session.is_compacting is True
        with pytest.raises(Exception, match="compaction is in progress"):
            await harness.session.prompt("must wait for the second navigation")
    finally:
        released.set()
        if continuation:
            await continuation[0].done.wait(5)
    assert continuation[0].done.is_set()
    assert continuation[0].error is None
    assert harness.session_manager.get_leaf_id() == next_target[0]
    assert harness.session.is_compacting is False


@pytest.mark.tonio
@pytest.mark.parametrize("event", ["command", "agent_settled"])
async def test_allows_navigation_from_an_extension(harnesses, event):
    target: list[str] = []
    navigated: list[bool] = []
    sessions: list = []

    async def factory(pi) -> None:
        async def navigate(*_args):
            await sessions[0].navigate_tree(target[0])
            navigated.append(True)

        if event == "command":
            pi.register_command("rewind", handler=navigate)
        else:
            pi.on("agent_settled", navigate)

    harness = await create_harness(extension_factories=[factory])
    harnesses.append(harness)
    sessions.append(harness.session)
    target.append(await _seed_history(harness))
    harness.set_responses([faux_assistant_message("done")])

    await harness.session.prompt("/rewind" if event == "command" else "finish then rewind")

    assert navigated == [True]
    assert harness.session_manager.get_leaf_id() == target[0]
