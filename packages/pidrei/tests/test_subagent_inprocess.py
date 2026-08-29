"""pidrei-own tests for the in-process subagent runner.

pi's subagent example spawns a `pi` child process per task, so these tests
have no upstream mirror (recipe `subagent-inprocess`): they pin the in-process
`run_single_agent` — the result-dict shape (`status`/`errorMessage` instead of
pi's `exitCode`/`stderr`), wire-dict message collection, typed usage
accumulation, and the per-delta streaming preview the child process could not
provide.
"""

import os
import threading
from types import SimpleNamespace

import pytest
import tonio.colored as tonio

from pidrei.core.auth_storage import AuthStorage
from pidrei.core.model_runtime import ModelRuntime
from pidrei.examples.extensions import subagent as subagent_module
from pidrei.examples.extensions.subagent import get_final_output, map_with_concurrency_limit, run_single_agent
from pidrei.examples.extensions.subagent.agents import AgentConfig
from pidrei_ai.auth.types import ApiKeyCredential
from pidrei_ai.providers.faux import faux_assistant_message, faux_provider
from pidrei_ai.registry import ModelsRefreshOptions


async def _create_faux_runtime(faux) -> ModelRuntime:
    """The harness recipe: an in-memory runtime with the faux provider
    registered and authenticated."""
    model = faux.get_model()
    auth_storage = AuthStorage.in_memory()

    async def set_key(_credential):
        return ApiKeyCredential(key="faux-key")

    await auth_storage.modify(model.provider, set_key)
    runtime = await ModelRuntime.create(credentials=auth_storage, models_path=None, allow_model_network=False)
    runtime.register_native_provider(faux.provider)
    await runtime.refresh(ModelsRefreshOptions(allow_network=False, providers=[faux.provider.id]))
    return runtime


def _patch_session_construction(request, tmp_path, runtime) -> None:
    """Point the runner's lean session construction at hermetic state: the
    prebuilt faux runtime instead of `ModelRuntime.create()` (which would read
    the developer's real auth/models files) and a temp agent dir. Restored via
    addfinalizer (predates tonio 0.9.14; `monkeypatch` works now)."""

    async def create():
        return runtime

    originals = (subagent_module.ModelRuntime, subagent_module.get_agent_dir)

    def restore() -> None:
        subagent_module.ModelRuntime, subagent_module.get_agent_dir = originals

    request.addfinalizer(restore)
    subagent_module.ModelRuntime = SimpleNamespace(create=create)
    subagent_module.get_agent_dir = lambda: str(tmp_path / "agent-dir")


def _agent(tmp_path, **overrides) -> AgentConfig:
    config = {
        "name": "echo",
        "description": "Test agent",
        "system_prompt": "You are a test subagent.",
        "source": "user",
        "file_path": str(tmp_path / "echo.md"),
    }
    config.update(overrides)
    return AgentConfig(**config)


def _make_details(results: list[dict]) -> dict:
    return {"mode": "single", "results": results}


@pytest.mark.tonio
async def test_run_single_agent_runs_an_in_process_session(tmp_path, request):
    faux = faux_provider()
    runtime = await _create_faux_runtime(faux)
    model = faux.get_model()
    _patch_session_construction(request, tmp_path, runtime)

    final_text = "The subagent looked around and found nothing worth reporting."
    faux.set_responses([faux_assistant_message(final_text)])

    cwd = tmp_path / "project"
    os.makedirs(cwd)
    agent = _agent(tmp_path)

    # (text, whether an assistant message had already landed) per update, read
    # at callback time because details alias the runner's mutable result dict.
    observed: list[tuple[str, bool]] = []

    def on_update(partial) -> None:
        messages = partial.details["results"][0]["messages"]
        has_assistant = any(m.get("role") == "assistant" for m in messages)
        observed.append((partial.content[0].text, has_assistant))

    result = await run_single_agent(
        str(cwd),
        {"model": f"{model.provider}/{model.id}", "thinkingLevel": None},
        [agent],
        "echo",
        "Look around.",
        None,
        None,
        None,
        on_update,
        _make_details,
    )

    assert result["status"] == "done"
    assert result["errorMessage"] is None if "errorMessage" in result else True
    assert "exitCode" not in result and "stderr" not in result
    assert get_final_output(result["messages"]) == final_text

    # Messages are wire dicts (camelCase), the shape details keep across
    # parent-session persistence.
    assistant = next(m for m in result["messages"] if m.get("role") == "assistant")
    assert assistant["stopReason"] == "stop"
    assert assistant["content"][0]["type"] == "text"

    # Usage was accumulated from the typed messages into the wire-dict shape.
    assert result["usage"]["turns"] == 1
    assert result["usage"]["output"] > 0
    assert result["usage"]["cost"] == assistant["usage"]["cost"]["total"]

    # Per-delta preview: at least one update streamed text before the
    # assistant message completed (the child-process variant only reported at
    # message boundaries).
    assert any(text and text != "(running...)" and not has_assistant for text, has_assistant in observed)
    assert observed[-1][0] == final_text


@pytest.mark.tonio
async def test_run_single_agent_fails_on_an_unknown_frontmatter_model(tmp_path, request):
    faux = faux_provider()
    runtime = await _create_faux_runtime(faux)
    _patch_session_construction(request, tmp_path, runtime)
    faux.set_responses([faux_assistant_message("never reached")])

    cwd = tmp_path / "project"
    os.makedirs(cwd)
    agent = _agent(tmp_path, model="no-such-provider/no-such-model")

    result = await run_single_agent(
        str(cwd), {"model": None, "thinkingLevel": None}, [agent], "echo", "Task", None, None, None, None, _make_details
    )

    assert result["status"] == "failed"
    assert result["errorMessage"]
    assert result["messages"] == []


@pytest.mark.tonio
async def test_map_with_concurrency_limit_bounds_in_flight_tasks():
    """The parallel mode's `concurrency` clamp is only as real as the helper's
    bound: with limit 3 over 8 items, exactly 3 tasks are in flight at once
    and results keep item order."""
    in_flight = {"count": 0, "peak": 0}
    guard = threading.Lock()
    release = tonio.Event()
    limit_reached = tonio.Event()

    async def fn(item: int, _index: int) -> int:
        with guard:
            in_flight["count"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["count"])
            if in_flight["count"] == 3:
                limit_reached.set()
        await release.wait(None)
        with guard:
            in_flight["count"] -= 1
        return item * 2

    join = tonio.spawn(map_with_concurrency_limit(list(range(8)), 3, fn))
    await limit_reached.wait(None)
    release.set()
    results = await join

    assert results == [item * 2 for item in range(8)]
    assert in_flight["peak"] == 3


@pytest.mark.tonio
async def test_run_single_agent_reports_an_unknown_agent(tmp_path):
    result = await run_single_agent(
        str(tmp_path), {"model": None, "thinkingLevel": None}, [], "ghost", "Task", None, 2, None, None, _make_details
    )

    assert result["status"] == "failed"
    assert 'Unknown agent: "ghost"' in result["errorMessage"]
    assert result["step"] == 2
