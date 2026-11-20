"""Mirror of pi coding-agent test/suite/regressions/10470-rpc-session-replacement-start-once.test.ts.

Regression for https://github.com/earendil-works/pi/issues/10470: an RPC session replacement ran the new
session's `session_start` twice, and events emitted by `session_start` handlers never reached the client.

RPC mode runs in-process on `test_rpc_mode.py`'s harness (pi mocks output-guard and the JSONL reader the
same way), over a real `AgentSessionRuntime` whose factory loads the inline extension for each runtime,
as pi's `extensionFactories` do. pi's faux provider becomes the canned `stream_simple` of
`test_agent_session_runtime.py`. pi's `vi.waitFor` polls become the harness's `until`; a replacement's
`session_start` handlers run inside the command, so its events are written before its response. The
listener snapshot and restore of pi's test have no counterpart: the harness owns the mode's input and
signal handlers.
"""

import os
from typing import Any

import pytest

from pidrei.core.agent_session_runtime import CreateAgentSessionRuntimeResult, create_agent_session_runtime
from pidrei.core.agent_session_services import (
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    create_agent_session_from_services,
)
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei_ai.providers.all import get_builtin_model

from .agent_session_helpers import create_test_resource_loader
from .harness import create_test_extensions_result
from .test_agent_session_runtime import _create_model_runtime
from .test_rpc_mode import _RpcHarness


async def _create_runtime(temp_dir: str, starts: list[str]):
    model = get_builtin_model("anthropic", "claude-sonnet-4-5")
    model_runtime = await _create_model_runtime(temp_dir)

    async def factory(pi) -> None:
        async def on_session_start(event, _ctx):
            marker = f"{event['reason']}:{len(starts)}"
            starts.append(marker)
            await pi.append_entry("start-marker", marker)

        pi.on("session_start", on_session_start)

    async def create_runtime(*, cwd, agent_dir, session_manager, session_start_event=None, **_kwargs):
        services = AgentSessionServices(
            cwd=cwd,
            agent_dir=agent_dir,
            model_runtime=model_runtime,
            settings_manager=await SettingsManager(cwd, agent_dir),
            resource_loader=create_test_resource_loader(await create_test_extensions_result([factory], cwd)),
        )
        result = await create_agent_session_from_services(
            CreateAgentSessionFromServicesOptions(
                services=services,
                session_manager=session_manager,
                session_start_event=session_start_event,
                model=model,
            )
        )
        return CreateAgentSessionRuntimeResult(
            session=result.session,
            services=services,
            extensions_result=result.extensions_result,
            model_fallback_message=result.model_fallback_message,
            diagnostics=services.diagnostics,
        )

    return await create_agent_session_runtime(
        create_runtime,
        cwd=temp_dir,
        agent_dir=temp_dir,
        session_manager=await SessionManager(temp_dir, os.path.join(temp_dir, "sessions")),
    )


def _streamed_start_entries(harness: _RpcHarness) -> list[str]:
    """Data of every streamed entry_appended event for the custom entries the extension writes in session_start."""
    return [
        record["entry"].get("data") or ""
        for record in harness.records()
        if record.get("type") == "entry_appended"
        and record["entry"].get("type") == "custom"
        and record["entry"].get("customType") == "start-marker"
    ]


async def _send(harness: _RpcHarness, command: dict[str, Any]) -> dict[str, Any]:
    request_id = f"req-{len(harness.output_lines)}"
    response = await harness.request({**command, "id": request_id})
    assert response.get("error") is None
    assert response.get("success") is True
    return response


async def _start_rpc(tmp_path) -> tuple[_RpcHarness, list[str]]:
    """Starts RPC mode on a fresh runtime; returns once it reads commands."""
    starts: list[str] = []
    runtime = await _create_runtime(str(tmp_path), starts)
    harness = _RpcHarness()
    await harness.start(runtime)
    return harness, starts


async def _prompt_once(harness: _RpcHarness) -> str:
    """Runs one prompt so the session has a user message and is written to disk; returns that message's entry id."""
    await _send(harness, {"type": "prompt", "message": "hello"})
    await harness.until(harness.settled)
    user_entry = next(
        (
            entry
            for entry in harness.runtime_host.session.session_manager.get_entries()
            if entry.get("type") == "message" and entry["message"].role == "user"
        ),
        None,
    )
    assert user_entry is not None
    return user_entry["id"]


async def _expect_one_start(harness: _RpcHarness, starts: list[str], command: dict[str, Any], reason: str) -> None:
    """Sends a replacing command and checks it ran session_start once and streamed the entry that handler appended."""
    before = len(starts)
    await _send(harness, command)
    marker = f"{reason}:{before}"
    assert starts[before:] == [marker]
    assert [entry for entry in _streamed_start_entries(harness) if entry == marker] == [marker]


class TestRpcSessionReplacementEmitsSessionStartOnce:
    @pytest.mark.tonio
    async def test_streams_entries_appended_by_the_startup_session_start(self, tmp_path):
        harness, starts = await _start_rpc(tmp_path)
        try:
            assert starts == ["startup:0"]
            assert _streamed_start_entries(harness) == ["startup:0"]
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_new_session(self, tmp_path):
        harness, starts = await _start_rpc(tmp_path)
        try:
            await _expect_one_start(harness, starts, {"type": "new_session"}, "new")
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_switch_session(self, tmp_path):
        harness, starts = await _start_rpc(tmp_path)
        try:
            await _prompt_once(harness)
            session_path = harness.runtime_host.session.session_file
            assert session_path is not None
            await _send(harness, {"type": "new_session"})
            await _expect_one_start(harness, starts, {"type": "switch_session", "sessionPath": session_path}, "resume")
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_fork(self, tmp_path):
        harness, starts = await _start_rpc(tmp_path)
        try:
            entry_id = await _prompt_once(harness)
            await _expect_one_start(harness, starts, {"type": "fork", "entryId": entry_id}, "fork")
        finally:
            await harness.stop()

    @pytest.mark.tonio
    async def test_clone(self, tmp_path):
        harness, starts = await _start_rpc(tmp_path)
        try:
            await _prompt_once(harness)
            await _expect_one_start(harness, starts, {"type": "clone"}, "fork")
        finally:
            await harness.stop()
