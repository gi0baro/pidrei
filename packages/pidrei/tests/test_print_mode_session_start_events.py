"""Mirror of pi coding-agent test/suite/print-mode-session-start-events.test.ts.

JSON print mode runs on a real `AgentSessionRuntime` whose factory loads the inline extension for each
runtime, as pi's `extensionFactories` do. pi's mocked output-guard is the module swap of
`test_print_mode.py`, and its faux provider the canned `stream_simple` of
`test_agent_session_runtime.py` (the `/renew` command never reaches it).
"""

import json

import pytest

from pidrei.core.agent_session_runtime import CreateAgentSessionRuntimeResult, create_agent_session_runtime
from pidrei.core.agent_session_services import (
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    create_agent_session_from_services,
)
from pidrei.core.session_manager import SessionManager
from pidrei.core.settings_manager import SettingsManager
from pidrei.modes import print_mode
from pidrei.modes.print_mode import PrintModeOptions, run_print_mode
from pidrei_ai.providers.all import get_builtin_model

from .agent_session_helpers import create_test_resource_loader
from .harness import create_test_extensions_result
from .test_agent_session_runtime import _create_model_runtime
from .test_print_mode import _patched


async def _factory(pi) -> None:
    async def on_session_start(event, _ctx):
        await pi.append_entry("start-marker", event["reason"])

    async def renew(_args, ctx):
        await ctx.new_session()

    pi.on("session_start", on_session_start)
    pi.register_command("renew", description="start a new session", handler=renew)


class TestPrintModeJsonOutputOfSessionStartEvents:
    @pytest.mark.tonio
    async def test_streams_entries_appended_by_session_start_at_startup_and_after_ctx_new_session(self, tmp_path):
        temp_dir = str(tmp_path)
        model = get_builtin_model("anthropic", "claude-sonnet-4-5")
        model_runtime = await _create_model_runtime(temp_dir)

        async def create_runtime(*, cwd, agent_dir, session_manager, session_start_event=None, **_kwargs):
            services = AgentSessionServices(
                cwd=cwd,
                agent_dir=agent_dir,
                model_runtime=model_runtime,
                settings_manager=await SettingsManager(cwd, agent_dir),
                resource_loader=create_test_resource_loader(await create_test_extensions_result([_factory], cwd)),
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

        runtime = await create_agent_session_runtime(
            create_runtime,
            cwd=temp_dir,
            agent_dir=temp_dir,
            session_manager=SessionManager.in_memory(temp_dir),
        )
        lines: list[str] = []

        async def drain():
            pass

        with _patched(print_mode, write_raw_stdout=lines.append, drain_output=drain):
            exit_code = await run_print_mode(runtime, PrintModeOptions(mode="json", messages=["/renew"]))

        assert exit_code == 0
        output = [json.loads(line) for chunk in lines for line in chunk.split("\n") if line.strip()]
        assert output[0].get("type") == "session"
        start_markers = [
            line["entry"].get("data")
            for line in output
            if line.get("type") == "entry_appended" and line["entry"].get("customType") == "start-marker"
        ]
        assert start_markers == ["startup", "new"]
