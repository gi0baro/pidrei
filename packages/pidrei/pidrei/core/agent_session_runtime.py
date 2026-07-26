"""Mirror of pi coding-agent src/core/agent-session-runtime.ts."""

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..utils.paths import resolve_path
from .agent_session import AgentSession
from .agent_session_services import AgentSessionRuntimeDiagnostic, AgentSessionServices
from .extensions.runner import emit_session_shutdown_event
from .session_cwd import assert_session_cwd_exists
from .session_manager import SessionManager


@dataclass(slots=True)
class CreateAgentSessionRuntimeResult:
    """Result returned by runtime creation: the created session, its cwd-bound
    services, and all diagnostics collected during setup."""

    session: AgentSession
    services: AgentSessionServices
    extensions_result: Any = None
    model_fallback_message: str | None = None
    diagnostics: list[AgentSessionRuntimeDiagnostic] = field(default_factory=list)


# CreateAgentSessionRuntimeFactory: async (options: dict) -> CreateAgentSessionRuntimeResult
# where options carries cwd, agent_dir, session_manager, session_start_event,
# project_trust_context.
CreateAgentSessionRuntimeFactory = Callable[..., Any]


class SessionImportFileNotFoundError(Exception):
    """Thrown when /import references a JSONL file path that does not exist."""

    def __init__(self, file_path: str):
        super().__init__(f"File not found: {file_path}")
        self.name = "SessionImportFileNotFoundError"
        self.file_path = file_path


def _extract_user_message_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    return "".join(
        part.text
        for part in content
        if getattr(part, "type", None) == "text" and isinstance(getattr(part, "text", None), str)
    )


class AgentSessionRuntime:
    """Owns the current AgentSession plus its cwd-bound services.

    Session replacement methods tear down the current runtime first, then
    create and apply the next runtime. If creation fails, the error propagates
    to the caller, who owns user-facing error handling."""

    def __init__(
        self,
        session: AgentSession,
        services: AgentSessionServices,
        create_runtime: CreateAgentSessionRuntimeFactory,
        diagnostics: list[AgentSessionRuntimeDiagnostic] | None = None,
        model_fallback_message: str | None = None,
    ):
        self._rebind_session: Callable[[AgentSession], Any] | None = None
        self._before_session_invalidate: Callable[[], None] | None = None
        self._session = session
        self._services = services
        self._create_runtime = create_runtime
        self._diagnostics = diagnostics if diagnostics is not None else []
        self._model_fallback_message = model_fallback_message

    @property
    def services(self) -> AgentSessionServices:
        return self._services

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def cwd(self) -> str:
        return self._services.cwd

    @property
    def diagnostics(self) -> list[AgentSessionRuntimeDiagnostic]:
        return self._diagnostics

    @property
    def model_fallback_message(self) -> str | None:
        return self._model_fallback_message

    def set_rebind_session(self, rebind_session: Callable[[AgentSession], Any] | None = None) -> None:
        self._rebind_session = rebind_session

    def set_before_session_invalidate(self, before_session_invalidate: Callable[[], None] | None = None) -> None:
        """Set a synchronous callback that runs after session_shutdown handlers
        finish but before the current session is invalidated. For host-owned UI
        teardown that must not yield to the scheduler."""
        self._before_session_invalidate = before_session_invalidate

    async def _emit_before_switch(self, reason: str, target_session_file: str | None = None) -> dict[str, bool]:
        runner = self.session.extension_runner
        if not runner.has_handlers("session_before_switch"):
            return {"cancelled": False}

        result = await runner.emit(
            {"type": "session_before_switch", "reason": reason, "targetSessionFile": target_session_file}
        )
        return {"cancelled": bool(isinstance(result, dict) and result.get("cancel") is True)}

    async def _emit_before_fork(self, entry_id: str, position: str) -> dict[str, bool]:
        runner = self.session.extension_runner
        if not runner.has_handlers("session_before_fork"):
            return {"cancelled": False}

        result = await runner.emit({"type": "session_before_fork", "entryId": entry_id, "position": position})
        return {"cancelled": bool(isinstance(result, dict) and result.get("cancel") is True)}

    async def _teardown_current(self, reason: str, target_session_file: str | None = None) -> None:
        await emit_session_shutdown_event(
            self.session.extension_runner,
            {"type": "session_shutdown", "reason": reason, "targetSessionFile": target_session_file},
        )
        if self._before_session_invalidate is not None:
            self._before_session_invalidate()
        self.session.dispose()

    def _apply(self, result: CreateAgentSessionRuntimeResult) -> None:
        self._session = result.session
        self._services = result.services
        self._diagnostics = result.diagnostics
        self._model_fallback_message = result.model_fallback_message

    async def _finish_session_replacement(self, with_session: Callable[[Any], Any] | None = None) -> None:
        if self._rebind_session is not None:
            await self._rebind_session(self.session)
        if with_session is not None:
            await with_session(self.session.create_replaced_session_context())

    async def switch_session(
        self,
        session_path: str,
        *,
        cwd_override: str | None = None,
        with_session: Callable[[Any], Any] | None = None,
        project_trust_context_factory: Callable[[str], Any] | None = None,
    ) -> dict[str, bool]:
        before_result = await self._emit_before_switch("resume", session_path)
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self.session.session_file
        session_manager = SessionManager.open(session_path, None, cwd_override)
        assert_session_cwd_exists(session_manager, self.cwd)
        await self._teardown_current("resume", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=session_manager.get_cwd(),
                agent_dir=self.services.agent_dir,
                session_manager=session_manager,
                session_start_event={
                    "type": "session_start",
                    "reason": "resume",
                    "previousSessionFile": previous_session_file,
                },
                project_trust_context=(
                    project_trust_context_factory(session_manager.get_cwd())
                    if project_trust_context_factory is not None
                    else None
                ),
            )
        )
        await self._finish_session_replacement(with_session)
        return {"cancelled": False}

    async def new_session(
        self,
        *,
        parent_session: str | None = None,
        setup: Callable[[SessionManager], Any] | None = None,
        with_session: Callable[[Any], Any] | None = None,
    ) -> dict[str, bool]:
        before_result = await self._emit_before_switch("new")
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self.session.session_file
        session_dir = self.session.session_manager.get_session_dir()
        session_manager = (
            SessionManager.create(self.cwd, session_dir)
            if self.session.session_manager.is_persisted()
            else SessionManager.in_memory(self.cwd)
        )
        if parent_session:
            session_manager.new_session({"parentSession": parent_session})

        await self._teardown_current("new", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=self.cwd,
                agent_dir=self.services.agent_dir,
                session_manager=session_manager,
                session_start_event={
                    "type": "session_start",
                    "reason": "new",
                    "previousSessionFile": previous_session_file,
                },
            )
        )
        if setup is not None:
            await setup(self.session.session_manager)
            self.session.agent.state.messages = self.session.session_manager.build_session_context().messages
        await self._finish_session_replacement(with_session)
        return {"cancelled": False}

    async def fork(
        self,
        entry_id: str,
        *,
        position: str = "before",
        with_session: Callable[[Any], Any] | None = None,
    ) -> dict[str, Any]:
        before_result = await self._emit_before_fork(entry_id, position)
        if before_result["cancelled"]:
            return {"cancelled": True}
        selected_text: str | None = None

        selected_entry = self.session.session_manager.get_entry(entry_id)
        if selected_entry is None:
            raise Exception("Invalid entry ID for forking")

        if position == "at":
            target_leaf_id: str | None = selected_entry["id"]
        else:
            if (
                selected_entry.get("type") != "message"
                or getattr(selected_entry.get("message"), "role", None) != "user"
            ):
                raise Exception("Invalid entry ID for forking")
            target_leaf_id = selected_entry.get("parentId")
            selected_text = _extract_user_message_text(selected_entry["message"].content)

        previous_session_file = self.session.session_file
        if self.session.session_manager.is_persisted():
            current_session_file = self.session.session_file
            if not current_session_file:
                raise Exception("Persisted session is missing a session file")
            session_dir = self.session.session_manager.get_session_dir()
            if not target_leaf_id:
                session_manager = SessionManager.create(self.cwd, session_dir)
                session_manager.new_session({"parentSession": current_session_file})
                await self._teardown_current("fork", session_manager.get_session_file())
                self._apply(
                    await self._create_runtime(
                        cwd=self.cwd,
                        agent_dir=self.services.agent_dir,
                        session_manager=session_manager,
                        session_start_event={
                            "type": "session_start",
                            "reason": "fork",
                            "previousSessionFile": previous_session_file,
                        },
                    )
                )
                await self._finish_session_replacement(with_session)
                return {"cancelled": False, "selectedText": selected_text}

            if not os.path.exists(current_session_file):
                raise Exception(
                    "This session has not been saved yet. Wait for the first assistant response "
                    "before cloning or forking it."
                )
            session_manager = SessionManager.open(current_session_file, session_dir)
            forked_session_path = session_manager.create_branched_session(target_leaf_id)
            if not forked_session_path:
                raise Exception("Failed to create forked session")
            await self._teardown_current("fork", session_manager.get_session_file())
            self._apply(
                await self._create_runtime(
                    cwd=session_manager.get_cwd(),
                    agent_dir=self.services.agent_dir,
                    session_manager=session_manager,
                    session_start_event={
                        "type": "session_start",
                        "reason": "fork",
                        "previousSessionFile": previous_session_file,
                    },
                )
            )
            await self._finish_session_replacement(with_session)
            return {"cancelled": False, "selectedText": selected_text}

        session_manager = self.session.session_manager
        if not target_leaf_id:
            session_manager.new_session({"parentSession": self.session.session_file})
        else:
            session_manager.create_branched_session(target_leaf_id)
        await self._teardown_current("fork", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=self.cwd,
                agent_dir=self.services.agent_dir,
                session_manager=session_manager,
                session_start_event={
                    "type": "session_start",
                    "reason": "fork",
                    "previousSessionFile": previous_session_file,
                },
            )
        )
        await self._finish_session_replacement(with_session)
        return {"cancelled": False, "selectedText": selected_text}

    async def import_from_jsonl(self, input_path: str, cwd_override: str | None = None) -> dict[str, bool]:
        """Import a session JSONL file and switch runtime state to it.

        Returns {"cancelled": True} when cancelled by session_before_switch.
        Raises SessionImportFileNotFoundError when the input path does not exist
        and MissingSessionCwdError when the imported session cwd is unresolvable."""
        resolved_path = resolve_path(input_path)
        if not os.path.exists(resolved_path):
            raise SessionImportFileNotFoundError(resolved_path)

        session_dir = self.session.session_manager.get_session_dir()
        if not os.path.exists(session_dir):
            os.makedirs(session_dir, exist_ok=True)

        destination_path = os.path.join(session_dir, os.path.basename(resolved_path))
        before_result = await self._emit_before_switch("resume", destination_path)
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self.session.session_file
        if resolve_path(destination_path) != resolved_path:
            shutil.copyfile(resolved_path, destination_path)

        session_manager = SessionManager.open(destination_path, session_dir, cwd_override)
        assert_session_cwd_exists(session_manager, self.cwd)
        await self._teardown_current("resume", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=session_manager.get_cwd(),
                agent_dir=self.services.agent_dir,
                session_manager=session_manager,
                session_start_event={
                    "type": "session_start",
                    "reason": "resume",
                    "previousSessionFile": previous_session_file,
                },
            )
        )
        await self._finish_session_replacement()
        return {"cancelled": False}

    async def dispose(self) -> None:
        await emit_session_shutdown_event(self.session.extension_runner, {"type": "session_shutdown", "reason": "quit"})
        if self._before_session_invalidate is not None:
            self._before_session_invalidate()
        self.session.dispose()


async def create_agent_session_runtime(
    create_runtime: CreateAgentSessionRuntimeFactory,
    *,
    cwd: str,
    agent_dir: str,
    session_manager: SessionManager,
    session_start_event: dict[str, Any] | None = None,
) -> AgentSessionRuntime:
    """Create the initial runtime from a runtime factory and initial session
    target. The same factory is stored on the returned AgentSessionRuntime and
    reused for later /new, /resume, /fork, and import flows."""
    assert_session_cwd_exists(session_manager, cwd)
    result = await create_runtime(
        cwd=cwd,
        agent_dir=agent_dir,
        session_manager=session_manager,
        session_start_event=session_start_event,
    )
    return AgentSessionRuntime(
        result.session,
        result.services,
        create_runtime,
        result.diagnostics,
        result.model_fallback_message,
    )
