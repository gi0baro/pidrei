"""Mirror of pi coding-agent src/core/agent-session-runtime.ts."""

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import tonio.colored as tonio
from tonio.colored import fs, sync

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


def _copy_file_exclusive_blocking(source: str, destination: str) -> None:
    """`copyFileSync(..., COPYFILE_EXCL)`: fail instead of overwriting a file that appeared meanwhile."""
    with open(source, "rb") as src, open(destination, "xb") as dst:
        shutil.copyfileobj(src, dst)


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
        self._rebind_session: Callable[[AgentSession, Callable[[], None]], Any] | None = None
        # One session replacement at a time (spec/ui-island.md, the guards): the
        # before-hooks, teardown, creation, swap and rebind of one
        # replacement never interleave with another's. `with_session` runs
        # after it, so the callback may replace the session again.
        self._replacement_lock = sync.Lock()
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

    def set_rebind_session(
        self, rebind_session: Callable[[AgentSession, Callable[[], None]], Any] | None = None
    ) -> None:
        """`rebind_session(new_session, swap)`: rebind the host to the new
        session. It must call `swap()` (which makes `new_session` current),
        typically together with its first rebind step."""
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
        # Settle any active response first so the aborted turn (including tool
        # results) is persisted to the outgoing session before it is replaced.
        await self.session.abort()
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

    async def _rebind(self, result: CreateAgentSessionRuntimeResult) -> None:
        """pi's `apply` + `rebindSession`: swap in the new runtime and rebind
        the host. The rebind callback performs the swap itself (`swap`), so a
        host that guards its state applies it together with its first rebind
        stretch (spec/ui-island.md, whole changes); with no callback it happens here."""

        def swap() -> None:
            self._apply(result)

        if self._rebind_session is None:
            swap()
            return
        await self._rebind_session(result.session, swap)
        if self._session is not result.session:
            raise RuntimeError("the rebind_session callback must call swap()")

    async def _run_with_session(self, result: dict, with_session: Callable[[Any], Any] | None) -> dict:
        if with_session is not None and not result.get("cancelled"):
            context = self.session.create_replaced_session_context()
            try:
                await with_session(context)
            finally:
                # Prompt options the callback checked out publish when it returns.
                context.publish_system_prompt_options()
        return result

    async def switch_session(
        self,
        session_path: str,
        *,
        cwd_override: str | None = None,
        with_session: Callable[[Any], Any] | None = None,
        project_trust_context_factory: Callable[[str], Any] | None = None,
    ) -> dict[str, bool]:
        async with self._replacement_lock:
            result = await self._switch_session(session_path, cwd_override, project_trust_context_factory)
        return await self._run_with_session(result, with_session)

    async def _switch_session(
        self,
        session_path: str,
        cwd_override: str | None,
        project_trust_context_factory: Callable[[str], Any] | None,
    ) -> dict[str, bool]:
        before_result = await self._emit_before_switch("resume", session_path)
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self.session.session_file
        session_manager = await SessionManager(cwd_override, session_file=session_path)
        await assert_session_cwd_exists(session_manager, self.cwd)
        await self._teardown_current("resume", session_manager.get_session_file())
        await self._rebind(
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
        return {"cancelled": False}

    async def new_session(
        self,
        *,
        parent_session: str | None = None,
        setup: Callable[[SessionManager], Any] | None = None,
        with_session: Callable[[Any], Any] | None = None,
    ) -> dict[str, bool]:
        async with self._replacement_lock:
            result = await self._new_session(parent_session, setup)
        return await self._run_with_session(result, with_session)

    async def _new_session(
        self,
        parent_session: str | None,
        setup: Callable[[SessionManager], Any] | None,
    ) -> dict[str, bool]:
        before_result = await self._emit_before_switch("new")
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self.session.session_file
        session_dir = self.session.session_manager.get_session_dir()
        session_manager = (
            await SessionManager(self.cwd, session_dir)
            if self.session.session_manager.is_persisted()
            else SessionManager.in_memory(self.cwd)
        )
        if parent_session:
            session_manager.new_session({"parentSession": parent_session})

        await self._teardown_current("new", session_manager.get_session_file())
        result = await self._create_runtime(
            cwd=self.cwd,
            agent_dir=self.services.agent_dir,
            session_manager=session_manager,
            session_start_event={
                "type": "session_start",
                "reason": "new",
                "previousSessionFile": previous_session_file,
            },
        )
        if setup is not None:
            # pi runs `setup` after `apply`; the swap now happens inside the
            # rebind, so it runs on the new session directly.
            await setup(result.session.session_manager)
            result.session.refresh_context()
        await self._rebind(result)
        return {"cancelled": False}

    async def fork(
        self,
        entry_id: str,
        *,
        position: str = "before",
        with_session: Callable[[Any], Any] | None = None,
    ) -> dict[str, Any]:
        async with self._replacement_lock:
            result = await self._fork(entry_id, position)
        return await self._run_with_session(result, with_session)

    async def _fork(self, entry_id: str, position: str) -> dict[str, Any]:
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
                session_manager = await SessionManager(self.cwd, session_dir)
                session_manager.new_session({"parentSession": current_session_file})
                await self._teardown_current("fork", session_manager.get_session_file())
                await self._rebind(
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
                return {"cancelled": False, "selectedText": selected_text}

            if not await fs.Path(current_session_file).exists():
                raise Exception("This session has not been saved yet. Send a message before cloning or forking it.")
            session_manager = await SessionManager(session_dir=session_dir, session_file=current_session_file)
            forked_session_path = await session_manager.create_branched_session(target_leaf_id)
            if not forked_session_path:
                raise Exception("Failed to create forked session")
            await self._teardown_current("fork", session_manager.get_session_file())
            await self._rebind(
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
            return {"cancelled": False, "selectedText": selected_text}

        session_manager = self.session.session_manager
        await self._teardown_current("fork", session_manager.get_session_file())
        if not target_leaf_id:
            session_manager.new_session({"parentSession": previous_session_file})
        else:
            await session_manager.create_branched_session(target_leaf_id)
        await self._rebind(
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
        return {"cancelled": False, "selectedText": selected_text}

    async def import_from_jsonl(self, input_path: str, cwd_override: str | None = None) -> dict[str, bool]:
        """Import a session JSONL file and switch runtime state to it.

        Returns {"cancelled": True} when cancelled by session_before_switch.
        Raises SessionImportFileNotFoundError when the input path does not exist
        and MissingSessionCwdError when the imported session cwd is unresolvable."""
        async with self._replacement_lock:
            return await self._import_from_jsonl(input_path, cwd_override)

    async def _import_from_jsonl(self, input_path: str, cwd_override: str | None) -> dict[str, bool]:
        resolved_path = resolve_path(input_path)
        if not await fs.Path(resolved_path).exists():
            raise SessionImportFileNotFoundError(resolved_path)

        session_dir = self.session.session_manager.get_session_dir()
        if not await fs.Path(session_dir).exists():
            await fs.Path(session_dir).mkdir(parents=True, exist_ok=True)

        destination_path = os.path.join(session_dir, os.path.basename(resolved_path))
        source_already_stored = resolve_path(destination_path) == resolved_path
        if not source_already_stored:
            name, ext = os.path.splitext(os.path.basename(destination_path))
            suffix = 1
            while await fs.Path(destination_path).exists():
                destination_path = os.path.join(session_dir, f"{name}-{suffix}{ext}")
                suffix += 1
        before_result = await self._emit_before_switch("resume", destination_path)
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self.session.session_file
        if not source_already_stored:
            # `shutil.copyfile` has no `fs` equivalent, so it goes to the pool;
            # the exclusive create mirrors pi's COPYFILE_EXCL.
            await tonio.spawn_blocking(_copy_file_exclusive_blocking, resolved_path, destination_path)

        session_manager = await SessionManager(cwd_override, session_dir, session_file=destination_path)
        await assert_session_cwd_exists(session_manager, self.cwd)
        await self._teardown_current("resume", session_manager.get_session_file())
        await self._rebind(
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
    await assert_session_cwd_exists(session_manager, cwd)
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
