"""SSH remote execution.

Demonstrates delegating tool operations to a remote machine via SSH. When
--ssh is provided, read/write/edit/bash (and user ! commands) run on the
remote; without it, everything stays local.

Usage:
    pidrei -e ./examples/extensions/ssh.py --ssh user@host
    pidrei -e ./examples/extensions/ssh.py --ssh user@host:/remote/path

Requirements:
    - SSH key-based auth (no password prompts)
    - bash, base64 and file(1) on the remote
"""

import base64
import dataclasses
import os
import shlex
import subprocess

import tonio.colored as tonio

from pidrei.core.tools.bash import BashExecResult, create_bash_tool_definition
from pidrei.core.tools.edit import create_edit_tool_definition
from pidrei.core.tools.read import create_read_tool_definition
from pidrei.core.tools.write import create_write_tool_definition


async def _ssh_exec(pi, remote: str, command: str) -> str:
    """Run a one-shot command on the remote. Raises on nonzero exit."""
    result = await pi.exec("ssh", [remote, command])
    if result.code != 0:
        raise Exception(f"SSH failed ({result.code}): {result.stderr.strip()}")
    return result.stdout


class RemoteReadOperations:
    """ReadOperations against the remote: cat becomes base64 so binary files
    (images) survive the text pipe of pi.exec."""

    def __init__(self, pi, remote: str, remote_cwd: str, local_cwd: str):
        self._pi = pi
        self._remote = remote
        self._remote_cwd = remote_cwd
        self._local_cwd = local_cwd

    def _to_remote(self, path: str) -> str:
        return path.replace(self._local_cwd, self._remote_cwd, 1)

    async def read_file(self, absolute_path: str) -> bytes:
        encoded = await _ssh_exec(self._pi, self._remote, f"base64 < {shlex.quote(self._to_remote(absolute_path))}")
        return base64.b64decode(encoded)

    async def access(self, absolute_path: str) -> None:
        await _ssh_exec(self._pi, self._remote, f"test -r {shlex.quote(self._to_remote(absolute_path))}")

    async def detect_image_mime_type(self, absolute_path: str) -> str | None:
        try:
            mime = await _ssh_exec(
                self._pi, self._remote, f"file --mime-type -b {shlex.quote(self._to_remote(absolute_path))}"
            )
            mime = mime.strip()
            return mime if mime in ("image/jpeg", "image/png", "image/gif", "image/webp") else None
        except Exception:
            return None


class RemoteWriteOperations:
    """WriteOperations against the remote: content travels base64-encoded to
    dodge shell quoting."""

    def __init__(self, pi, remote: str, remote_cwd: str, local_cwd: str):
        self._pi = pi
        self._remote = remote
        self._remote_cwd = remote_cwd
        self._local_cwd = local_cwd

    def _to_remote(self, path: str) -> str:
        return path.replace(self._local_cwd, self._remote_cwd, 1)

    async def write_file(self, absolute_path: str, content: str) -> None:
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        await _ssh_exec(
            self._pi,
            self._remote,
            f"echo {shlex.quote(encoded)} | base64 -d > {shlex.quote(self._to_remote(absolute_path))}",
        )

    async def mkdir(self, dir: str) -> None:
        await _ssh_exec(self._pi, self._remote, f"mkdir -p {shlex.quote(self._to_remote(dir))}")


class RemoteEditOperations:
    """EditOperations: read + write + access, composed from the two above."""

    def __init__(self, pi, remote: str, remote_cwd: str, local_cwd: str):
        self._read = RemoteReadOperations(pi, remote, remote_cwd, local_cwd)
        self._write = RemoteWriteOperations(pi, remote, remote_cwd, local_cwd)

    async def read_file(self, absolute_path: str) -> bytes:
        return await self._read.read_file(absolute_path)

    async def access(self, absolute_path: str) -> None:
        await self._read.access(absolute_path)

    async def write_file(self, absolute_path: str, content: str) -> None:
        await self._write.write_file(absolute_path, content)


class RemoteBashOperations:
    """BashOperations running ssh as a local subprocess, streaming output.

    pi.exec cannot stream, so this mirrors LocalBashOperations' structure on
    tonio.open_process; killing the local ssh client stands in for killing the
    remote process tree.
    """

    def __init__(self, remote: str, remote_cwd: str, local_cwd: str):
        self._remote = remote
        self._remote_cwd = remote_cwd
        self._local_cwd = local_cwd

    async def exec(self, command: str, cwd: str, *, on_data, cancel=None, timeout=None, env=None) -> BashExecResult:
        del env  # the remote shell provides its own environment
        remote_cwd = cwd.replace(self._local_cwd, self._remote_cwd, 1)
        remote_command = f"cd {shlex.quote(remote_cwd)} && {command}"
        process = await tonio.open_process(
            ["ssh", self._remote, remote_command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        timed_out = False
        exited = tonio.Event()

        def kill() -> None:
            try:
                process.kill()
            except Exception:
                pass

        async def read_stream(stream) -> None:
            if stream is None:
                return
            try:
                while True:
                    chunk = await stream.receive_some()
                    if not chunk:
                        return
                    on_data(chunk)
            except Exception:
                pass  # Broken pipe after kill.

        async def watchdog() -> None:
            nonlocal timed_out
            await exited.wait(timeout)
            if not exited.is_set():
                timed_out = True
                kill()

        watchdog_join = tonio.spawn(watchdog()) if timeout else None
        unsubscribe = cancel.on_cancel(lambda _reason: kill()) if cancel is not None else None

        try:
            await tonio.spawn(read_stream(process.stdout), read_stream(process.stderr))
            exit_code = await process.wait()
        finally:
            exited.set()
            if watchdog_join is not None:
                await watchdog_join
            if unsubscribe is not None:
                unsubscribe()

        if cancel is not None and cancel.cancelled:
            raise Exception("aborted")
        if timed_out:
            raise Exception(f"timeout:{timeout:g}")
        return BashExecResult(exit_code=exit_code)


def extension(pi):
    pi.register_flag("ssh", type="string", description="SSH remote: user@host or user@host:/path")

    local_cwd = os.getcwd()
    local_read = create_read_tool_definition(local_cwd)
    local_write = create_write_tool_definition(local_cwd)
    local_edit = create_edit_tool_definition(local_cwd)
    local_bash = create_bash_tool_definition(local_cwd)

    # Resolved lazily on session_start (CLI flags not available during factory)
    state: dict = {"remote": None, "remote_cwd": None}

    def remote_bash_operations() -> RemoteBashOperations:
        return RemoteBashOperations(state["remote"], state["remote_cwd"], local_cwd)

    def wrap(local_definition, create_definition, remote_operations_cls):
        """Same-named override of the built-in tool: at call time, route to a
        remote-operations definition when SSH is on, the local one otherwise."""

        async def execute(tool_call_id, params, cancel=None, on_update=None, ctx=None):
            if state["remote"] is not None:
                definition = create_definition(
                    local_cwd,
                    operations=remote_operations_cls(pi, state["remote"], state["remote_cwd"], local_cwd),
                )
                return await definition.execute(tool_call_id, params, cancel, on_update, ctx)
            return await local_definition.execute(tool_call_id, params, cancel, on_update, ctx)

        return dataclasses.replace(local_definition, execute=execute)

    pi.register_tool(wrap(local_read, create_read_tool_definition, RemoteReadOperations))
    pi.register_tool(wrap(local_write, create_write_tool_definition, RemoteWriteOperations))
    pi.register_tool(wrap(local_edit, create_edit_tool_definition, RemoteEditOperations))

    async def execute_bash(tool_call_id, params, cancel=None, on_update=None, ctx=None):
        if state["remote"] is not None:
            definition = create_bash_tool_definition(local_cwd, operations=remote_bash_operations())
            return await definition.execute(tool_call_id, params, cancel, on_update, ctx)
        return await local_bash.execute(tool_call_id, params, cancel, on_update, ctx)

    pi.register_tool(dataclasses.replace(local_bash, execute=execute_bash))

    async def on_session_start(_event, ctx) -> None:
        # Resolve SSH config now that CLI flags are available
        arg = pi.get_flag("ssh")
        if not arg:
            return
        if ":" in arg:
            remote, _, path = arg.partition(":")
        else:
            # No path given, evaluate pwd on remote
            remote = arg
            path = (await _ssh_exec(pi, remote, "pwd")).strip()
        state["remote"] = remote
        state["remote_cwd"] = path
        ctx.ui.set_status("ssh", ctx.ui.theme.fg("accent", f"SSH: {remote}:{path}"))
        ctx.ui.notify(f"SSH mode: {remote}:{path}", "info")

    # Handle user ! commands via SSH
    async def on_user_bash(_event, _ctx):
        if state["remote"] is None:
            return None  # No SSH, use local execution
        return {"operations": remote_bash_operations()}

    # Replace local cwd with remote cwd in system prompt
    async def on_before_agent_start(event, _ctx):
        if state["remote"] is None:
            return None
        modified = event["systemPrompt"].replace(
            f"Current working directory: {local_cwd}",
            f"Current working directory: {state['remote_cwd']} (via SSH: {state['remote']})",
        )
        return {"systemPrompt": modified}

    pi.on("session_start", on_session_start)
    pi.on("user_bash", on_user_bash)
    pi.on("before_agent_start", on_before_agent_start)
