"""Local execution environment (port of pi `harness/env/nodejs.ts`).

`NodeExecutionEnv` becomes `LocalExecutionEnv`: filesystem operations run
through `tonio.spawn_blocking` (never blocking a runtime worker), and shell
execution runs on `tonio.open_process` with a detached process group
(`start_new_session`), SIGKILL tree-kill, timeout watchdog, and the same
post-exit stdio grace window Node's `waitForChildProcess` implements.

Deviations from pi (documented):
- POSIX-only: tonio does not support Windows, so pi's win32 branches (Git-bash
  discovery, `taskkill` tree-kill, legacy-WSL stdin command transport, `~\\`
  expansion) are not ported.
- File operations honor `CancelToken` at operation boundaries (before/between
  blocking steps); Node additionally aborts mid-flight via `fs` signals.
- `findBashOnPath` uses `shutil.which` instead of spawning `which` (same PATH
  semantics, no subprocess).
- Text decoding uses `errors="replace"`, mirroring Node's UTF-8 decoding.
"""

import codecs
import errno as errno_module
import math
import os
import shutil
import signal
import stat as stat_module
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.request import url2pathname

import tonio.colored as tonio

from pidrei_ai.utils.cancel import CancelToken

from ..types import (
    Err,
    ExecutionError,
    FileError,
    FileInfo,
    FileKind,
    Result,
    ShellExecOptions,
    ShellExecResult,
    err,
    ok,
    to_error,
)


MAX_TIMEOUT_MS = 2_147_483_647
MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS / 1000
EXIT_STDIO_GRACE_SECONDS = 0.1


def _resolve_timeout_seconds(timeout: float | None) -> Result[float | None, ExecutionError]:
    if timeout is None:
        return ok(None)
    if not math.isfinite(timeout) or timeout <= 0:
        return err(ExecutionError("timeout", "Invalid timeout: must be a finite number of seconds"))
    if timeout * 1000 > MAX_TIMEOUT_MS:
        return err(ExecutionError("timeout", f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds"))
    return ok(timeout)


def _resolve_path(cwd: str, path: str) -> str:
    normalized = path
    if normalized == "~":
        normalized = str(Path.home())
    elif normalized.startswith("~/"):
        normalized = os.path.join(str(Path.home()), normalized[2:])
    elif normalized.startswith("file://"):
        try:
            normalized = url2pathname(normalized, require_scheme=True)
        except Exception:
            # Keep malformed URLs as ordinary paths so filesystem methods
            # preserve their non-raising contract.
            pass
    if os.path.isabs(normalized):
        return os.path.normpath(normalized)
    return os.path.normpath(os.path.join(cwd, normalized))


def _file_kind_from_mode(mode: int) -> FileKind | None:
    if stat_module.S_ISREG(mode):
        return "file"
    if stat_module.S_ISDIR(mode):
        return "directory"
    if stat_module.S_ISLNK(mode):
        return "symlink"
    return None


def _file_info_from_stat(path: str, stat_result: os.stat_result) -> Result[FileInfo, FileError]:
    kind = _file_kind_from_mode(stat_result.st_mode)
    if kind is None:
        return err(FileError("invalid", "Unsupported file type", path))
    name = os.path.basename(path.rstrip("/")) or path
    return ok(
        FileInfo(
            name=name,
            path=path,
            kind=kind,
            size=stat_result.st_size,
            mtime_ms=stat_result.st_mtime * 1000,
        )
    )


def _to_file_error(error: Exception, fallback_path: str | None = None) -> FileError:
    if isinstance(error, FileError):
        return error
    cause = to_error(error)
    message = str(cause)
    # Prefer the path the OS actually reported over the caller's fallback.
    path = error.filename if isinstance(error, OSError) and isinstance(error.filename, str) else fallback_path
    if isinstance(error, OSError):
        match error.errno:
            case errno_module.ENOENT:
                return FileError("not_found", message, path, cause)
            case errno_module.EACCES | errno_module.EPERM:
                return FileError("permission_denied", message, path, cause)
            case errno_module.ENOTDIR:
                return FileError("not_directory", message, path, cause)
            case errno_module.EISDIR:
                return FileError("is_directory", message, path, cause)
            case errno_module.EINVAL:
                return FileError("invalid", message, path, cause)
    return FileError("unknown", message, path, cause)


def _abort_result(cancel: CancelToken | None, path: str | None = None) -> Err[FileError] | None:
    if cancel is not None and cancel.cancelled:
        return err(FileError("aborted", "aborted", path))
    return None


def _path_exists(path: str) -> bool:
    return os.access(path, os.F_OK)


def _find_bash_on_path() -> str | None:
    found = shutil.which("bash")
    if found and _path_exists(found):
        return found
    return None


@dataclass(slots=True)
class ShellConfig:
    shell: str
    args: list[str]


async def _get_shell_config(custom_shell_path: str | None) -> Result[ShellConfig, ExecutionError]:
    if custom_shell_path:
        if await tonio.spawn_blocking(_path_exists, custom_shell_path):
            return ok(ShellConfig(custom_shell_path, ["-c"]))
        return err(ExecutionError("shell_unavailable", f"Custom shell path not found: {custom_shell_path}"))
    if await tonio.spawn_blocking(_path_exists, "/bin/bash"):
        return ok(ShellConfig("/bin/bash", ["-c"]))
    bash_on_path = await tonio.spawn_blocking(_find_bash_on_path)
    if bash_on_path:
        return ok(ShellConfig(bash_on_path, ["-c"]))
    return ok(ShellConfig("sh", ["-c"]))


def _get_shell_env(
    base_env: dict[str, str] | None,
    extra_env: dict[str, str] | None,
    inherit_env: bool = True,
) -> dict[str, str]:
    if not inherit_env:
        return {**(extra_env or {})}
    return {**os.environ, **(base_env or {}), **(extra_env or {})}


def kill_process_tree(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            # Process already dead.
            pass


class LocalExecutionEnv:
    """Local `ExecutionEnv` implementation (pi: `NodeExecutionEnv`)."""

    def __init__(self, cwd: str, shell_path: str | None = None, shell_env: dict[str, str] | None = None):
        self.cwd = cwd
        self._shell_path = shell_path
        self._shell_env = shell_env
        self._active_child_pids: set[int] = set()
        self._active_child_pids_lock = threading.Lock()

    # --- filesystem -----------------------------------------------------------

    async def absolute_path(self, path: str, cancel: CancelToken | None = None) -> Result[str, FileError]:
        return ok(_resolve_path(self.cwd, path))

    async def join_path(self, parts: list[str], cancel: CancelToken | None = None) -> Result[str, FileError]:
        if not parts:
            return ok(".")
        return ok(os.path.normpath(os.path.join(*parts)))

    async def read_text_file(self, path: str, cancel: CancelToken | None = None) -> Result[str, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if (aborted := _abort_result(cancel, resolved)) is not None:
            return aborted

        def read() -> str:
            # newline="" disables universal-newline translation: Node's
            # readFile preserves \r\n and the edit tool depends on it.
            with open(resolved, encoding="utf-8", errors="replace", newline="") as file:
                return file.read()

        try:
            return ok(await tonio.spawn_blocking(read))
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def read_text_lines(
        self, path: str, max_lines: int | None = None, cancel: CancelToken | None = None
    ) -> Result[list[str], FileError]:
        resolved = _resolve_path(self.cwd, path)
        if (aborted := _abort_result(cancel, resolved)) is not None:
            return aborted
        if max_lines is not None and max_lines <= 0:
            return ok([])

        def read() -> Result[list[str], FileError]:
            lines: list[str] = []
            with open(resolved, encoding="utf-8", errors="replace") as file:
                for line in file:
                    if (loop_abort := _abort_result(cancel, resolved)) is not None:
                        return loop_abort
                    lines.append(line.removesuffix("\n"))
                    if max_lines is not None and len(lines) >= max_lines:
                        break
            if (after_abort := _abort_result(cancel, resolved)) is not None:
                return after_abort
            return ok(lines)

        try:
            return await tonio.spawn_blocking(read)
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def read_binary_file(self, path: str, cancel: CancelToken | None = None) -> Result[bytes, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if (aborted := _abort_result(cancel, resolved)) is not None:
            return aborted

        def read() -> bytes:
            with open(resolved, "rb") as file:
                return file.read()

        try:
            return ok(await tonio.spawn_blocking(read))
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def write_file(
        self, path: str, content: str | bytes, cancel: CancelToken | None = None
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        if (aborted := _abort_result(cancel, resolved)) is not None:
            return aborted
        try:
            await tonio.spawn_blocking(os.makedirs, os.path.dirname(resolved) or ".", exist_ok=True)
            if (after_mkdir_abort := _abort_result(cancel, resolved)) is not None:
                return after_mkdir_abort

            def write() -> None:
                data = content.encode("utf-8") if isinstance(content, str) else content
                with open(resolved, "wb") as file:
                    file.write(data)

            await tonio.spawn_blocking(write)
            return ok(None)
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def append_file(
        self, path: str, content: str | bytes, cancel: CancelToken | None = None
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        try:
            await tonio.spawn_blocking(os.makedirs, os.path.dirname(resolved) or ".", exist_ok=True)

            def append() -> None:
                data = content.encode("utf-8") if isinstance(content, str) else content
                with open(resolved, "ab") as file:
                    file.write(data)

            await tonio.spawn_blocking(append)
            return ok(None)
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def rename_file(
        self, source_path: str, destination_path: str, cancel: CancelToken | None = None
    ) -> Result[None, FileError]:
        source = _resolve_path(self.cwd, source_path)
        destination = _resolve_path(self.cwd, destination_path)
        if (aborted := _abort_result(cancel, destination)) is not None:
            return aborted
        try:
            await tonio.spawn_blocking(os.replace, source, destination)
            return ok(None)
        except Exception as error:
            return err(_to_file_error(error, source))

    async def file_info(self, path: str, cancel: CancelToken | None = None) -> Result[FileInfo, FileError]:
        resolved = _resolve_path(self.cwd, path)
        try:
            return _file_info_from_stat(resolved, await tonio.spawn_blocking(os.lstat, resolved))
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def list_dir(self, path: str, cancel: CancelToken | None = None) -> Result[list[FileInfo], FileError]:
        resolved = _resolve_path(self.cwd, path)
        if (aborted := _abort_result(cancel, resolved)) is not None:
            return aborted

        # One pool hop lists and stats the whole directory: a hop per entry
        # made a 500-entry listing cost 500 round-trips.
        def list_and_stat() -> list[FileInfo]:
            infos: list[FileInfo] = []
            for entry in os.listdir(resolved):
                entry_path = os.path.normpath(os.path.join(resolved, entry))
                info = _file_info_from_stat(entry_path, os.lstat(entry_path))
                if info.ok:
                    infos.append(info.value)
            return infos

        try:
            return ok(await tonio.spawn_blocking(list_and_stat))
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def canonical_path(self, path: str, cancel: CancelToken | None = None) -> Result[str, FileError]:
        resolved = _resolve_path(self.cwd, path)
        try:
            return ok(await tonio.spawn_blocking(os.path.realpath, resolved, strict=True))
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def exists(self, path: str, cancel: CancelToken | None = None) -> Result[bool, FileError]:
        result = await self.file_info(path, cancel)
        if result.ok:
            return ok(True)
        if result.error.code == "not_found":
            return ok(False)
        return err(result.error)

    async def create_dir(
        self, path: str, recursive: bool = True, cancel: CancelToken | None = None
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        try:
            if recursive:
                # Node's recursive mkdir is idempotent for existing directories.
                await tonio.spawn_blocking(os.makedirs, resolved, exist_ok=True)
            else:
                await tonio.spawn_blocking(os.mkdir, resolved)
            return ok(None)
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def remove(
        self, path: str, recursive: bool = False, force: bool = False, cancel: CancelToken | None = None
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)

        def do_remove() -> None:
            try:
                mode = os.lstat(resolved).st_mode
            except FileNotFoundError:
                if force:
                    return
                raise
            if stat_module.S_ISDIR(mode):
                if recursive:
                    shutil.rmtree(resolved)
                else:
                    os.rmdir(resolved)
            else:
                os.remove(resolved)

        try:
            await tonio.spawn_blocking(do_remove)
            return ok(None)
        except Exception as error:
            return err(_to_file_error(error, resolved))

    async def create_temp_dir(self, prefix: str = "tmp-", cancel: CancelToken | None = None) -> Result[str, FileError]:
        try:
            return ok(await tonio.spawn_blocking(tempfile.mkdtemp, prefix=prefix))
        except Exception as error:
            return err(_to_file_error(error))

    async def create_temp_file(
        self, prefix: str = "", suffix: str = "", cancel: CancelToken | None = None
    ) -> Result[str, FileError]:
        directory = await self.create_temp_dir("tmp-")
        if not directory.ok:
            return directory
        file_path = os.path.join(directory.value, f"{prefix}{uuid.uuid4()}{suffix}")

        def create() -> None:
            with open(file_path, "wb"):
                pass

        try:
            await tonio.spawn_blocking(create)
            return ok(file_path)
        except Exception as error:
            return err(_to_file_error(error, file_path))

    # --- shell ----------------------------------------------------------------

    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[ShellExecResult, ExecutionError]:
        options = options if options is not None else ShellExecOptions()
        if options.cancel is not None and options.cancel.cancelled:
            return err(ExecutionError("aborted", "aborted"))
        timeout_result = _resolve_timeout_seconds(options.timeout)
        if not timeout_result.ok:
            return timeout_result
        timeout_seconds = timeout_result.value

        cwd = _resolve_path(self.cwd, options.cwd) if options.cwd else self.cwd
        shell_config = await _get_shell_config(self._shell_path)
        if not shell_config.ok:
            return shell_config
        if not await tonio.spawn_blocking(_path_exists, cwd):
            return err(
                ExecutionError(
                    "spawn_error",
                    f"Working directory does not exist: {cwd}\nCannot execute bash commands.",
                )
            )

        argv = [shell_config.value.shell, *shell_config.value.args, command]

        try:
            process = await tonio.open_process(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=_get_shell_env(self._shell_env, options.env, options.inherit_env),
                start_new_session=True,
            )
        except Exception as error:
            cause = to_error(error)
            return err(ExecutionError("spawn_error", str(cause), cause))

        pid = process.pid
        with self._active_child_pids_lock:
            self._active_child_pids.add(pid)

        state_lock = threading.Lock()
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        callback_error: ExecutionError | None = None
        timed_out = False
        activity_count = 0
        readers_done = tonio.Event()

        def handle_chunk(parts: list[str], callback, text: str) -> None:
            nonlocal callback_error, activity_count
            with state_lock:
                parts.append(text)
                activity_count += 1
            if callback is not None:
                try:
                    callback(text)
                except Exception as error:
                    cause = to_error(error)
                    with state_lock:
                        if callback_error is None:
                            callback_error = ExecutionError("callback_error", str(cause), cause)
                    kill_process_tree(pid)

        async def read_stream(stream, parts: list[str], callback) -> None:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            try:
                while True:
                    chunk = await stream.receive_some()
                    if not chunk:
                        break
                    text = decoder.decode(chunk)
                    if text:
                        handle_chunk(parts, callback, text)
                final = decoder.decode(b"", True)
                if final:
                    handle_chunk(parts, callback, final)
            except Exception:
                # Stream force-closed by the post-exit grace logic or broken pipe.
                pass

        async def read_streams() -> None:
            try:
                await tonio.spawn(
                    read_stream(process.stdout, stdout_parts, options.on_stdout),
                    read_stream(process.stderr, stderr_parts, options.on_stderr),
                )
            finally:
                readers_done.set()

        tonio.spawn.without_tracking(read_streams())

        exited = tonio.Event()

        async def watchdog() -> None:
            nonlocal timed_out
            await exited.wait(timeout_seconds)
            if not exited.is_set():
                timed_out = True
                kill_process_tree(pid)

        watchdog_join = tonio.spawn(watchdog()) if timeout_seconds is not None else None

        unsubscribe = None
        if options.cancel is not None:
            unsubscribe = options.cancel.on_cancel(lambda _reason: kill_process_tree(pid))

        try:
            exit_code = await process.wait()
        except Exception as error:
            exited.set()
            if watchdog_join is not None:
                await watchdog_join
            if unsubscribe is not None:
                unsubscribe()
            with self._active_child_pids_lock:
                self._active_child_pids.discard(pid)
            cause = to_error(error)
            return err(ExecutionError("spawn_error", str(cause), cause))
        exited.set()
        if watchdog_join is not None:
            await watchdog_join

        # Post-exit stdio grace: detached descendants can keep the inherited
        # pipes open. Give the readers a grace window per burst of data (pi
        # re-arms a 100 ms idle timer on each chunk), then force-close.
        while not readers_done.is_set():
            with state_lock:
                before = activity_count
            await readers_done.wait(EXIT_STDIO_GRACE_SECONDS)
            if readers_done.is_set():
                break
            with state_lock:
                unchanged = activity_count == before
            if unchanged:
                break

        # Release the pipe fds (pi destroys the stdio streams at finalize).
        # When a detached descendant still holds the write ends, this also
        # unblocks any reader parked on them.
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass

        if unsubscribe is not None:
            unsubscribe()
        with self._active_child_pids_lock:
            self._active_child_pids.discard(pid)

        with state_lock:
            captured_callback_error = callback_error
            stdout = "".join(stdout_parts)
            stderr = "".join(stderr_parts)
        if captured_callback_error is not None:
            return err(captured_callback_error)
        if timed_out:
            return err(ExecutionError("timeout", f"timeout:{options.timeout}"))
        if options.cancel is not None and options.cancel.cancelled:
            return err(ExecutionError("aborted", "aborted"))
        # Node reports `null` for signal-killed children and pi maps it to 0.
        return ok(ShellExecResult(stdout=stdout, stderr=stderr, exit_code=max(exit_code, 0)))

    async def cleanup(self) -> None:
        with self._active_child_pids_lock:
            pids = list(self._active_child_pids)
            self._active_child_pids.clear()
        for pid in pids:
            kill_process_tree(pid)


__all__ = ["EXIT_STDIO_GRACE_SECONDS", "LocalExecutionEnv", "ShellConfig", "kill_process_tree"]
