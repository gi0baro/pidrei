"""Mirror of pi agent/test/harness/nodejs-env.test.ts."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import tonio.colored as tonio
from tonio.colored import time as tonio_time

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.types import FileError, ShellExecOptions, ShellExecResult, get_or_throw
from pidrei_agent.harness.utils.shell_output import execute_shell_with_capture
from pidrei_ai.utils.cancel import CancelToken


# Same gate as pidrei's TestBashTool (see TONIO_BUGS.md #7): tonio's
# `Process.wait()` crashes intermittently on the GHA macOS runner, and the
# failure roams across whichever process-spawning test it likes. Applied to
# the tests that reach a real `await process.wait()`; the file-operation and
# pre-spawn-error tests keep running on macOS CI.
SKIP_ON_MACOS_CI = pytest.mark.skipif(
    sys.platform == "darwin" and bool(os.environ.get("CI")),
    reason="TONIO_BUGS #7: tonio Process.wait() crashes intermittently on the GHA macOS runner",
)


def create_temp_dir() -> str:
    return tempfile.mkdtemp(prefix="pidrei-agent-test-")


@pytest.mark.tonio
async def test_reads_writes_lists_and_removes_files_and_directories():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    assert get_or_throw(await env.absolute_path("nested/child")) == os.path.join(root, "nested/child")
    assert get_or_throw(await env.join_path([root, "nested", "child"])) == os.path.join(root, "nested", "child")
    get_or_throw(await env.create_dir("nested/child"))
    get_or_throw(await env.write_file("nested/child/file.txt", "hel"))
    get_or_throw(await env.append_file("nested/child/file.txt", "lo"))
    assert get_or_throw(await env.read_text_file("nested/child/file.txt")) == "hello"
    assert get_or_throw(await env.read_text_lines("nested/child/file.txt", max_lines=1)) == ["hello"]
    assert get_or_throw(await env.read_binary_file("nested/child/file.txt")) == b"hello"

    entries = get_or_throw(await env.list_dir("nested/child"))
    assert len(entries) == 1
    assert entries[0].name == "file.txt"
    assert entries[0].path == os.path.join(root, "nested/child/file.txt")
    assert entries[0].kind == "file"
    assert entries[0].size == 5
    assert isinstance(entries[0].mtime_ms, float)

    assert get_or_throw(await env.exists("nested/child/file.txt")) is True
    get_or_throw(await env.remove("nested/child/file.txt"))
    assert get_or_throw(await env.exists("nested/child/file.txt")) is False


@pytest.mark.tonio
async def test_expands_home_relative_paths_and_file_urls():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    home = str(Path.home())
    assert get_or_throw(await env.absolute_path("~/pidrei-local-env-test")) == os.path.join(
        home, "pidrei-local-env-test"
    )
    file_path = os.path.join(root, "file with spaces.txt")
    assert get_or_throw(await env.absolute_path(Path(file_path).as_uri())) == file_path


@pytest.mark.tonio
async def test_file_info_for_files_directories_and_symlinks_without_following():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.create_dir("dir", recursive=True))
    get_or_throw(await env.write_file("dir/file.txt", "hello"))
    os.symlink(os.path.join(root, "dir/file.txt"), os.path.join(root, "file-link"))
    os.symlink(os.path.join(root, "dir"), os.path.join(root, "dir-link"))

    dir_info = get_or_throw(await env.file_info("dir"))
    assert (dir_info.name, dir_info.path, dir_info.kind) == ("dir", os.path.join(root, "dir"), "directory")
    file_info = get_or_throw(await env.file_info("dir/file.txt"))
    assert (file_info.name, file_info.path, file_info.kind, file_info.size) == (
        "file.txt",
        os.path.join(root, "dir/file.txt"),
        "file",
        5,
    )
    file_link = get_or_throw(await env.file_info("file-link"))
    assert (file_link.name, file_link.path, file_link.kind) == ("file-link", os.path.join(root, "file-link"), "symlink")
    dir_link = get_or_throw(await env.file_info("dir-link"))
    assert (dir_link.name, dir_link.path, dir_link.kind) == ("dir-link", os.path.join(root, "dir-link"), "symlink")
    assert get_or_throw(await env.canonical_path("file-link")) == os.path.realpath(os.path.join(root, "dir/file.txt"))


@pytest.mark.tonio
async def test_lists_symlinks_as_symlinks():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("target.txt", "hello"))
    os.symlink(os.path.join(root, "target.txt"), os.path.join(root, "link.txt"))

    entries = get_or_throw(await env.list_dir("."))
    assert sorted((entry.name, entry.kind) for entry in entries) == [
        ("link.txt", "symlink"),
        ("target.txt", "file"),
    ]


@pytest.mark.tonio
async def test_stops_reading_text_lines_at_the_requested_limit():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("file.txt", "one\ntwo\nthree"))
    assert get_or_throw(await env.read_text_lines("file.txt", max_lines=1)) == ["one"]


@pytest.mark.tonio
async def test_file_error_for_missing_paths_and_exists_false():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    info = await env.file_info("missing.txt")
    assert info.ok is False
    assert isinstance(info.error, FileError)
    assert info.error.code == "not_found"
    assert info.error.path == os.path.join(root, "missing.txt")
    assert get_or_throw(await env.exists("missing.txt")) is False


@pytest.mark.tonio
async def test_file_error_for_listing_non_directories():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("file.txt", "hello"))
    result = await env.list_dir("file.txt")
    assert result.ok is False
    assert isinstance(result.error, FileError)
    assert result.error.code == "not_directory"


@pytest.mark.tonio
async def test_appends_to_new_files_and_creates_parent_directories():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.append_file("new/nested/file.txt", "a"))
    get_or_throw(await env.append_file("new/nested/file.txt", "b"))
    assert get_or_throw(await env.read_text_file("new/nested/file.txt")) == "ab"


@pytest.mark.tonio
async def test_creates_temporary_directories_and_files():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    temp_dir = get_or_throw(await env.create_temp_dir("local-env-test-"))
    assert os.path.exists(temp_dir)
    temp_file = get_or_throw(await env.create_temp_file(prefix="prefix-", suffix=".txt"))
    assert os.path.exists(temp_file)
    assert temp_file.endswith(".txt")


@pytest.mark.tonio
async def test_honors_create_dir_recursive_false_and_remove_recursive_force_options():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    create_result = await env.create_dir("missing/child", recursive=False)
    assert create_result.ok is False
    assert create_result.error.code == "not_found"

    get_or_throw(await env.write_file("dir/child/file.txt", "hello"))
    remove_directory = await env.remove("dir", recursive=False)
    assert remove_directory.ok is False
    get_or_throw(await env.remove("dir", recursive=True))
    assert get_or_throw(await env.exists("dir")) is False

    remove_missing = await env.remove("missing", force=False)
    assert remove_missing.ok is False
    get_or_throw(await env.remove("missing", force=True))


@pytest.mark.tonio
async def test_returns_aborted_results_for_pre_aborted_cancellable_file_operations():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file("file.txt", "hello"))
    token = CancelToken()
    token.cancel()

    results = [
        await env.read_text_file("file.txt", cancel=token),
        await env.read_text_lines("file.txt", cancel=token),
        await env.read_binary_file("file.txt", cancel=token),
        await env.write_file("other.txt", "hello", cancel=token),
        await env.list_dir(".", cancel=token),
    ]
    for result in results:
        assert result.ok is False
        assert result.error.code == "aborted"


@pytest.mark.tonio
async def test_cleanup_is_best_effort():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    assert await env.cleanup() is None


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_executes_commands_in_cwd_with_env_overrides():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    result = get_or_throw(
        await env.exec(
            'printf \'%s:%s\' "$PWD" "$LOCAL_ENV_TEST"',
            ShellExecOptions(env={"LOCAL_ENV_TEST": "ok"}),
        )
    )
    assert result == ShellExecResult(stdout=f"{os.path.realpath(root)}:ok", stderr="", exit_code=0)


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_can_replace_rather_than_inherit_the_default_shell_environment():
    root = create_temp_dir()
    inherited_key = "PIDREI_LOCAL_ENV_INHERITED_TEST"
    configured_key = "PIDREI_LOCAL_ENV_CONFIGURED_TEST"
    explicit_key = "PIDREI_LOCAL_ENV_EXPLICIT_TEST"
    previous_inherited = os.environ.get(inherited_key)
    os.environ[inherited_key] = "host"
    try:
        env = LocalExecutionEnv(cwd=root, shell_env={configured_key: "configured"})
        result = get_or_throw(
            await env.exec(
                f'printf \'%s:%s:%s\' "${{{inherited_key}-}}" "${{{configured_key}-}}" "${{{explicit_key}-}}"',
                ShellExecOptions(inherit_env=False, env={explicit_key: "explicit"}),
            )
        )
        assert result.stdout == "::explicit"
    finally:
        if previous_inherited is None:
            os.environ.pop(inherited_key, None)
        else:
            os.environ[inherited_key] = previous_inherited


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_cleanup_terminates_active_shell_processes():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    execution = tonio.spawn(env.exec("touch started; sleep 60"))
    for _ in range(100):
        if get_or_throw(await env.exists("started")):
            break
        await tonio.sleep(0.01)
    assert get_or_throw(await env.exists("started")) is True
    await env.cleanup()
    result, completed = await tonio_time.timeout(execution, 3.0)
    assert completed is True
    assert result.ok is True


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_streams_stdout_and_stderr_chunks():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    result = get_or_throw(
        await env.exec(
            "printf out; printf err >&2",
            ShellExecOptions(on_stdout=stdout_chunks.append, on_stderr=stderr_chunks.append),
        )
    )
    assert result == ShellExecResult(stdout="out", stderr="err", exit_code=0)
    assert "".join(stdout_chunks) == "out"
    assert "".join(stderr_chunks) == "err"


@pytest.mark.tonio
async def test_reports_a_missing_working_directory_before_spawning():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=os.path.join(root, "missing"))
    result = await env.exec("printf ok")

    assert result.ok is False
    assert result.error.code == "spawn_error"
    assert "Working directory does not exist" in result.error.message


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_returns_non_zero_command_exit_codes_as_successful_execution_results():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    result = get_or_throw(await env.exec("exit 7"))
    assert result == ShellExecResult(stdout="", stderr="", exit_code=7)


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_returns_timeout_errors_for_commands_exceeding_the_timeout():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    result = await env.exec("sleep 5", ShellExecOptions(timeout=0.01))
    assert result.ok is False
    assert result.error.code == "timeout"


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_returns_callback_errors_from_exec_stream_handlers():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)

    def failing_callback(_chunk: str) -> None:
        raise Exception("callback failed")

    result = await env.exec("printf out", ShellExecOptions(on_stdout=failing_callback))
    assert result.ok is False
    assert result.error.code == "callback_error"
    assert result.error.message == "callback failed"


@pytest.mark.tonio
async def test_returns_shell_unavailable_and_spawn_errors():
    root = create_temp_dir()
    missing_shell_env = LocalExecutionEnv(cwd=root, shell_path=os.path.join(root, "missing-shell"))
    missing_shell = await missing_shell_env.exec("printf ok")
    assert missing_shell.ok is False
    assert missing_shell.error.code == "shell_unavailable"

    shell_path = os.path.join(root, "not-executable-shell")
    env = LocalExecutionEnv(cwd=root)
    get_or_throw(await env.write_file(shell_path, "not executable"))
    spawn_error_env = LocalExecutionEnv(cwd=root, shell_path=shell_path)
    spawn_error = await spawn_error_env.exec("printf ok")
    assert spawn_error.ok is False
    assert spawn_error.error.code == "spawn_error"


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_returns_an_aborted_result_for_aborted_commands():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    token = CancelToken()
    execution = tonio.spawn(env.exec("sleep 5", ShellExecOptions(cancel=token)))
    await tonio.sleep(0.05)
    token.cancel()
    result = await execution
    assert result.ok is False
    assert result.error.code == "aborted"


@SKIP_ON_MACOS_CI
@pytest.mark.tonio
async def test_captures_large_shell_output_to_a_full_output_file_through_the_execution_env():
    root = create_temp_dir()
    env = LocalExecutionEnv(cwd=root)
    result = get_or_throw(await execute_shell_with_capture(env, "yes line | head -n 15000"))
    assert result.truncated is True
    assert result.full_output_path is not None
    full_output = get_or_throw(await env.read_text_file(result.full_output_path))
    assert len(full_output.split("\n")) > 10000
    assert len(result.output) < len(full_output)
