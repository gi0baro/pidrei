"""Mirror of pi coding-agent test/tools.test.ts.

Adaptations: the win32-only shell tests (stdin transport, legacy WSL bash)
are not ported; the getShellConfig spy tests patch the bash module binding
in-test; grep/find tests run against the system rg/fd (skipped when absent);
the unified patch is applied with the same hand-rolled applier used by the
agent-package mirror.
"""

import base64
import os
import shutil
import struct
import sys

import pytest
import tonio.colored as tonio
from tonio.colored import time as tonio_time

import pidrei.core.tools.bash as bash_module
from pidrei.core.bash_executor import execute_bash_with_operations
from pidrei.core.tools import (
    create_bash_tool,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_local_bash_operations,
    create_ls_tool,
    create_read_tool,
    create_write_tool,
)
from pidrei.core.tools.bash import BASH_TOOL_CONFIG, BashExecResult, ShellToolConfig, create_shell_tool_definition
from pidrei.core.tools.edit_diff import EditDiffError, compute_edits_diff
from pidrei.core.tools.renderers.bash import _format_shell_call
from pidrei.modes.interactive.theme import init_theme_sync
from pidrei.utils.ansi import strip_ansi
from pidrei_ai.utils.cancel import CancelToken


HAS_RG = shutil.which("rg") is not None
HAS_FD = shutil.which("fd") is not None


def get_text_output(result) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "type", None) == "text")


def apply_patch(original: str, patch: str) -> str:
    """Minimal unified-diff applier (mirror of the agent-package test helper)."""
    lines = original.split("\n")
    result: list[str] = []
    cursor = 0
    for line in patch.split("\n"):
        if line.startswith(("---", "+++")) or not line:
            continue
        if line.startswith("@@"):
            header = line.split("@@")[1].strip()
            old_start = int(header.split(" ")[0].lstrip("-").split(",")[0])
            while cursor < old_start - 1:
                result.append(lines[cursor])
                cursor += 1
            continue
        if line.startswith("+"):
            result.append(line[1:])
        elif line.startswith("-"):
            cursor += 1
        elif line.startswith(" "):
            result.append(line[1:])
            cursor += 1
        elif line == "\\ No newline at end of file":
            continue
    result.extend(lines[cursor:])
    return "\n".join(result)


def create_tiny_bmp_1x1_red_24bpp() -> bytes:
    buffer = bytearray(58)
    buffer[0:2] = b"BM"
    struct.pack_into("<I", buffer, 2, len(buffer))
    struct.pack_into("<I", buffer, 10, 54)
    struct.pack_into("<I", buffer, 14, 40)
    struct.pack_into("<i", buffer, 18, 1)
    struct.pack_into("<i", buffer, 22, 1)
    struct.pack_into("<H", buffer, 26, 1)
    struct.pack_into("<H", buffer, 28, 24)
    struct.pack_into("<I", buffer, 30, 0)
    struct.pack_into("<I", buffer, 34, 4)
    buffer[56] = 0xFF
    return bytes(buffer)


PNG_1X1_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="


class TestReadTool:
    @pytest.mark.tonio
    async def test_reads_file_contents_that_fit_within_limits(self, tmp_path):
        test_file = tmp_path / "test.txt"
        content = "Hello, world!\nLine 2\nLine 3"
        test_file.write_text(content)
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-1", {"path": str(test_file)})

        assert get_text_output(result) == content
        assert "Use offset=" not in get_text_output(result)
        assert result.details is None

    @pytest.mark.tonio
    async def test_handles_non_existent_files(self, tmp_path):
        read_tool = create_read_tool(str(tmp_path))
        with pytest.raises(Exception, match="No such file|not found"):
            await read_tool.execute("test-call-2", {"path": str(tmp_path / "nonexistent.txt")})

    @pytest.mark.tonio
    async def test_truncates_files_exceeding_line_limit(self, tmp_path):
        test_file = tmp_path / "large.txt"
        test_file.write_text("\n".join(f"Line {i + 1}" for i in range(2500)))
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-3", {"path": str(test_file)})
        output = get_text_output(result)

        assert "Line 1" in output
        assert "Line 2000" in output
        assert "Line 2001" not in output
        assert "[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in output

    @pytest.mark.tonio
    async def test_truncates_when_byte_limit_exceeded(self, tmp_path):
        import re

        test_file = tmp_path / "large-bytes.txt"
        test_file.write_text("\n".join(f"Line {i + 1}: {'x' * 200}" for i in range(500)))
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-4", {"path": str(test_file)})
        output = get_text_output(result)

        assert "Line 1:" in output
        assert re.search(r"\[Showing lines 1-\d+ of 500 \(.* limit\)\. Use offset=\d+ to continue\.\]", output)

    @pytest.mark.tonio
    async def test_handles_offset_parameter(self, tmp_path):
        test_file = tmp_path / "offset-test.txt"
        test_file.write_text("\n".join(f"Line {i + 1}" for i in range(100)))
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-5", {"path": str(test_file), "offset": 51})
        output = get_text_output(result)

        assert "Line 50" not in output
        assert "Line 51" in output
        assert "Line 100" in output
        assert "Use offset=" not in output

    @pytest.mark.tonio
    async def test_handles_limit_parameter(self, tmp_path):
        test_file = tmp_path / "limit-test.txt"
        test_file.write_text("\n".join(f"Line {i + 1}" for i in range(100)))
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-6", {"path": str(test_file), "limit": 10})
        output = get_text_output(result)

        assert "Line 1" in output
        assert "Line 10" in output
        assert "Line 11" not in output
        assert "[90 more lines in file. Use offset=11 to continue.]" in output

    @pytest.mark.tonio
    async def test_handles_offset_and_limit_together(self, tmp_path):
        test_file = tmp_path / "offset-limit-test.txt"
        test_file.write_text("\n".join(f"Line {i + 1}" for i in range(100)))
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-7", {"path": str(test_file), "offset": 41, "limit": 20})
        output = get_text_output(result)

        assert "Line 40" not in output
        assert "Line 41" in output
        assert "Line 60" in output
        assert "Line 61" not in output
        assert "[40 more lines in file. Use offset=61 to continue.]" in output

    @pytest.mark.tonio
    async def test_shows_error_when_offset_is_beyond_file_length(self, tmp_path):
        test_file = tmp_path / "short.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3")
        read_tool = create_read_tool(str(tmp_path))

        with pytest.raises(Exception, match=r"Offset 100 is beyond end of file \(3 lines total\)"):
            await read_tool.execute("test-call-8", {"path": str(test_file), "offset": 100})

    @pytest.mark.tonio
    async def test_includes_truncation_details_when_truncated(self, tmp_path):
        test_file = tmp_path / "large-file.txt"
        test_file.write_text("\n".join(f"Line {i + 1}" for i in range(2500)))
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-9", {"path": str(test_file)})

        assert result.details is not None
        assert result.details.truncation is not None
        assert result.details.truncation.truncated is True
        assert result.details.truncation.truncated_by == "lines"
        assert result.details.truncation.total_lines == 2500
        assert result.details.truncation.output_lines == 2000

    @pytest.mark.tonio
    async def test_detects_image_mime_type_from_file_magic_not_extension(self, tmp_path):
        test_file = tmp_path / "image.txt"
        test_file.write_bytes(base64.b64decode(PNG_1X1_BASE64))
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-img-1", {"path": str(test_file)})

        assert result.content[0].type == "text"
        assert "Read image file [image/png]" in get_text_output(result)

        image_block = next((block for block in result.content if block.type == "image"), None)
        assert image_block is not None
        assert image_block.mime_type == "image/png"
        assert isinstance(image_block.data, str)
        assert len(image_block.data) > 0

    @pytest.mark.tonio
    async def test_reads_bmp_files_from_disk_as_png_image_attachments(self, tmp_path):
        test_file = tmp_path / "image.bmp"
        test_file.write_bytes(create_tiny_bmp_1x1_red_24bpp())
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-img-bmp", {"path": str(test_file)})

        assert result.content[0].type == "text"
        assert "Read image file [image/png]" in get_text_output(result)
        assert "[Image converted from image/bmp to image/png.]" in get_text_output(result)

        image_block = next((block for block in result.content if block.type == "image"), None)
        assert image_block is not None
        assert image_block.mime_type == "image/png"
        assert base64.b64decode(image_block.data)[0] == 0x89

    @pytest.mark.tonio
    async def test_treats_files_with_image_extension_but_non_image_content_as_text(self, tmp_path):
        test_file = tmp_path / "not-an-image.png"
        test_file.write_text("definitely not a png")
        read_tool = create_read_tool(str(tmp_path))

        result = await read_tool.execute("test-call-img-2", {"path": str(test_file)})
        output = get_text_output(result)

        assert "definitely not a png" in output
        assert not any(block.type == "image" for block in result.content)


class TestWriteTool:
    @pytest.mark.tonio
    async def test_writes_file_contents(self, tmp_path):
        test_file = tmp_path / "write-test.txt"
        write_tool = create_write_tool(str(tmp_path))

        result = await write_tool.execute("test-call-3", {"path": str(test_file), "content": "Test content"})

        assert "Successfully wrote" in get_text_output(result)
        assert str(test_file) in get_text_output(result)
        assert result.details is None

    @pytest.mark.tonio
    async def test_creates_parent_directories(self, tmp_path):
        test_file = tmp_path / "nested" / "dir" / "test.txt"
        write_tool = create_write_tool(str(tmp_path))

        result = await write_tool.execute("test-call-4", {"path": str(test_file), "content": "Nested content"})

        assert "Successfully wrote" in get_text_output(result)


class TestEditTool:
    @pytest.mark.tonio
    async def test_replaces_text_in_file(self, tmp_path):
        test_file = tmp_path / "edit-test.txt"
        original_content = "Hello, world!"
        test_file.write_text(original_content)
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-call-5", {"path": str(test_file), "edits": [{"oldText": "world", "newText": "testing"}]}
        )

        assert "Successfully replaced" in get_text_output(result)
        assert result.details is not None
        assert isinstance(result.details.diff, str)
        assert "testing" in result.details.diff
        assert "--- " in result.details.patch
        assert "+++ " in result.details.patch
        assert "@@" in result.details.patch
        assert "-Hello, world!" in result.details.patch
        assert "+Hello, testing!" in result.details.patch
        assert apply_patch(original_content, result.details.patch) == "Hello, testing!"

    @pytest.mark.tonio
    async def test_fails_if_text_not_found(self, tmp_path):
        test_file = tmp_path / "edit-test.txt"
        test_file.write_text("Hello, world!")
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception, match="Could not find the exact text"):
            await edit_tool.execute(
                "test-call-6", {"path": str(test_file), "edits": [{"oldText": "nonexistent", "newText": "testing"}]}
            )

    @pytest.mark.tonio
    async def test_includes_enoent_when_the_edit_target_does_not_exist(self, tmp_path):
        missing_file = tmp_path / "missing.txt"
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception) as excinfo:
            await edit_tool.execute(
                "test-call-6b", {"path": str(missing_file), "edits": [{"oldText": "hello", "newText": "world"}]}
            )
        assert str(excinfo.value) == f"Could not edit file: {missing_file}. Error code: ENOENT."

    @pytest.mark.tonio
    async def test_fails_if_text_appears_multiple_times(self, tmp_path):
        test_file = tmp_path / "edit-test.txt"
        test_file.write_text("foo foo foo")
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception, match="Found 3 occurrences"):
            await edit_tool.execute(
                "test-call-7", {"path": str(test_file), "edits": [{"oldText": "foo", "newText": "bar"}]}
            )

    @pytest.mark.tonio
    async def test_replaces_multiple_disjoint_regions_in_one_call(self, tmp_path):
        test_file = tmp_path / "edit-multi.txt"
        test_file.write_text("alpha\nbeta\ngamma\ndelta\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-call-8",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "alpha\n", "newText": "ALPHA\n"},
                    {"oldText": "gamma\n", "newText": "GAMMA\n"},
                ],
            },
        )

        assert "Successfully replaced 2 block(s)" in get_text_output(result)
        assert test_file.read_text() == "ALPHA\nbeta\nGAMMA\ndelta\n"
        assert "ALPHA" in result.details.diff
        assert "GAMMA" in result.details.diff

    @pytest.mark.tonio
    async def test_collapses_large_unchanged_gaps_in_multi_edit_diffs(self, tmp_path):
        test_file = tmp_path / "edit-multi-large-gap.txt"
        lines = [f"line {str(i + 1).zfill(3)}" for i in range(600)]
        test_file.write_text("\n".join(lines) + "\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-call-8b",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "line 100\n", "newText": "LINE 100\n"},
                    {"oldText": "line 300\n", "newText": "LINE 300\n"},
                    {"oldText": "line 500\n", "newText": "LINE 500\n"},
                ],
            },
        )

        diff = result.details.diff
        assert "LINE 100" in diff
        assert "LINE 300" in diff
        assert "LINE 500" in diff
        assert "..." in diff
        assert "line 250" not in diff
        assert len(diff.split("\n")) < 50

    @pytest.mark.tonio
    async def test_matches_edits_against_the_original_file_not_incrementally(self, tmp_path):
        test_file = tmp_path / "edit-multi-original.txt"
        test_file.write_text("foo\nbar\nbaz\n")
        edit_tool = create_edit_tool(str(tmp_path))

        await edit_tool.execute(
            "test-call-9",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "foo\n", "newText": "foo bar\n"},
                    {"oldText": "bar\n", "newText": "BAR\n"},
                ],
            },
        )

        assert test_file.read_text() == "foo bar\nBAR\nbaz\n"

    @pytest.mark.tonio
    async def test_fails_when_edits_is_empty(self, tmp_path):
        test_file = tmp_path / "edit-empty-edits.txt"
        test_file.write_text("hello\nworld\n")
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception, match="edits must contain at least one replacement"):
            await edit_tool.execute("test-call-11", {"path": str(test_file), "edits": []})

    @pytest.mark.tonio
    async def test_fails_when_multi_edit_regions_overlap(self, tmp_path):
        test_file = tmp_path / "edit-overlap.txt"
        test_file.write_text("one\ntwo\nthree\n")
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception, match="overlap"):
            await edit_tool.execute(
                "test-call-12",
                {
                    "path": str(test_file),
                    "edits": [
                        {"oldText": "one\ntwo\n", "newText": "ONE\nTWO\n"},
                        {"oldText": "two\nthree\n", "newText": "TWO\nTHREE\n"},
                    ],
                },
            )

    @pytest.mark.tonio
    async def test_does_not_partially_apply_edits_when_one_edit_fails(self, tmp_path):
        test_file = tmp_path / "edit-no-partial.txt"
        original_content = "alpha\nbeta\ngamma\n"
        test_file.write_text(original_content)
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception, match="Could not find"):
            await edit_tool.execute(
                "test-call-13",
                {
                    "path": str(test_file),
                    "edits": [
                        {"oldText": "alpha\n", "newText": "ALPHA\n"},
                        {"oldText": "missing\n", "newText": "MISSING\n"},
                    ],
                },
            )

        assert test_file.read_text() == original_content

    @pytest.mark.tonio
    async def test_includes_eacces_for_read_only_files(self, tmp_path):
        test_file = tmp_path / "edit-readonly.txt"
        test_file.write_text("hello\n")
        os.chmod(test_file, 0o444)
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception) as excinfo:
            await edit_tool.execute(
                "test-call-14", {"path": str(test_file), "edits": [{"oldText": "hello", "newText": "world"}]}
            )
        assert str(excinfo.value) == f"Could not edit file: {test_file}. Error code: EACCES."

    @pytest.mark.tonio
    async def test_includes_the_original_error_message_for_unknown_edit_access_errors(self, tmp_path):
        class FailingOperations:
            async def access(self, _absolute_path):
                raise Exception("disk offline")

            async def read_file(self, _absolute_path):
                return b"hello\n"

            async def write_file(self, _absolute_path, _content):
                pass

        generic_failure_tool = create_edit_tool(str(tmp_path), operations=FailingOperations())

        with pytest.raises(Exception) as excinfo:
            await generic_failure_tool.execute(
                "test-call-16", {"path": "broken.txt", "edits": [{"oldText": "hello", "newText": "world"}]}
            )
        assert str(excinfo.value) == "Could not edit file: broken.txt. Exception: disk offline."

    @pytest.mark.tonio
    async def test_includes_enoent_in_diff_preview_for_missing_files(self, tmp_path):
        from pidrei.core.tools.edit_diff import Edit

        missing_file = tmp_path / "missing-preview.txt"
        result = await compute_edits_diff(str(missing_file), [Edit(old_text="hello", new_text="world")], str(tmp_path))

        assert isinstance(result, EditDiffError)
        assert result.error == f"Could not edit file: {missing_file}. Error code: ENOENT."

    @pytest.mark.tonio
    async def test_includes_eacces_in_diff_preview_for_unreadable_files(self, tmp_path):
        from pidrei.core.tools.edit_diff import Edit

        unreadable_file = tmp_path / "unreadable-preview.txt"
        unreadable_file.write_text("hello\n")
        os.chmod(unreadable_file, 0o222)

        result = await compute_edits_diff(
            str(unreadable_file), [Edit(old_text="hello", new_text="world")], str(tmp_path)
        )

        assert isinstance(result, EditDiffError)
        assert result.error == f"Could not edit file: {unreadable_file}. Error code: EACCES."


class ChattyFailOperations:
    def __init__(self, error_message: str):
        self._error_message = error_message

    async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
        for i in range(1, 3001):
            on_data(f"{i}\n".encode())
        raise Exception(self._error_message)


class TestBashTool:
    @pytest.mark.tonio
    async def test_executes_simple_commands(self, tmp_path):
        bash_tool = create_bash_tool(str(tmp_path))
        result = await bash_tool.execute("test-call-8", {"command": "echo 'test output'"})

        assert "test output" in get_text_output(result)
        assert result.details is None

    @pytest.mark.tonio
    async def test_handles_command_errors(self, tmp_path):
        bash_tool = create_bash_tool(str(tmp_path))
        with pytest.raises(Exception, match="Command failed|code 1"):
            await bash_tool.execute("test-call-9", {"command": "exit 1"})

    @pytest.mark.tonio
    async def test_respects_timeout(self, tmp_path):
        bash_tool = create_bash_tool(str(tmp_path))
        with pytest.raises(Exception, match="timed out"):
            await bash_tool.execute("test-call-10", {"command": "sleep 5", "timeout": 0.05})

    @pytest.mark.tonio
    async def test_includes_full_output_path_for_truncated_timeout_and_abort_errors(self, tmp_path):
        import re

        for error_message, expected in (
            ("timeout:5", "Command timed out after 5 seconds"),
            ("aborted", "Command aborted"),
        ):
            bash = create_bash_tool(str(tmp_path), operations=ChattyFailOperations(error_message))

            with pytest.raises(Exception) as excinfo:
                await bash.execute(f"test-call-{error_message}", {"command": "chatty-fail"})

            message = str(excinfo.value)
            assert expected in message
            assert re.search(r"\[Showing lines \d+-\d+ of \d+\. Full output: ", message)
            assert "Full output: None" not in message
            full_output_path = re.search(r"Full output: ([^\]\n]+)", message).group(1)
            assert os.path.exists(full_output_path)
            with open(full_output_path, encoding="utf-8") as f:
                full_output = f.read()
            assert "1\n2\n3" in full_output
            assert "2998\n2999\n3000" in full_output

    @pytest.mark.tonio
    async def test_throws_error_when_cwd_does_not_exist(self):
        bash_tool = create_bash_tool("/this/directory/definitely/does/not/exist/12345")
        with pytest.raises(Exception, match="Working directory does not exist"):
            await bash_tool.execute("test-call-11", {"command": "echo test"})

    @pytest.mark.tonio
    async def test_handles_process_spawn_errors(self, tmp_path):
        from pidrei.utils.shell import ShellConfig

        original = bash_module.get_shell_config
        bash_module.get_shell_config = lambda _custom=None: ShellConfig(  # noqa: S604
            shell="/nonexistent-shell-path-xyz123", args=["-c"]
        )
        try:
            bash_with_bad_shell = create_bash_tool(str(tmp_path))
            with pytest.raises(Exception, match="No such file|ENOENT"):
                await bash_with_bad_shell.execute("test-call-12", {"command": "echo test"})
        finally:
            bash_module.get_shell_config = original

    @pytest.mark.tonio
    async def test_passes_shell_path_through_to_shell_resolution(self, tmp_path):
        ops = create_local_bash_operations(shell_path="/custom/bash")
        with pytest.raises(Exception, match="Custom shell path not found: /custom/bash"):
            await ops.exec("echo test", str(tmp_path), on_data=lambda _data: None)

    @pytest.mark.tonio
    async def test_prepends_command_prefix_when_configured(self, tmp_path):
        bash_with_prefix = create_bash_tool(str(tmp_path), command_prefix="export TEST_VAR=hello")
        result = await bash_with_prefix.execute("test-prefix-1", {"command": "echo $TEST_VAR"})
        assert get_text_output(result).strip() == "hello"

    @pytest.mark.tonio
    async def test_includes_output_from_both_prefix_and_command(self, tmp_path):
        bash_with_prefix = create_bash_tool(str(tmp_path), command_prefix="echo prefix-output")
        result = await bash_with_prefix.execute("test-prefix-2", {"command": "echo command-output"})
        assert get_text_output(result).strip() == "prefix-output\ncommand-output"

    @pytest.mark.tonio
    async def test_works_without_command_prefix(self, tmp_path):
        bash_without_prefix = create_bash_tool(str(tmp_path))
        result = await bash_without_prefix.execute("test-prefix-3", {"command": "echo no-prefix"})
        assert get_text_output(result).strip() == "no-prefix"

    @pytest.mark.tonio
    async def test_coalesces_streaming_updates_for_chatty_output(self, tmp_path):
        class ChattyOperations:
            async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
                for i in range(5000):
                    on_data(f"line {i}\n".encode())
                return BashExecResult(exit_code=0)

        updates = []
        bash = create_bash_tool(str(tmp_path), operations=ChattyOperations())

        result = await bash.execute(
            "test-call-chatty-updates", {"command": "chatty"}, None, lambda update: updates.append(update)
        )

        assert len(updates) < 25
        assert "line 4999" in get_text_output(result)

    @pytest.mark.tonio
    async def test_does_not_count_a_trailing_newline_as_an_extra_truncated_line(self, tmp_path):
        import re

        class ManyLinesOperations:
            async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
                for i in range(1, 4001):
                    on_data(f"line-{str(i).zfill(4)}\n".encode())
                return BashExecResult(exit_code=0)

        bash = create_bash_tool(str(tmp_path), operations=ManyLinesOperations())

        result = await bash.execute("test-call-trailing-newline-line-count", {"command": "many-lines"})
        output = get_text_output(result)

        assert result.details.truncation.total_lines == 4000
        assert result.details.truncation.output_lines == 2000
        assert "line-2001" in output
        assert "line-4000" in output
        assert re.search(r"\[Showing lines 2001-4000 of 4000\. Full output: ", output)
        assert "4001" not in output

    @pytest.mark.tonio
    async def test_decodes_utf8_characters_split_across_output_chunks(self, tmp_path):
        euro = "€\n".encode()

        class SplitUtf8Operations:
            async def exec(self, _command, _cwd, *, on_data, cancel=None, timeout=None, env=None):
                on_data(euro[:1])
                on_data(euro[1:])
                return BashExecResult(exit_code=0)

        bash = create_bash_tool(str(tmp_path), operations=SplitUtf8Operations())

        result = await bash.execute("test-call-split-utf8", {"command": "split-utf8"})

        assert get_text_output(result).strip() == "€"

    @pytest.mark.tonio
    async def test_exposes_local_bash_operations_for_extension_reuse(self, tmp_path):
        ops = create_local_bash_operations()
        chunks: list[bytes] = []

        result = await ops.exec(
            "echo $TEST_LOCAL_BASH_OPS",
            str(tmp_path),
            on_data=lambda data: chunks.append(data),
            env={**os.environ, "TEST_LOCAL_BASH_OPS": "from-local-ops"},
        )

        assert result.exit_code == 0
        assert b"".join(chunks).decode().strip() == "from-local-ops"

    @pytest.mark.tonio
    async def test_preserves_execute_bash_sanitization_when_using_local_bash_operations(self):
        result = await execute_bash_with_operations(
            "printf '\\033[31mred\\033[0m\\r\\n'", os.getcwd(), create_local_bash_operations()
        )

        assert result.exit_code == 0
        assert result.output == "red\n"

    @pytest.mark.tonio
    async def test_persists_full_output_when_truncation_happens_by_line_count_only(self, tmp_path):
        import re

        bash = create_bash_tool(str(tmp_path))
        result = await bash.execute("test-call-line-truncation", {"command": "seq 3000"})
        output = get_text_output(result)
        full_output_path = result.details.full_output_path

        assert result.details.truncation.truncated is True
        assert result.details.truncation.truncated_by == "lines"
        assert full_output_path is not None
        assert re.search(r"\[Showing lines \d+-\d+ of \d+\. Full output: ", output)
        assert "Full output: None" not in output

        for _ in range(20):
            if os.path.exists(full_output_path):
                break
            await tonio_time.sleep(0.01)

        assert os.path.exists(full_output_path)
        with open(full_output_path, encoding="utf-8") as f:
            full_output = f.read()
        assert "1\n2\n3" in full_output
        assert "2998\n2999\n3000" in full_output

    @pytest.mark.tonio
    async def test_execute_bash_persists_full_output_when_truncated_by_line_count(self):
        result = await execute_bash_with_operations("seq 3000", os.getcwd(), create_local_bash_operations())
        full_output_path = result.full_output_path

        assert result.truncated is True
        assert full_output_path is not None
        assert os.path.exists(full_output_path)
        with open(full_output_path, encoding="utf-8") as f:
            full_output = f.read()
        assert "1\n2\n3" in full_output
        assert "2998\n2999\n3000" in full_output

    @pytest.mark.tonio
    async def test_abort_kills_running_command(self, tmp_path):
        bash = create_bash_tool(str(tmp_path))
        cancel = CancelToken()

        async def abort_later():
            await tonio_time.sleep(0.05)
            cancel.cancel()

        tonio.spawn.without_tracking(abort_later())

        with pytest.raises(Exception, match="Command aborted"):
            await bash.execute("test-call-abort", {"command": "sleep 5"}, cancel)

    def test_shell_tool_config_drives_the_shared_definition(self, tmp_path):
        """pi shares one implementation between `bash` and its Windows-only
        `powershell` tool; powershell is dropped surface (POSIX-only), so the
        seam is exercised with a config of its own instead."""
        config = ShellToolConfig(
            name="shellish",
            label="Shellish",
            shell_name="shellish",
            prompt=">",
            prompt_snippet="Run shellish commands",
            prompt_guidelines=("Prefer shellish.",),
            temp_file_prefix="pidrei-shellish",
        )
        init_theme_sync("dark")
        definition = create_shell_tool_definition(str(tmp_path), config)

        assert definition.name == "shellish"
        assert definition.label == "Shellish"
        assert definition.description.startswith("Execute a shellish command in the current working directory.")
        assert definition.prompt_snippet == "Run shellish commands"
        assert definition.prompt_guidelines == ["Prefer shellish."]
        without_env = create_shell_tool_definition(str(tmp_path), config, expose_session_environment=False)
        assert without_env.prompt_guidelines is None
        assert strip_ansi(_format_shell_call({"command": "ls"}, config.prompt)) == "> ls"
        assert strip_ansi(_format_shell_call({"command": "ls"}, BASH_TOOL_CONFIG.prompt)) == "$ ls"


@pytest.mark.skipif(not HAS_RG, reason="ripgrep not installed")
class TestGrepTool:
    @pytest.mark.tonio
    async def test_includes_filename_when_searching_a_single_file(self, tmp_path):
        test_file = tmp_path / "example.txt"
        test_file.write_text("first line\nmatch line\nlast line")
        grep_tool = create_grep_tool(str(tmp_path))

        result = await grep_tool.execute("test-call-11", {"pattern": "match", "path": str(test_file)})

        assert "example.txt:2: match line" in get_text_output(result)

    @pytest.mark.tonio
    async def test_respects_global_limit_and_includes_context_lines(self, tmp_path):
        test_file = tmp_path / "context.txt"
        test_file.write_text("before\nmatch one\nafter\nmiddle\nmatch two\nafter two")
        grep_tool = create_grep_tool(str(tmp_path))

        result = await grep_tool.execute(
            "test-call-12", {"pattern": "match", "path": str(test_file), "limit": 1, "context": 1}
        )

        output = get_text_output(result)
        assert "context.txt-1- before" in output
        assert "context.txt:2: match one" in output
        assert "context.txt-3- after" in output
        assert "[1 matches limit reached. Use limit=2 for more, or refine pattern]" in output
        assert "match two" not in output

    @pytest.mark.tonio
    async def test_treats_flag_like_patterns_as_search_text(self, tmp_path):
        marker = tmp_path / "grep-injection-marker"
        payload = tmp_path / "payload.sh"
        test_file = tmp_path / "target.txt"
        payload.write_text(f'#!/bin/sh\necho executed > {marker}\ncat "$1"\n')
        os.chmod(payload, 0o755)
        test_file.write_text("target\n")
        grep_tool = create_grep_tool(str(tmp_path))

        result = await grep_tool.execute(
            "test-call-grep-injection", {"pattern": f"--pre={payload}", "path": str(tmp_path)}
        )

        assert "No matches found" in get_text_output(result)
        assert not marker.exists()


@pytest.mark.tonio
async def test_streaming_lines_kills_the_process_at_the_limit():
    # An unbounded producer: only killing it once `on_line` says "enough"
    # (pi's killedDueToLimit) lets this return instead of reading to EOF.
    from pidrei.core.tools.grep import _run_streaming_lines

    seen: list[str] = []

    def on_line(line: str) -> bool:
        seen.append(line)
        return len(seen) >= 3

    script = (
        "import sys\nn = 0\nwhile True:\n    n += 1\n    sys.stdout.write(f'line {n}\\n')\n    sys.stdout.flush()\n"
    )
    _, completed = await tonio_time.timeout(_run_streaming_lines([sys.executable, "-c", script], None, on_line), 10)

    assert completed
    assert seen[:3] == ["line 1", "line 2", "line 3"]


@pytest.mark.skipif(not HAS_FD, reason="fd not installed")
class TestFindTool:
    @pytest.mark.tonio
    async def test_includes_hidden_files_that_are_not_gitignored(self, tmp_path):
        hidden_dir = tmp_path / ".secret"
        hidden_dir.mkdir()
        (hidden_dir / "hidden.txt").write_text("hidden")
        (tmp_path / "visible.txt").write_text("visible")
        find_tool = create_find_tool(str(tmp_path))

        result = await find_tool.execute("test-call-13", {"pattern": "**/*.txt", "path": str(tmp_path)})

        output_lines = [line.strip() for line in get_text_output(result).split("\n") if line.strip()]
        assert "visible.txt" in output_lines
        assert ".secret/hidden.txt" in output_lines

    @pytest.mark.tonio
    async def test_respects_gitignore(self, tmp_path):
        (tmp_path / ".gitignore").write_text("ignored.txt\n")
        (tmp_path / "ignored.txt").write_text("ignored")
        (tmp_path / "kept.txt").write_text("kept")
        find_tool = create_find_tool(str(tmp_path))

        result = await find_tool.execute("test-call-14", {"pattern": "**/*.txt", "path": str(tmp_path)})

        output = get_text_output(result)
        assert "kept.txt" in output
        assert "ignored.txt" not in output

    @pytest.mark.tonio
    async def test_surfaces_fd_glob_parse_errors(self, tmp_path):
        find_tool = create_find_tool(str(tmp_path))
        with pytest.raises(Exception, match="error parsing glob|fd exited with code 1|fd error"):
            await find_tool.execute("test-call-15", {"pattern": "[", "path": str(tmp_path)})

    @pytest.mark.tonio
    async def test_treats_flag_like_patterns_as_search_text(self, tmp_path):
        find_tool = create_find_tool(str(tmp_path))
        result = await find_tool.execute("test-call-find-flag-pattern", {"pattern": "--help", "path": str(tmp_path)})
        assert "No files found matching pattern" in get_text_output(result)


class TestLsTool:
    @pytest.mark.tonio
    async def test_lists_dotfiles_and_directories(self, tmp_path):
        (tmp_path / ".hidden-file").write_text("secret")
        (tmp_path / ".hidden-dir").mkdir()
        ls_tool = create_ls_tool(str(tmp_path))

        result = await ls_tool.execute("test-call-15", {"path": str(tmp_path)})
        output = get_text_output(result)

        assert ".hidden-file" in output
        assert ".hidden-dir/" in output


class TestEditToolFuzzyMatching:
    @pytest.mark.tonio
    async def test_matches_text_with_trailing_whitespace_stripped(self, tmp_path):
        test_file = tmp_path / "trailing-ws.txt"
        test_file.write_text("line one   \nline two  \nline three\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-1",
            {"path": str(test_file), "edits": [{"oldText": "line one\nline two\n", "newText": "replaced\n"}]},
        )

        assert "Successfully replaced" in get_text_output(result)
        assert test_file.read_text() == "replaced\nline three\n"

    @pytest.mark.tonio
    async def test_matches_fullwidth_punctuation_in_chinese_text(self, tmp_path):
        test_file = tmp_path / "chinese-punctuation.txt"
        test_file.write_text("你好，世界\n你好（世界）\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-chinese",
            {
                "path": str(test_file),
                "edits": [{"oldText": "你好,世界\n你好(世界)\n", "newText": "你好，pidrei\n你好(pidrei)\n"}],
            },
        )

        assert "Successfully replaced" in get_text_output(result)
        assert test_file.read_text() == "你好，pidrei\n你好(pidrei)\n"

    @pytest.mark.tonio
    async def test_matches_compatibility_equivalent_unicode_forms(self, tmp_path):
        test_file = tmp_path / "unicode-compatibility.txt"
        test_file.write_text("ＡＢＣ１２３\ncafé\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-unicode",
            {"path": str(test_file), "edits": [{"oldText": "ABC123\ncafé\n", "newText": "XYZ789\ncoffee\n"}]},
        )

        assert "Successfully replaced" in get_text_output(result)
        assert test_file.read_text() == "XYZ789\ncoffee\n"

    @pytest.mark.tonio
    async def test_matches_smart_single_quotes_to_ascii_quotes(self, tmp_path):
        test_file = tmp_path / "smart-quotes.txt"
        test_file.write_text("console.log(\u2018hello\u2019);\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-2",
            {
                "path": str(test_file),
                "edits": [{"oldText": "console.log('hello');", "newText": "console.log('world');"}],
            },
        )

        assert "Successfully replaced" in get_text_output(result)
        assert "world" in test_file.read_text()

    @pytest.mark.tonio
    async def test_matches_smart_double_quotes_to_ascii_quotes(self, tmp_path):
        test_file = tmp_path / "smart-double-quotes.txt"
        test_file.write_text("const msg = \u201cHello World\u201d;\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-3",
            {
                "path": str(test_file),
                "edits": [{"oldText": 'const msg = "Hello World";', "newText": 'const msg = "Goodbye";'}],
            },
        )

        assert "Successfully replaced" in get_text_output(result)
        assert "Goodbye" in test_file.read_text()

    @pytest.mark.tonio
    async def test_matches_unicode_dashes_to_ascii_hyphen(self, tmp_path):
        test_file = tmp_path / "unicode-dashes.txt"
        test_file.write_text("range: 1\u20135\nbreak\u2014here\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-4",
            {
                "path": str(test_file),
                "edits": [{"oldText": "range: 1-5\nbreak-here", "newText": "range: 10-50\nbreak--here"}],
            },
        )

        assert "Successfully replaced" in get_text_output(result)
        assert "10-50" in test_file.read_text()

    @pytest.mark.tonio
    async def test_matches_non_breaking_space_to_regular_space(self, tmp_path):
        test_file = tmp_path / "nbsp.txt"
        test_file.write_text("hello world\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-5",
            {"path": str(test_file), "edits": [{"oldText": "hello world", "newText": "hello universe"}]},
        )

        assert "Successfully replaced" in get_text_output(result)
        assert "universe" in test_file.read_text()

    @pytest.mark.tonio
    async def test_prefers_exact_match_over_fuzzy_match(self, tmp_path):
        test_file = tmp_path / "exact-preferred.txt"
        test_file.write_text("const x = 'exact';\nconst y = 'other';\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-6",
            {"path": str(test_file), "edits": [{"oldText": "const x = 'exact';", "newText": "const x = 'changed';"}]},
        )

        assert "Successfully replaced" in get_text_output(result)
        assert test_file.read_text() == "const x = 'changed';\nconst y = 'other';\n"

    @pytest.mark.tonio
    async def test_still_fails_when_text_is_not_found_even_with_fuzzy_matching(self, tmp_path):
        test_file = tmp_path / "no-match.txt"
        test_file.write_text("completely different content\n")
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception, match="Could not find the exact text"):
            await edit_tool.execute(
                "test-fuzzy-7",
                {"path": str(test_file), "edits": [{"oldText": "this does not exist", "newText": "replacement"}]},
            )

    @pytest.mark.tonio
    async def test_detects_duplicates_after_fuzzy_normalization(self, tmp_path):
        test_file = tmp_path / "fuzzy-dups.txt"
        test_file.write_text("hello world   \nhello world\n")
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception, match="Found 2 occurrences"):
            await edit_tool.execute(
                "test-fuzzy-8", {"path": str(test_file), "edits": [{"oldText": "hello world", "newText": "replaced"}]}
            )

    @pytest.mark.tonio
    async def test_supports_fuzzy_matching_in_multi_edit_mode(self, tmp_path):
        test_file = tmp_path / "fuzzy-multi.txt"
        test_file.write_text("console.log(‘hello’);\nhello world\n")
        edit_tool = create_edit_tool(str(tmp_path))

        await edit_tool.execute(
            "test-fuzzy-9",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "console.log('hello');\n", "newText": "console.log('world');\n"},
                    {"oldText": "hello world\n", "newText": "hello universe\n"},
                ],
            },
        )

        assert test_file.read_text() == "console.log('world');\nhello universe\n"

    @pytest.mark.tonio
    async def test_preserves_the_correct_occurrence_when_fuzzy_replacement_equals_a_nearby_line(self, tmp_path):
        test_file = tmp_path / "fuzzy-preserve-duplicate-line.txt"
        original_content = "replace me   \nafter   \n"
        test_file.write_text(original_content)
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-preserve-duplicate-line",
            {"path": str(test_file), "edits": [{"oldText": "replace me\n", "newText": "after\n"}]},
        )

        expected_content = "after\nafter   \n"
        assert test_file.read_text() == expected_content
        assert apply_patch(original_content, result.details.patch) == expected_content

    @pytest.mark.tonio
    async def test_preserves_untouched_lines_and_produces_an_applicable_patch_for_fuzzy_multi_edits(self, tmp_path):
        test_file = tmp_path / "fuzzy-preserve-multi.txt"
        original_content = (
            "keep before  \nfirst target  \nfirst after\nkeep middle   \nsecond target  \nsecond after\nkeep after  \n"
        )
        test_file.write_text(original_content)
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-fuzzy-preserve-multi",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "first target\nfirst after", "newText": "FIRST\nFIRST2"},
                    {"oldText": "second target\nsecond after", "newText": "SECOND\nSECOND2"},
                ],
            },
        )

        expected_content = "keep before  \nFIRST\nFIRST2\nkeep middle   \nSECOND\nSECOND2\nkeep after  \n"
        assert test_file.read_text() == expected_content
        assert apply_patch(original_content, result.details.patch) == expected_content


class TestEditToolCrlfHandling:
    @pytest.mark.tonio
    async def test_matches_lf_old_text_against_crlf_file_content(self, tmp_path):
        test_file = tmp_path / "crlf-test.txt"
        test_file.write_bytes(b"line one\r\nline two\r\nline three\r\n")
        edit_tool = create_edit_tool(str(tmp_path))

        result = await edit_tool.execute(
            "test-crlf-1",
            {"path": str(test_file), "edits": [{"oldText": "line two\n", "newText": "replaced line\n"}]},
        )

        assert "Successfully replaced" in get_text_output(result)

    @pytest.mark.tonio
    async def test_preserves_crlf_line_endings_after_edit(self, tmp_path):
        test_file = tmp_path / "crlf-preserve.txt"
        test_file.write_bytes(b"first\r\nsecond\r\nthird\r\n")
        edit_tool = create_edit_tool(str(tmp_path))

        await edit_tool.execute(
            "test-crlf-2", {"path": str(test_file), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]}
        )

        assert test_file.read_bytes() == b"first\r\nREPLACED\r\nthird\r\n"

    @pytest.mark.tonio
    async def test_preserves_lf_line_endings_for_lf_files(self, tmp_path):
        test_file = tmp_path / "lf-preserve.txt"
        test_file.write_bytes(b"first\nsecond\nthird\n")
        edit_tool = create_edit_tool(str(tmp_path))

        await edit_tool.execute(
            "test-lf-1", {"path": str(test_file), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]}
        )

        assert test_file.read_bytes() == b"first\nREPLACED\nthird\n"

    @pytest.mark.tonio
    async def test_detects_duplicates_across_crlf_lf_variants(self, tmp_path):
        test_file = tmp_path / "mixed-endings.txt"
        test_file.write_bytes(b"hello\r\nworld\r\n---\r\nhello\nworld\n")
        edit_tool = create_edit_tool(str(tmp_path))

        with pytest.raises(Exception, match="Found 2 occurrences"):
            await edit_tool.execute(
                "test-crlf-dup",
                {"path": str(test_file), "edits": [{"oldText": "hello\nworld\n", "newText": "replaced\n"}]},
            )

    @pytest.mark.tonio
    async def test_preserves_utf8_bom_after_edit(self, tmp_path):
        test_file = tmp_path / "bom-test.txt"
        test_file.write_bytes("\ufefffirst\r\nsecond\r\nthird\r\n".encode())
        edit_tool = create_edit_tool(str(tmp_path))

        await edit_tool.execute(
            "test-bom", {"path": str(test_file), "edits": [{"oldText": "second\n", "newText": "REPLACED\n"}]}
        )

        assert test_file.read_bytes().decode("utf-8") == "\ufefffirst\r\nREPLACED\r\nthird\r\n"

    @pytest.mark.tonio
    async def test_preserves_crlf_line_endings_and_bom_in_multi_edit_mode(self, tmp_path):
        test_file = tmp_path / "bom-crlf-multi.txt"
        test_file.write_bytes("\ufefffirst\r\nsecond\r\nthird\r\nfourth\r\n".encode())
        edit_tool = create_edit_tool(str(tmp_path))

        await edit_tool.execute(
            "test-crlf-multi",
            {
                "path": str(test_file),
                "edits": [
                    {"oldText": "second\n", "newText": "SECOND\n"},
                    {"oldText": "fourth\n", "newText": "FOURTH\n"},
                ],
            },
        )

        assert test_file.read_bytes().decode("utf-8") == "\ufefffirst\r\nSECOND\r\nthird\r\nFOURTH\r\n"
