"""Mirror of pi agent/test/harness/tools.test.ts (read/write/edit portions)."""

import base64
import os
import re
import struct
import tempfile

import pytest
import tonio.colored as tonio

from pidrei_agent.harness.env.local import LocalExecutionEnv
from pidrei_agent.harness.tools.edit import create_edit_tool
from pidrei_agent.harness.tools.read import ReadImageProcessorResult, ReadToolOptions, create_read_tool
from pidrei_agent.harness.tools.tool_context import ExecutionToolContext
from pidrei_agent.harness.tools.write import create_write_tool
from pidrei_agent.harness.types import get_or_throw
from pidrei_agent.types import AgentToolResult
from pidrei_ai.types import ImageContent
from pidrei_ai.utils.cancel import CancelToken


def create_temp_dir() -> str:
    return tempfile.mkdtemp(prefix="pidrei-agent-test-")


def text_output(result: AgentToolResult) -> str:
    return "\n".join(part.text for part in result.content if part.type == "text")


def create_context() -> ExecutionToolContext:
    return ExecutionToolContext(env=LocalExecutionEnv(cwd=create_temp_dir()))


def apply_patch(original: str, patch: str) -> str:
    """Minimal unified-patch applier (mirror of the TS test's jsdiff `applyPatch`)."""
    src = original.splitlines(keepends=True)
    out: list[str] = []
    src_index = 0
    patch_lines = patch.split("\n")
    index = 0
    while index < len(patch_lines):
        line = patch_lines[index]
        header = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
        if header is None:
            index += 1
            continue
        old_start = int(header.group(1)) - 1
        assert old_start >= src_index, "hunks out of order"
        out.extend(src[src_index:old_start])
        src_index = old_start
        index += 1
        while index < len(patch_lines) and patch_lines[index][:1] in (" ", "-", "+", "\\"):
            hunk_line = patch_lines[index]
            if hunk_line.startswith("\\"):
                # "\ No newline at end of file": strip the padded newline.
                if out and out[-1].endswith("\n"):
                    out[-1] = out[-1][:-1]
            elif hunk_line.startswith(" "):
                assert src[src_index].rstrip("\n") == hunk_line[1:], "context mismatch"
                out.append(src[src_index])
                src_index += 1
            elif hunk_line.startswith("-"):
                assert src[src_index].rstrip("\n") == hunk_line[1:], "removed-line mismatch"
                src_index += 1
            else:
                out.append(hunk_line[1:] + "\n")
            index += 1
    out.extend(src[src_index:])
    return "".join(out)


class SlowReadExecutionEnv(LocalExecutionEnv):
    async def read_text_file(self, path, cancel=None):
        await tonio.sleep(0.02)
        return await super().read_text_file(path, cancel)


class BlockingWriteExecutionEnv(LocalExecutionEnv):
    def __init__(self, cwd: str):
        super().__init__(cwd)
        self.first_write_started = tonio.Event()
        self.finish_first_write = tonio.Event()
        self.second_write_started = False

    async def write_file(self, path, content, cancel=None):
        if content == "first\n":
            self.first_write_started.set()
            await self.finish_first_write.wait(None)
        elif content == "second\n":
            self.second_write_started = True
        return await super().write_file(path, content, cancel)


class BlockingEditExecutionEnv(LocalExecutionEnv):
    def __init__(self, cwd: str):
        super().__init__(cwd)
        self.first_edit_write_started = tonio.Event()
        self.finish_first_edit_write = tonio.Event()
        self.first_edit_write_settled = False
        self.second_edit_write_started = False

    async def write_file(self, path, content, cancel=None):
        if content == "ALPHA\nbeta\n":
            self.first_edit_write_started.set()
            await self.finish_first_edit_write.wait(None)
            result = await super().write_file(path, content)
            self.first_edit_write_settled = True
            return result
        if content in ("ALPHA\nBETA\n", "alpha\nBETA\n"):
            self.second_edit_write_started = True
        return await super().write_file(path, content, cancel)


def create_tiny_bmp() -> bytes:
    data = bytearray(58)
    data[0] = 0x42
    data[1] = 0x4D
    struct.pack_into("<I", data, 2, len(data))
    struct.pack_into("<I", data, 10, 54)
    struct.pack_into("<I", data, 14, 40)
    struct.pack_into("<i", data, 18, 1)
    struct.pack_into("<i", data, 22, 1)
    struct.pack_into("<H", data, 26, 1)
    struct.pack_into("<H", data, 28, 24)
    struct.pack_into("<I", data, 34, 4)
    return bytes(data)


# --- read ---------------------------------------------------------------------


@pytest.mark.tonio
async def test_read_reads_text_with_offsets_limits_and_continuation_notices():
    context = create_context()
    get_or_throw(await context.env.write_file("test.txt", "\n".join(f"Line {i + 1}" for i in range(100))))

    result = await create_read_tool().execute(
        "read-1", {"path": "test.txt", "offset": 41, "limit": 20}, None, None, context
    )
    output = text_output(result)

    assert "Line 40" not in output
    assert "Line 41" in output
    assert "Line 60" in output
    assert "Line 61" not in output
    assert "[40 more lines in file. Use offset=61 to continue.]" in output


@pytest.mark.tonio
async def test_read_truncates_large_text_by_line_count():
    context = create_context()
    get_or_throw(await context.env.write_file("large.txt", "\n".join(f"Line {i + 1}" for i in range(2500))))

    result = await create_read_tool().execute("read-2", {"path": "large.txt"}, None, None, context)

    assert "[Showing lines 1-2000 of 2500. Use offset=2001 to continue.]" in text_output(result)
    assert result.details is not None
    truncation = result.details.truncation
    assert (truncation.truncated, truncation.truncated_by, truncation.total_lines, truncation.output_lines) == (
        True,
        "lines",
        2500,
        2000,
    )


@pytest.mark.tonio
async def test_read_does_not_count_a_trailing_newline_as_an_extra_line_at_the_truncation_limit():
    context = create_context()
    get_or_throw(await context.env.write_file("exact.txt", "\n".join("x" for _ in range(2000)) + "\n"))

    result = await create_read_tool().execute("read-exact", {"path": "exact.txt"}, None, None, context)

    assert result.details is None
    assert "Use offset=" not in text_output(result)


@pytest.mark.tonio
async def test_read_rejects_offsets_beyond_the_file():
    context = create_context()
    get_or_throw(await context.env.write_file("short.txt", "one\ntwo\nthree"))

    with pytest.raises(Exception, match=re.escape("Offset 100 is beyond end of file (3 lines total)")):
        await create_read_tool().execute("read-3", {"path": "short.txt", "offset": 100}, None, None, context)


@pytest.mark.tonio
async def test_read_detects_supported_images_by_content():
    context = create_context()
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAX+XDSwAAAABJRU5ErkJggg=="
    png = base64.b64decode(png_base64)
    get_or_throw(await context.env.write_file("image.txt", png))

    result = await create_read_tool().execute("read-4", {"path": "image.txt"}, None, None, context)

    assert "Read image file [image/png]" in text_output(result)
    assert ImageContent(data=png_base64, mime_type="image/png") in result.content


@pytest.mark.tonio
async def test_read_delegates_image_conversion_and_resizing_to_an_injected_processor():
    context = create_context()
    bmp = create_tiny_bmp()
    get_or_throw(await context.env.write_file("image.bmp", bmp))
    received = {}

    async def image_processor(data, mime_type, auto_resize_images):
        received["bytes"] = data
        received["mime_type"] = mime_type
        received["auto_resize_images"] = auto_resize_images
        return ReadImageProcessorResult(
            ok=True,
            data="converted",
            mime_type="image/png",
            hints=["[Image converted from image/bmp to image/png.]"],
        )

    tool = create_read_tool(ReadToolOptions(auto_resize_images=False, image_processor=image_processor))

    result = await tool.execute("read-bmp", {"path": "image.bmp"}, None, None, context)

    assert received["mime_type"] == "image/bmp"
    assert received["auto_resize_images"] is False
    assert received["bytes"] == bmp
    assert "[Image converted from image/bmp to image/png.]" in text_output(result)
    assert ImageContent(data="converted", mime_type="image/png") in result.content


# --- write --------------------------------------------------------------------


@pytest.mark.tonio
async def test_write_writes_files_and_creates_parent_directories():
    context = create_context()
    result = await create_write_tool().execute(
        "write-1", {"path": "nested/dir/file.txt", "content": "hello"}, None, None, context
    )

    assert text_output(result) == "Successfully wrote 5 bytes to nested/dir/file.txt"
    assert get_or_throw(await context.env.read_text_file("nested/dir/file.txt")) == "hello"


@pytest.mark.tonio
async def test_write_keeps_the_mutation_queue_locked_until_an_aborted_write_settles():
    env = BlockingWriteExecutionEnv(cwd=create_temp_dir())
    tool = create_write_tool()
    controller = CancelToken()
    first_write = tonio.spawn(
        tool.execute(
            "write-first", {"path": "file.txt", "content": "first\n"}, controller, None, ExecutionToolContext(env=env)
        )
    )
    await env.first_write_started.wait(None)
    controller.cancel()
    second_write = tonio.spawn(
        tool.execute(
            "write-second", {"path": "file.txt", "content": "second\n"}, None, None, ExecutionToolContext(env=env)
        )
    )

    await tonio.sleep(0.02)
    assert env.second_write_started is False
    env.finish_first_write.set()
    with pytest.raises(Exception):
        await first_write
    await second_write
    assert get_or_throw(await env.read_text_file("file.txt")) == "second\n"


# --- edit ---------------------------------------------------------------------


@pytest.mark.tonio
async def test_edit_applies_disjoint_edits_and_returns_both_diff_formats():
    context = create_context()
    original = "alpha\nbeta\ngamma\ndelta\n"
    get_or_throw(await context.env.write_file("edit.txt", original))

    result = await create_edit_tool().execute(
        "edit-1",
        {
            "path": "edit.txt",
            "edits": [
                {"oldText": "alpha\n", "newText": "ALPHA\n"},
                {"oldText": "gamma\n", "newText": "GAMMA\n"},
            ],
        },
        None,
        None,
        context,
    )

    assert text_output(result) == "Successfully replaced 2 block(s) in edit.txt."
    assert result.details is not None
    assert "ALPHA" in result.details.diff
    assert "GAMMA" in result.details.diff
    assert apply_patch(original, result.details.patch) == "ALPHA\nbeta\nGAMMA\ndelta\n"
    assert get_or_throw(await context.env.read_text_file("edit.txt")) == "ALPHA\nbeta\nGAMMA\ndelta\n"


@pytest.mark.tonio
async def test_edit_matches_all_edits_against_the_original_and_rejects_overlaps():
    context = create_context()
    get_or_throw(await context.env.write_file("edit.txt", "one\ntwo\nthree\n"))

    with pytest.raises(Exception, match="overlap"):
        await create_edit_tool().execute(
            "edit-2",
            {
                "path": "edit.txt",
                "edits": [
                    {"oldText": "one\ntwo\n", "newText": "ONE\nTWO\n"},
                    {"oldText": "two\nthree\n", "newText": "TWO\nTHREE\n"},
                ],
            },
            None,
            None,
            context,
        )
    assert get_or_throw(await context.env.read_text_file("edit.txt")) == "one\ntwo\nthree\n"


@pytest.mark.tonio
async def test_edit_rejects_missing_and_duplicate_target_text():
    context = create_context()
    get_or_throw(await context.env.write_file("edit.txt", "foo foo foo"))
    tool = create_edit_tool()

    with pytest.raises(Exception, match="Could not find the exact text"):
        await tool.execute(
            "edit-3", {"path": "edit.txt", "edits": [{"oldText": "bar", "newText": "baz"}]}, None, None, context
        )
    with pytest.raises(Exception, match="Found 3 occurrences"):
        await tool.execute(
            "edit-4", {"path": "edit.txt", "edits": [{"oldText": "foo", "newText": "bar"}]}, None, None, context
        )


@pytest.mark.tonio
async def test_edit_keeps_the_mutation_queue_locked_until_an_aborted_edit_write_settles():
    env = BlockingEditExecutionEnv(cwd=create_temp_dir())
    get_or_throw(await env.write_file("file.txt", "alpha\nbeta\n"))
    tool = create_edit_tool()
    controller = CancelToken()
    first_edit = tonio.spawn(
        tool.execute(
            "edit-first",
            {"path": "file.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}]},
            controller,
            None,
            ExecutionToolContext(env=env),
        )
    )
    await env.first_edit_write_started.wait(None)
    controller.cancel()
    second_edit = tonio.spawn(
        tool.execute(
            "edit-second",
            {"path": "file.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]},
            None,
            None,
            ExecutionToolContext(env=env),
        )
    )

    await tonio.sleep(0.02)
    assert env.second_edit_write_started is False
    env.finish_first_edit_write.set()
    with pytest.raises(ExceptionGroup) as excinfo:
        await first_edit
    assert "Operation aborted" in str(excinfo.value.exceptions[0])
    await second_edit
    assert env.first_edit_write_settled is True
    assert get_or_throw(await env.read_text_file("file.txt")) == "ALPHA\nBETA\n"


@pytest.mark.tonio
async def test_edit_serializes_concurrent_edits_through_canonical_and_symlink_paths():
    env = SlowReadExecutionEnv(cwd=create_temp_dir())
    get_or_throw(await env.write_file("target.txt", "alpha\nbeta\ngamma\n"))
    os.symlink("target.txt", os.path.join(env.cwd, "link.txt"))
    tool = create_edit_tool()

    await tonio.spawn(
        tool.execute(
            "edit-target",
            {"path": "target.txt", "edits": [{"oldText": "alpha", "newText": "ALPHA"}]},
            None,
            None,
            ExecutionToolContext(env=env),
        ),
        tool.execute(
            "edit-link",
            {"path": "link.txt", "edits": [{"oldText": "beta", "newText": "BETA"}]},
            None,
            None,
            ExecutionToolContext(env=env),
        ),
    )

    assert get_or_throw(await env.read_text_file("target.txt")) == "ALPHA\nBETA\ngamma\n"


@pytest.mark.tonio
async def test_edit_edits_regular_files_through_symlinks():
    context = create_context()
    get_or_throw(await context.env.write_file("target.txt", "before\n"))
    os.symlink("target.txt", os.path.join(context.env.cwd, "link.txt"))

    await create_edit_tool().execute(
        "edit-symlink", {"path": "link.txt", "edits": [{"oldText": "before", "newText": "after"}]}, None, None, context
    )

    assert get_or_throw(await context.env.read_text_file("target.txt")) == "after\n"


@pytest.mark.tonio
async def test_edit_preserves_bom_and_crlf_line_endings():
    context = create_context()
    get_or_throw(await context.env.write_file("edit.txt", "﻿one\r\ntwo\r\n"))

    await create_edit_tool().execute(
        "edit-5", {"path": "edit.txt", "edits": [{"oldText": "two", "newText": "TWO"}]}, None, None, context
    )

    assert get_or_throw(await context.env.read_text_file("edit.txt")) == "﻿one\r\nTWO\r\n"
