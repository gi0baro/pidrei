"""Mirror of pi coding-agent test/grep-operations.test.ts.

pi runs the deprecated `readFile` hook case twice, with a synchronous and an
asynchronous hook; operations members are async-only here, so the pair is
one case.
"""

import shutil

import pytest

from pidrei.core.tools import create_grep_tool
from pidrei_ai.types import TextContent


pytestmark = pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")

EXPECTED_CONTENT = [TextContent(text="file.txt-1- before\nfile.txt:2: needle\nfile.txt-3- after")]


# #9276: restoring the public hook must not reintroduce whole-file reads for context.
@pytest.mark.tonio
async def test_accepts_a_deprecated_read_file_hook_without_calling_it(tmp_path):
    (tmp_path / "file.txt").write_text("before\nneedle\nafter\n")
    calls: list[str] = []

    class Operations:
        async def is_directory(self, _absolute_path: str) -> bool:
            return True

        async def read_file(self, absolute_path: str) -> str:
            calls.append(absolute_path)
            return "custom context"

    tool = create_grep_tool(str(tmp_path), operations=Operations())

    result = await tool.execute("grep", {"pattern": "needle", "context": 1})

    assert result.content == EXPECTED_CONTENT
    assert calls == []


@pytest.mark.tonio
async def test_allows_custom_operations_without_a_read_file_hook(tmp_path):
    (tmp_path / "file.txt").write_text("before\nneedle\nafter\n")

    class Operations:
        async def is_directory(self, _absolute_path: str) -> bool:
            return True

    tool = create_grep_tool(str(tmp_path), operations=Operations())

    result = await tool.execute("grep", {"pattern": "needle", "context": 1})

    assert result.content == EXPECTED_CONTENT
