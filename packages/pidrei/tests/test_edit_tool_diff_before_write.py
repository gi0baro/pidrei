"""Mirror of pi coding-agent test/edit-tool-diff-before-write.test.ts.

pi mocks `generateEditDiff` in the edit-diff module; the edit module's own
binding is swapped here.
"""

import os

import pytest

import pidrei.core.tools.edit as edit_module
from pidrei.core.tools import create_edit_tool


@pytest.mark.tonio
async def test_computes_the_diff_before_writing_so_a_diff_failure_leaves_the_file_unchanged(monkeypatch):
    def failing_generate_edit_diff(*_args, **_kwargs):
        raise Exception("diff failed")

    monkeypatch.setattr(edit_module, "generate_edit_diff", failing_generate_edit_diff)
    writes: list[str] = []

    class Operations:
        async def access(self, _absolute_path: str) -> None:
            return None

        async def read_file(self, _absolute_path: str) -> bytes:
            return b"a\nb\n"

        async def write_file(self, absolute_path: str, _content: str) -> None:
            writes.append(absolute_path)

    tool = create_edit_tool(os.getcwd(), operations=Operations())

    with pytest.raises(Exception, match="diff failed"):
        await tool.execute("test-diff-before-write", {"path": "f.txt", "edits": [{"oldText": "b", "newText": "B"}]})
    assert writes == []
