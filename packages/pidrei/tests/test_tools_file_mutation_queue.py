"""Mirror of pi coding-agent test/file-mutation-queue.test.ts."""

import os

import pytest
import tonio.colored as tonio
from tonio.colored import time as tonio_time

from pidrei.core.tools.file_mutation_queue import with_file_mutation_queue
from pidrei.core.tools.write import create_write_tool


class TestWithFileMutationQueue:
    @pytest.mark.tonio
    async def test_serializes_operations_for_the_same_file(self, tmp_path):
        order: list[str] = []
        path = str(tmp_path / "file-mutation-queue-same")

        async def first_op():
            order.append("first:start")
            await tonio_time.sleep(0.03)
            order.append("first:end")

        async def second_op():
            order.append("second:start")
            order.append("second:end")

        await tonio.spawn(
            with_file_mutation_queue(path, first_op),
            with_file_mutation_queue(path, second_op),
        )

        assert order == ["first:start", "first:end", "second:start", "second:end"]

    @pytest.mark.tonio
    async def test_allows_different_files_to_proceed_in_parallel(self, tmp_path):
        order: list[str] = []

        async def op_a():
            order.append("a:start")
            await tonio_time.sleep(0.03)
            order.append("a:end")

        async def op_b():
            order.append("b:start")
            await tonio_time.sleep(0.03)
            order.append("b:end")

        await tonio.spawn(
            with_file_mutation_queue(str(tmp_path / "file-mutation-queue-a"), op_a),
            with_file_mutation_queue(str(tmp_path / "file-mutation-queue-b"), op_b),
        )

        assert order.index("a:start") < order.index("a:end")
        assert order.index("b:start") < order.index("b:end")
        assert order.index("b:start") < order.index("a:end")

    @pytest.mark.tonio
    async def test_uses_the_same_queue_for_symlink_aliases(self, tmp_path):
        target_path = tmp_path / "target.txt"
        symlink_path = tmp_path / "alias.txt"
        target_path.write_text("original")
        os.symlink(target_path, symlink_path)

        order: list[str] = []

        async def through_target():
            order.append("target:start")
            await tonio_time.sleep(0.03)
            order.append("target:end")

        async def through_alias():
            order.append("alias:start")
            order.append("alias:end")

        await tonio.spawn(
            with_file_mutation_queue(str(target_path), through_target),
            with_file_mutation_queue(str(symlink_path), through_alias),
        )

        assert order == ["target:start", "target:end", "alias:start", "alias:end"]

    @pytest.mark.tonio
    async def test_serializes_concurrent_write_tool_calls_to_the_same_file(self, tmp_path):
        write_tool = create_write_tool(str(tmp_path))
        test_file = tmp_path / "concurrent.txt"

        await tonio.spawn(
            write_tool.execute("write-1", {"path": str(test_file), "content": "one\n" * 200}),
            write_tool.execute("write-2", {"path": str(test_file), "content": "two\n" * 200}),
        )

        content = test_file.read_text()
        # One writer must fully win; interleaving would mix lines.
        assert content in ("one\n" * 200, "two\n" * 200)
