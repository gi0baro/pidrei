"""Mirror of pi's compaction-nested-calls.test.ts."""

from pidrei.core.compaction.utils import compute_file_lists, create_file_ops, extract_file_ops_from_message
from pidrei_ai.types import NestedToolCallRecord, NestedToolCalls, ToolResultMessage


def test_include_files_touched_by_nested_calls_recorded_on_tool_results():
    result = ToolResultMessage(
        tool_call_id="codemode-1",
        tool_name="codemode",
        content=[],
        is_error=False,
        timestamp=0,
        nested_calls=NestedToolCalls(
            calls=[
                NestedToolCallRecord(id="codemode-1/1", name="read", arguments={"path": "a.ts"}, status="ok"),
                NestedToolCallRecord(
                    id="codemode-1/2", name="edit", arguments={"path": "b.ts", "edits": []}, status="ok"
                ),
                NestedToolCallRecord(id="codemode-1/3", name="write", arguments_bytes=40000, status="ok"),
            ],
            complete=False,
        ),
    )
    file_ops = create_file_ops()
    extract_file_ops_from_message(result, file_ops)
    assert compute_file_lists(file_ops) == (["a.ts"], ["b.ts"])
