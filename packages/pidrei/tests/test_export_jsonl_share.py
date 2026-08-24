"""Mirror of pi coding-agent test/export-jsonl-share.test.ts.

pi's case drives `exportSessionForShare` from `session-share.ts`, which
appends a `pi.share` presentation entry for the Radius upload. Radius is
dropped surface here (see FEASIBILITY), so the trailing entry is built in
the test instead — what is under test either way is
`export_session_to_jsonl`: that a trailing entry chains onto the last
conversation entry without disturbing any of their ids or links, that a
plain export emits none, and that the file still reopens as a session.
"""

import json
import shutil
import tempfile

import pytest

from pidrei.core.session_export import export_session_to_jsonl
from pidrei.core.session_manager import SessionManager
from pidrei_ai.types import AssistantMessage, TextContent, ToolCall, ToolResultMessage

from .coding_session_helpers import assistant_msg, now_ms, user_msg
from .harness import create_harness


def read_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle.read().strip().split("\n")]


@pytest.mark.tonio
async def test_adds_presentation_data_without_changing_conversation_ids_or_links():
    temp_dir = tempfile.mkdtemp(prefix="pidrei-jsonl-share-")
    harness = await create_harness()
    session_manager = harness.session.session_manager
    try:
        user_id = await session_manager.append_message(user_msg("hello"))
        assistant: AssistantMessage = assistant_msg(
            "",
            content=[ToolCall(id="call-1", name="share_tool", arguments={"value": "example"})],
            stop_reason="toolUse",
        )
        assistant_id = await session_manager.append_message(assistant)
        result = ToolResultMessage(
            tool_call_id="call-1",
            tool_name="share_tool",
            content=[TextContent(text="done")],
            is_error=False,
            timestamp=now_ms(),
        )
        result_id = await session_manager.append_message(result)
        original_entry_ids = [entry["id"] for entry in session_manager.get_branch()]

        normal_path = f"{temp_dir}/normal.jsonl"
        await harness.session.export_to_jsonl(normal_path)
        normal_records = read_records(normal_path)
        assert not any(record.get("type") == "custom" for record in normal_records)
        assert [record["id"] for record in normal_records[1:]] == original_entry_ids

        share_path = f"{temp_dir}/share.jsonl"

        def trailing(parent_id, timestamp):
            return [
                {
                    "type": "custom",
                    "customType": "pidrei.share",
                    "id": "share001",
                    "parentId": parent_id,
                    "timestamp": timestamp,
                    "data": {"systemPrompt": harness.session.state.system_prompt},
                }
            ]

        await export_session_to_jsonl(session_manager, share_path, trailing)
        records = read_records(share_path)
        conversation_records = records[1:-1]
        assert [record["id"] for record in conversation_records] == original_entry_ids
        assert [record["parentId"] for record in conversation_records] == [None, *original_entry_ids[:-1]]
        assert [record["id"] for record in conversation_records[-3:]] == [user_id, assistant_id, result_id]

        share_entry = records[-1]
        assert share_entry["type"] == "custom"
        assert share_entry["customType"] == "pidrei.share"
        assert share_entry["parentId"] == result_id
        assert share_entry["timestamp"] == records[0]["timestamp"]
        assert share_entry["data"]["systemPrompt"] == harness.session.state.system_prompt

        imported = await SessionManager.open(share_path)
        assert imported.get_leaf_id() == share_entry["id"]
        assert [message.role for message in imported.build_session_context().messages] == [
            "user",
            "assistant",
            "toolResult",
        ]
    finally:
        harness.cleanup()
        shutil.rmtree(temp_dir, ignore_errors=True)
