"""Mirror of pi coding-agent src/core/compaction/utils.ts.

pi's coding-agent compaction utils are a comment-level sibling of the agent
package's harness/compaction/utils.ts (the harness variant additionally guards
tool-call argument serialization with safe_json_stringify; output text is
identical for serializable arguments). pidrei keeps the single implementation
in pidrei_agent and re-exports it here. The summarization system prompt lives
in this module in pi's coding-agent layout, so it is re-exported here too.
"""

from pidrei_agent.harness.compaction.compaction import SUMMARIZATION_SYSTEM_PROMPT
from pidrei_agent.harness.compaction.utils import (
    TOOL_RESULT_MAX_CHARS,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)


__all__ = [
    "SUMMARIZATION_SYSTEM_PROMPT",
    "TOOL_RESULT_MAX_CHARS",
    "FileOperations",
    "compute_file_lists",
    "create_file_ops",
    "extract_file_ops_from_message",
    "format_file_operations",
    "serialize_conversation",
]
