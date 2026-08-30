"""Mirror of pi coding-agent src/modes/rpc/rpc-types.ts.

RPC protocol types for headless operation. Commands are sent as JSON lines
on stdin; responses and events are emitted as JSON lines on stdout.

pi declares the full command/response union as TypeScript types over plain
JSON objects. The wire format is identical here, so commands and responses
stay plain dicts; this module documents the protocol and keeps the few
shapes that are constructed as values.

Commands (stdin, camelCase keys, optional "id" for correlation):
  prompt {message, images?, streamingBehavior?}, steer {message, images?},
  follow_up {message, images?}, abort, clear_queue,
  new_session {parentSession?},
  get_state, set_model {provider, modelId}, cycle_model,
  get_available_models, set_thinking_level {level}, cycle_thinking_level,
  get_available_thinking_levels, set_steering_mode {mode},
  set_follow_up_mode {mode}, compact {customInstructions?},
  set_auto_compaction {enabled}, set_auto_retry {enabled}, abort_retry,
  bash {command, excludeFromContext?}, abort_bash, get_session_stats,
  export_html {outputPath?}, switch_session {sessionPath}, fork {entryId},
  clone, get_fork_messages, get_entries {since?}, get_tree,
  get_last_assistant_text, set_session_name {name}, get_messages,
  get_commands.

Responses (stdout): {id?, type: "response", command, success: true, data?}
or {id?, type: "response", command, success: false, error}.

Extension UI requests (stdout): {type: "extension_ui_request", id, method,
...} with methods select/confirm/input/editor/notify/setStatus/setWidget/
setTitle/set_editor_text. The client answers select/confirm/input/editor
with {type: "extension_ui_response", id, value | confirmed | cancelled}.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RpcSlashCommand:
    """A command available for invocation via prompt (get_commands data)."""

    # Command name (without leading slash)
    name: str
    # What kind of command this is: "extension" | "prompt" | "skill"
    source: str
    # Source metadata for the owning resource
    source_info: Any = None
    # Human-readable description
    description: str | None = None


@dataclass(slots=True)
class RpcSessionState:
    """get_state response data (serialized to camelCase on the wire)."""

    thinking_level: str
    is_streaming: bool
    is_compacting: bool
    steering_mode: str
    follow_up_mode: str
    session_id: str
    auto_compaction_enabled: bool
    message_count: int
    pending_message_count: int
    model: Any = None
    session_file: str | None = None
    session_name: str | None = None
