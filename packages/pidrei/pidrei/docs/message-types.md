# Message types

pidrei passes `AgentMessage` values through agent state, lifecycle and
extension events, and persisted session `message` entries. This page defines
those messages and their content blocks.

Each message has two shapes:

- **In process** — extensions, event handlers, and library code see dataclass
  instances with snake_case attributes, as shown below. The `pidrei_ai`
  messages and blocks are frozen: build a changed copy with
  `dataclasses.replace`.
- **On the wire** — session JSONL files and `--mode json` output carry JSON
  objects with pi's camelCase keys (`tool_call_id` → `toolCallId`,
  `cache_write_1h` → `cacheWrite1h`). Optional fields that are `None` are
  omitted; `fromId` is the exception and is written as `null`. This is the
  same shape pi writes, so sessions stay interchangeable.

Message timestamps are Unix timestamps in milliseconds. Session entries carry
their own ISO 8601 `timestamp` strings, which are different.

Source definitions:

- `pidrei_ai.types` — provider-facing messages and content blocks.
- `pidrei_agent.types` — the open `AgentMessage` union.
- `pidrei.core.messages` — the coding-agent roles (implemented in
  `pidrei_agent.harness.messages`).

## Content blocks

### TextContent

```python
class TextContent:
    text: str
    text_signature: str | None = None
    type: Literal["text"] = "text"
```

`text_signature` holds provider-specific message metadata. Treat it as opaque.

### ImageContent

```python
class ImageContent:
    data: str  # base64
    mime_type: str  # e.g. "image/png", "image/jpeg"
    type: Literal["image"] = "image"
```

### ThinkingContent

```python
class ThinkingContent:
    thinking: str
    thinking_signature: str | None = None
    redacted: bool = False
    type: Literal["thinking"] = "thinking"
```

Thinking signatures hold provider-specific replay data. Treat them as opaque. A
redacted block can have no visible thinking text while keeping an encrypted
payload in `thinking_signature`. On the wire, `redacted` appears only when true.

### ToolCall

```python
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    thought_signature: str | None = None
    namespace: str | None = None
    type: Literal["toolCall"] = "toolCall"
```

`thought_signature` is provider-specific. `namespace` identifies an OpenAI
Responses namespace for dynamically loaded or namespaced tools.

## Usage

Assistant messages always carry usage. Tool results can carry usage when the
tool did nested model work.

```python
class Usage:
    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    cache_write_1h: int | None = None
    reasoning: int | None = None
    total_tokens: int = 0
    cost: UsageCost  # input, output, cache_read, cache_write, total (floats)
```

When present, `reasoning` is already included in `output`; do not add it again.
`cache_write_1h` is the part of `cache_write` written with one-hour retention.

## Base messages

### SystemMessage

```python
class SystemMessage:
    content: str | list[TextContent]
    timestamp: int
    sections: dict[str, str | None] | None = None
    tools_added: list[Tool] | None = None
    tools_removed: list[ToolReference] | None = None
    role: Literal["system"] = "system"
```

The leading system message declares the initial prompt and tools. Later system
messages append instructions, replace named prompt sections (`None` removes
one), and add or remove tools. Replaying them in order yields the current
prompt and tool set.

### UserMessage

```python
class UserMessage:
    content: str | list[TextContent | ImageContent]
    timestamp: int
    role: Literal["user"] = "user"
```

### AssistantMessage

```python
class AssistantMessage:
    content: list[TextContent | ThinkingContent | ToolCall]
    api: str
    provider: str
    model: str
    usage: Usage
    stop_reason: Literal["pending", "stop", "length", "toolUse", "error", "aborted", "deferred"]
    timestamp: int
    response_model: str | None = None
    response_id: str | None = None
    provider_thinking_level: str | None = None
    diagnostics: list[AssistantMessageDiagnostic] | None = None
    error_message: str | None = None
    raw_stop_reason: str | None = None
    end_turn: bool | None = None
    deferred: DeferredHandle | None = None
    role: Literal["assistant"] = "assistant"
```

`response_model` records the concrete model the provider reports when it
differs from the requested one. `response_id`, `provider_thinking_level`,
`diagnostics`, and `raw_stop_reason` preserve provider or runtime details.

`"pending"` marks a partial message while it streams. The completed message in
`message_end` has a terminal stop reason, and only completed messages are
written to the session.

A `"deferred"` response carries the handle needed to retrieve it:

```python
class DeferredHandle:
    provider: str
    model_id: str
    api: str
    id: str
    expires_at: int | None = None
    poll_after_ms: int | None = None
    data: Any = None
```

### ToolResultMessage

```python
class ToolResultMessage:
    tool_call_id: str
    tool_name: str
    content: list[TextContent | ImageContent]
    is_error: bool
    timestamp: int
    details: Any = None
    usage: Usage | None = None
    role: Literal["toolResult"] = "toolResult"
```

`details` is tool-specific. The optional `usage` reports nested model work done
by the tool; it counts toward session statistics but not toward the main
model-call usage.

## Coding-agent messages

The coding agent adds four roles. They are regular (non-frozen) dataclasses
with keyword-only fields.

### BashExecutionMessage

Created by `!command` in the interactive editor; `!!command` sets
`exclude_from_context`. It is not a tool result.

```python
class BashExecutionMessage:
    command: str
    output: str
    exit_code: int | None
    cancelled: bool
    truncated: bool
    timestamp: int
    full_output_path: str | None = None
    exclude_from_context: bool | None = None
    role: Literal["bashExecution"] = "bashExecution"
```

Unless `exclude_from_context` is true, pidrei turns it into user-role text
before the next model request.

### CustomMessage

Created when an extension sends a context message, for example
`pi.send_message({"customType": "my-type", "content": "...", "display": True})`.

```python
class CustomMessage:
    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool
    timestamp: int
    details: Any = None
    role: Literal["custom"] = "custom"
```

pidrei sends its content to the model as a user message. `display` controls
terminal rendering; `details` is never sent to the model. It is persisted as a
`custom_message` session entry.

### BranchSummaryMessage

```python
class BranchSummaryMessage:
    summary: str
    from_id: str | None
    timestamp: int
    role: Literal["branchSummary"] = "branchSummary"
```

Built for model context from a persisted `branch_summary` entry.

### CompactionSummaryMessage

```python
class CompactionSummaryMessage:
    summary: str
    tokens_before: int
    timestamp: int
    role: Literal["compactionSummary"] = "compactionSummary"
```

Built for model context from a persisted `compaction` entry. Both summaries
reach the model as user messages wrapping the text in `<summary>` tags.

## The AgentMessage union

In the coding agent, `AgentMessage` is in practice one of:

```python
SystemMessage | UserMessage | AssistantMessage | ToolResultMessage
| BashExecutionMessage | CustomMessage | BranchSummaryMessage | CompactionSummaryMessage
```

`pidrei_agent.types` declares it as `Message | Any`: pi extends the union
through TypeScript declaration merging, and pidrei instead accepts any object
with a `role` and a `timestamp`. The agent's `convert_to_llm` decides what
reaches the model and drops roles it does not know, so code that accepts
messages from other extensions should tolerate unknown roles.
