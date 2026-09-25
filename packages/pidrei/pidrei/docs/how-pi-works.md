# How pidrei works

pidrei coordinates model requests, tool execution, context assembly, and
session storage. A session is pidrei's record of a conversation: messages, tool
calls and results, model changes, compactions, and other events.

The entries of a session form a tree. Each path through it is a branch, and the
branch ending at the current entry (the leaf) is the active branch: it supplies
the history for the next model request.

## Agent loop

A submitted message is appended to the active branch. pidrei builds a model
request from the system prompt, the active branch, the available tools, and the
model settings, and sends it to the selected provider.

The provider streams back an assistant response, which can contain text,
thinking, and tool calls. pidrei records the response, executes each tool call,
and records the results. That is one turn. If tool results or queued messages
need another model request, pidrei starts another turn; otherwise the run ends.

Steering messages are delivered after the current assistant turn and its tool
calls. Follow-up messages are delivered once the agent has no pending work.
Aborting stops the current run and returns queued messages to the editor.

## Context

The active branch supplies the conversation history. pidrei converts its
session entries into model-compatible system, user, assistant, and tool-result
messages; see [message-types.md](message-types.md).

The system prompt is built from pidrei's base instructions (or `SYSTEM.md`),
`APPEND_SYSTEM.md`, and the discovered context files
([configuration.md](configuration.md)). The request also carries the tool
definitions and the skill descriptions. A skill's full instructions are read
only when the model asks for them. Extensions can add instructions and rewrite
the context of each request.

Prompt templates expand editor input before it becomes a user message.
Referenced files, images, pasted text, and shell output (`!command`) can become
message content.

## Sessions

Persistent sessions are JSONL files, in the same format pi writes. Each tree
entry has an `id` and a `parentId`; the leaf identifies the active branch.
By default they live under `~/.pidrei/agent/sessions/`, one directory per
working directory.

Continuing from an earlier entry (`/tree`) creates another branch in the same
file. `/fork` and `/clone` copy selected history into a new session file.

Model context is rebuilt from the active branch on every request. Compaction
appends a summary entry that stands in for older messages in later requests; a
`context_edit` entry omits or replaces one earlier entry's content. Either
way, the original entries stay in the file.

## Interfaces

Interactive mode renders session and agent events in the terminal. Print mode
(`-p`) runs a prompt and writes the final response. JSON mode
(`--mode json`) writes agent events as JSONL. See [cli.md](cli.md).

pidrei's Python packages ([sdk.md](sdk.md)) create and control agent sessions
in process. Every interface uses the same agent and session machinery.

## Extensions and resources

Extensions are Python modules loaded into the pidrei process. Their
`async def extension(pi)` factories register tools, commands, shortcuts, flags,
providers, event handlers, renderers, and terminal UI
([extensions.md](extensions.md)).

Skills provide on-demand instructions and supporting files. Prompt templates
provide reusable message text. Themes provide terminal colors. Packages
distribute all of these from git or a local path ([packages.md](packages.md)).

## Trust and permissions

pidrei resolves project trust before it loads project settings and resources;
after the trust decision it loads extensions, then the other resources and the
context files. Enabled tools run with the operating-system permissions of the
pidrei process, and extensions execute inside that process — there is no
sandbox.
