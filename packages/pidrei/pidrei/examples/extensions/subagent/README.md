# Subagent Example

Delegate tasks to specialized subagents with isolated context windows.

## Features

- **Isolated context**: Each subagent runs as its own in-process agent session
  with a fresh context window (pi's example spawns a `pi` subprocess instead;
  on tonio, in-process sessions run genuinely in parallel and skip the
  per-task interpreter start)
- **Streaming output**: See tool calls, streaming text, and progress as they
  happen
- **Parallel streaming**: All parallel tasks stream updates simultaneously
- **Markdown rendering**: Final output rendered with proper formatting (expanded view)
- **Usage tracking**: Shows turns, tokens, cost, and context usage per agent
- **Abort support**: Ctrl+C aborts the subagent sessions

## Structure

```
subagent/
├── README.md            # This file
├── __init__.py          # The extension (entry point)
├── agents.py            # Agent discovery logic
├── agents/              # Sample agent definitions
│   ├── scout.md         # Fast recon, returns compressed context
│   ├── planner.md       # Creates implementation plans
│   ├── reviewer.md      # Code review
│   └── worker.md        # General-purpose (full capabilities)
└── prompts/             # Workflow presets (prompt templates)
    ├── implement.md     # scout -> planner -> worker
    ├── scout-and-plan.md    # scout -> planner (no implementation)
    └── implement-and-review.md  # worker -> reviewer -> worker
```

## Installation

Run it directly:

```bash
pidrei -e ./examples/extensions/subagent
```

or install it permanently from the repository root by symlinking the files:

```bash
# Symlink the extension (a directory extension needs its __init__.py)
mkdir -p ~/.pidrei/agent/extensions/subagent
ln -sf "$(pwd)/packages/pidrei/pidrei/examples/extensions/subagent/__init__.py" ~/.pidrei/agent/extensions/subagent/__init__.py
ln -sf "$(pwd)/packages/pidrei/pidrei/examples/extensions/subagent/agents.py" ~/.pidrei/agent/extensions/subagent/agents.py

# Symlink agents
mkdir -p ~/.pidrei/agent/agents
for f in packages/pidrei/pidrei/examples/extensions/subagent/agents/*.md; do
  ln -sf "$(pwd)/$f" ~/.pidrei/agent/agents/$(basename "$f")
done

# Symlink workflow prompts
mkdir -p ~/.pidrei/agent/prompts
for f in packages/pidrei/pidrei/examples/extensions/subagent/prompts/*.md; do
  ln -sf "$(pwd)/$f" ~/.pidrei/agent/prompts/$(basename "$f")
done
```

## Security Model

This tool runs an agent session with a delegated system prompt and tool/model configuration. Subagent sessions load no extensions (so a subagent cannot spawn subagents), persist nothing, and read settings from the task's working directory like a fresh `pidrei` run would.

**Project-local agents** (`.pidrei/agents/*.md`) are repo-controlled prompts that can instruct the model to read files, run bash commands, etc.

**Default behavior:** Only loads **user-level agents** from `~/.pidrei/agent/agents`.

To enable project-local agents, pass `agentScope: "both"` (or `"project"`). Only do this for repositories you trust.

When running interactively, the tool prompts for confirmation before running project-local agents in untrusted projects. Trusted projects skip the additional prompt. Set `confirmProjectAgents: false` to disable confirmation.

## Usage

### Single agent
```
Use scout to find all authentication code
```

### Parallel execution
```
Run 2 scouts in parallel: one to find models, one to find providers
```

### Chained workflow
```
Use a chain: first have scout find the read tool, then have planner suggest improvements
```

### Workflow prompts
```
/implement add Redis caching to the session store
/scout-and-plan refactor auth to support OAuth
/implement-and-review add input validation to API endpoints
```

## Tool Modes

| Mode | Parameter | Description |
|------|-----------|-------------|
| Single | `{ agent, task }` | One agent, one task |
| Parallel | `{ tasks: [...] }` | Multiple agents run concurrently (max 8 tasks; 4 at once by default, tunable via `concurrency`) |
| Chain | `{ chain: [...] }` | Sequential with `{previous}` placeholder |

## Output Display

**Collapsed view** (default):
- Status icon (✓/✗/⏳) and agent name
- Last 5-10 items (tool calls and text)
- Usage stats: `3 turns ↑input ↓output RcacheRead WcacheWrite $cost ctx:contextTokens model`

**Expanded view** (Ctrl+O):
- Full task text
- All tool calls with formatted arguments
- Final output rendered as Markdown
- Per-task usage (for chain/parallel)

**Parallel mode streaming**:
- Shows all tasks with live status (⏳ running, ✓ done, ✗ failed)
- Updates as each task makes progress
- Shows "2/3 done, 1 running" status
- Returns each completed task's final output to the parent model, capped at 50 KB per task
- Returns failure diagnostics from error messages when a task fails before producing output

**Tool call formatting** (mimics built-in tools):
- `$ command` for bash
- `read ~/path:1-10` for read
- `grep /pattern/ in ~/path` for grep
- etc.

## Agent Definitions

Agents are markdown files with YAML frontmatter:

```markdown
---
name: my-agent
description: What this agent does
tools: read, grep, find, ls
model: claude-haiku-4-5
---

System prompt for the agent goes here.
```

When `model` is omitted, the subagent inherits the dispatching session's active model and thinking level.

**Locations:**
- `~/.pidrei/agent/agents/*.md` - User-level (always loaded)
- `.pidrei/agents/*.md` - Project-level (only with `agentScope: "project"` or `"both"`)

Project agents override user agents with the same name when `agentScope: "both"`.

## Sample Agents

| Agent | Purpose | Model | Tools |
|-------|---------|-------|-------|
| `scout` | Fast codebase recon | Haiku | read, grep, find, ls, bash |
| `planner` | Implementation plans | Sonnet | read, grep, find, ls |
| `reviewer` | Code review | Sonnet | read, grep, find, ls, bash |
| `worker` | General-purpose | Sonnet | (all default) |

## Workflow Prompts

| Prompt | Flow |
|--------|------|
| `/implement <query>` | scout → planner → worker |
| `/scout-and-plan <query>` | scout → planner |
| `/implement-and-review <query>` | worker → reviewer → worker |

## Error Handling

- **stopReason "error"**: LLM error propagated with error message (pidrei tools signal failure by raising)
- **stopReason "aborted"**: User abort (Ctrl+C) aborts the session, raises
- **Unknown agent/model, missing cwd, no available models**: Task fails with a diagnostic message
- **Chain mode**: Stops at first failing step, reports which step failed

## Limitations

- Output truncated to last 10 items in collapsed view (expand to see all)
- Parallel model-visible output is capped at 50 KB per task; full results remain in tool details
- Agents discovered fresh on each invocation (allows editing mid-session)
- Parallel mode limited to 8 tasks; 4 run at once by default (`concurrency`
  raises or lowers that, clamped to the task cap — the sessions are cheap
  in-process, so the cap mainly guards the provider account's rate limits)
