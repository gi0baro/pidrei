# Examples

Working extensions, runnable as they are:

```bash
pidrei -e <path-to-example>
```

## Extensions

`plan_mode` is the one to read if you only read one. It is also a directory
extension, so it shows the `__init__.py` layout and relative imports.

### Getting started

| Example | Shows |
|---------|-------|
| [`extensions/hello.py`](extensions/hello.py) | Minimal custom tool — `ToolDefinition`, `AgentToolResult` |
| [`extensions/commands.py`](extensions/commands.py) | `pi.get_commands()`, argument completions, select/confirm dialogs |
| [`extensions/input_transform.py`](extensions/input_transform.py) | The `input` event — transform, handle, or pass through |
| [`extensions/input_transform_streaming.py`](extensions/input_transform_streaming.py) | Rewriting user input; streaming output back |
| [`extensions/inline_bash.py`](extensions/inline_bash.py) | Expanding `!{command}` in prompts via `pi.exec` |
| [`extensions/pirate.py`](extensions/pirate.py) | Toggling a system-prompt append from a command via `before_agent_start` |
| [`extensions/system_prompt_header.py`](extensions/system_prompt_header.py) | `ctx.get_system_prompt()` and a status entry via `ctx.ui.set_status` |
| [`extensions/claude_rules.py`](extensions/claude_rules.py) | Scanning project files with `tonio.colored.fs` into the system prompt |
| [`extensions/prompt_customizer.py`](extensions/prompt_customizer.py) | Reading `systemPromptOptions` in `before_agent_start` to tailor the system prompt |

### Tools

| Example | Shows |
|---------|-------|
| [`extensions/dynamic_tools.py`](extensions/dynamic_tools.py) | Registering tools after startup — one at `session_start`, more at runtime from a command |
| [`extensions/tools.py`](extensions/tools.py) | A `/tools` toggle dialog with `SettingsList`, persisted per branch and restored on `session_tree` navigation |
| [`extensions/kimi_deferred_tools.py`](extensions/kimi_deferred_tools.py) | Deferred tool loading: a `tool_search` tool that activates other tools via `set_active_tools` |
| [`extensions/built_in_tool_renderer.py`](extensions/built_in_tool_renderer.py) | Re-registering built-in tools with delegated execution and compact custom renderers |
| [`extensions/minimal_mode.py`](extensions/minimal_mode.py) | Overriding all built-in tools with minimal renderers: call-only collapsed, full output expanded |
| [`extensions/tool_override.py`](extensions/tool_override.py) | Overriding the built-in `read` tool for auditing and access control |
| [`extensions/truncated_tool.py`](extensions/truncated_tool.py) | Proper output truncation for custom tools: `truncate_head`, byte/line limits, temp-file spillover |
| [`extensions/structured_output.py`](extensions/structured_output.py) | `terminate=True` so the agent ends on a tool call without a follow-up LLM turn |
| [`extensions/question.py`](extensions/question.py) | A custom tool with a full `ctx.ui.custom` UI: option list, inline `Editor`, sequential execution mode |
| [`extensions/questionnaire.py`](extensions/questionnaire.py) | Single- and multi-question flows in one custom component: tab bar, per-question answers, Submit tab |
| [`extensions/bash_spawn_hook.py`](extensions/bash_spawn_hook.py) | `spawn_hook=` on the bash tool — adjusting command, cwd, and env before every execution |
| [`extensions/ssh.py`](extensions/ssh.py) | Delegating read/write/edit/bash to a remote host via `operations=` overrides and `user_bash` |
| [`extensions/preset.py`](extensions/preset.py) | Named presets switching model, thinking level, tools and system prompt; flag, command, shortcut |

### Guards and permissions

| Example | Shows |
|---------|-------|
| [`extensions/permission_gate.py`](extensions/permission_gate.py) | Gating dangerous bash commands behind a `tool_call` confirmation dialog |
| [`extensions/protected_paths.py`](extensions/protected_paths.py) | Blocking write/edit tool calls on protected paths |
| [`extensions/confirm_destructive.py`](extensions/confirm_destructive.py) | Cancelling session clear/switch/fork via the `session_before_*` events |
| [`extensions/dirty_repo_guard.py`](extensions/dirty_repo_guard.py) | Guarding session changes behind a `git status` check with `pi.exec` |
| [`extensions/project_trust.py`](extensions/project_trust.py) | Deciding project trust from the `project_trust` event |
| [`extensions/timed_confirm.py`](extensions/timed_confirm.py) | Timed dialogs: the `timeout` option and the manual `CancelToken` approach |

### Sessions and messaging

| Example | Shows |
|---------|-------|
| [`extensions/trigger_compact.py`](extensions/trigger_compact.py) | Triggering compaction from an event handler |
| [`extensions/custom_compaction.py`](extensions/custom_compaction.py) | Replacing default compaction via `session_before_compact`, summarizing on a different model |
| [`extensions/bookmark.py`](extensions/bookmark.py) | `pi.set_label` / `session_manager.get_label` to bookmark entries for `/tree` navigation |
| [`extensions/session_name.py`](extensions/session_name.py) | `set_session_name`/`get_session_name` for friendly names in the session selector |
| [`extensions/send_user_message.py`](extensions/send_user_message.py) | `pi.send_user_message` with steer/followUp delivery and structured content |
| [`extensions/handoff.py`](extensions/handoff.py) | Serializing the branch, generating a handoff prompt, carrying it into a new session |
| [`extensions/summarize.py`](extensions/summarize.py) | Off-loop completion on a fixed model, rendered in a custom bordered Markdown component |
| [`extensions/qna.py`](extensions/qna.py) | The "prompt generator" pattern: loader UI, `model_registry.complete`, result into the editor |
| [`extensions/todo.py`](extensions/todo.py) | Tool state stored in tool-result details, rebuilt from `get_branch()` so branching restores it |
| [`extensions/tic_tac_toe.py`](extensions/tic_tac_toe.py) | Sequential tools, a full-screen game component, custom message renderers |
| [`extensions/entry_renderer.py`](extensions/entry_renderer.py) | Durable custom session entries with `pi.append_entry` + `register_entry_renderer` |
| [`extensions/message_renderer.py`](extensions/message_renderer.py) | `register_message_renderer` with `expanded`/`outputPad` options |
| [`extensions/subagent/`](extensions/subagent/) | Delegating to isolated `pidrei --mode json` subprocesses — single/parallel/chain modes |
| [`extensions/event_bus.py`](extensions/event_bus.py) | `pi.events` for extension-to-extension communication |
| [`extensions/file_trigger.py`](extensions/file_trigger.py) | A background tonio polling task injecting external file content as a turn-triggering message |
| [`extensions/shutdown_command.py`](extensions/shutdown_command.py) | `ctx.shutdown()` from a command and from LLM-callable tools, with streamed progress |
| [`extensions/reload_runtime.py`](extensions/reload_runtime.py) | `ctx.reload()` from a command; a tool queuing a follow-up command via `send_user_message` |

### Git

| Example | Shows |
|---------|-------|
| [`extensions/git_merge_and_resolve.py`](extensions/git_merge_and_resolve.py) | `pi.exec`, follow-up messages, keeping blocking I/O off the event loop |
| [`extensions/auto_commit_on_exit.py`](extensions/auto_commit_on_exit.py) | `session_shutdown` cleanup work; reading the last assistant message from session entries |
| [`extensions/git_checkpoint.py`](extensions/git_checkpoint.py) | Per-turn git stash checkpoints; `session_before_fork` with a restore prompt |

### UI

| Example | Shows |
|---------|-------|
| [`extensions/custom_header.py`](extensions/custom_header.py) | Replacing the built-in header via `ctx.ui.set_header()` with a themed component |
| [`extensions/custom_footer.py`](extensions/custom_footer.py) | A custom footer: token stats from the session, git branch from the footer data provider |
| [`extensions/status_line.py`](extensions/status_line.py) | Persistent footer status text via `ctx.ui.set_status()`, updated per turn |
| [`extensions/model_status.py`](extensions/model_status.py) | Reacting to the `model_select` event |
| [`extensions/hidden_thinking_label.py`](extensions/hidden_thinking_label.py) | Customizing the collapsed-thinking label |
| [`extensions/titlebar_spinner.py`](extensions/titlebar_spinner.py) | Animating the terminal title from a cancellable tonio task |
| [`extensions/widget_placement.py`](extensions/widget_placement.py) | The `placement` option of `ctx.ui.set_widget()` (above vs. below the editor) |
| [`extensions/border_status_editor.py`](extensions/border_status_editor.py) | Custom editor borders as a status line; suppressing footer and working indicator |
| [`extensions/modal_editor.py`](extensions/modal_editor.py) | Subclassing `CustomEditor` for vim-like modal input handling |
| [`extensions/rainbow_editor.py`](extensions/rainbow_editor.py) | Editor render post-processing with an animated highlight |
| [`extensions/working_indicator.py`](extensions/working_indicator.py) | `set_working_indicator` frames/intervals, switched at runtime by a command |
| [`extensions/working_message_test.py`](extensions/working_message_test.py) | `set_working_message` / `set_working_indicator` persisting across loader recreations |
| [`extensions/interactive_shell.py`](extensions/interactive_shell.py) | Intercepting `user_bash` to run interactive commands (vim, htop) with the TUI suspended |
| [`extensions/overlay_test.py`](extensions/overlay_test.py) | A focusable overlay with inline text inputs and wide-char/emoji compositing edge cases |
| [`extensions/overlay_qa_tests.py`](extensions/overlay_qa_tests.py) | The whole overlay API: anchors, margins, percent positioning, stacking, `OverlayHandle` |
| [`extensions/snake.py`](extensions/snake.py) | A game loop on `Interval`, render caching, pause/resume persisted through session entries |
| [`extensions/space_invaders.py`](extensions/space_invaders.py) | Kitty keyboard protocol key-release handling for smooth held-key movement |
| [`extensions/notify.py`](extensions/notify.py) | Native terminal notifications on `agent_end` via OSC 777/99 and a Windows toast |
| [`extensions/mac_system_theme.py`](extensions/mac_system_theme.py) | Following the macOS system appearance with `ctx.ui.set_theme` from a polling task |
| [`extensions/github_issue_autocomplete.py`](extensions/github_issue_autocomplete.py) | `ctx.ui.add_autocomplete_provider` and `fuzzy_filter` — `#` completes against `gh` issues |
| [`extensions/rpc_demo.py`](extensions/rpc_demo.py) | Every RPC-forwardable UI call — dialogs, statuses, widgets, title, editor prefill |

### Providers and resources

| Example | Shows |
|---------|-------|
| [`extensions/custom_provider_anthropic/`](extensions/custom_provider_anthropic/) | Registering a custom provider: OAuth login/refresh, env API key, custom api-id dispatch |
| [`extensions/custom_provider_gitlab_duo/`](extensions/custom_provider_gitlab_duo/) | A gateway provider: per-request token exchange, dispatch to Anthropic/OpenAI wire implementations |
| [`extensions/provider_payload.py`](extensions/provider_payload.py) | Logging (or replacing) provider request payloads off the event loop |
| [`extensions/dynamic_resources/`](extensions/dynamic_resources/) | Contributing a skill, prompt template, and theme via `resources_discover` |

### The full tour

| Example | Shows |
|---------|-------|
| [`extensions/plan_mode/`](extensions/plan_mode/) | Flags, commands, shortcuts, tool gating, context filtering, widgets, custom message types — the widest use of the API |

## Writing your own

[docs/extensions.md](../docs/extensions.md) documents the whole API. The short
version:

```python
def extension(pi):
    async def handle(_args, ctx):
        ctx.ui.notify("hello", "info")

    pi.register_command("hello", handler=handle, description="Say hello")
```

Save as `.pidrei/extensions/hello.py` in a project and it loads on next start.

## A note on style

pi keeps each example in a single factory closure, which is the JavaScript
idiom. These follow that where it stays readable, and use a class where it does
not — `plan_mode` is a `PlanMode` object with a `wire()` method, because a
closure holding that much state reads worse in Python and is harder to test.
Both forms are equally valid extensions.
