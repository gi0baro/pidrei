# CLI integration

Run with a terminal on both ends, `pidrei` opens the TUI. When stdin or stdout is
piped or redirected it switches to print mode, and scripts can also pick print
or JSON mode explicitly.

Every mode runs the same agent, sessions, resources, and tools. The mode only
decides how input arrives and how output leaves. Model, tools, resources, and
session persistence are still chosen by the usual flags — see
[Command line](cli.md).

To skip the process boundary and drive the agent from Python instead, see
[Library use](sdk.md).

## Choose a mode

| Mode | Output | Lifetime | Use it when |
|------|--------|----------|-------------|
| Interactive | Terminal UI | Until the user quits | A person is at the keyboard |
| Print | Final text on stdout | One invocation | A script needs the final answer |
| JSON | JSONL events on stdout | One invocation | A program needs structured progress |

## Print to stdout

```bash
pidrei -p "Summarize the changes in this repository"
```

Print mode runs the prompts, writes the final assistant text to stdout, and
exits. Intermediate events are not shown, which suits command substitution,
pipelines, and one-shot jobs. Errors go to stderr, and a final response that
stopped with `error` or `aborted` exits nonzero.

Without an explicit mode, a non-TTY stdin or stdout selects print mode too, so
`git diff | pidrei "Review this"` needs no `-p`.

## Stream JSON events

```bash
pidrei --mode json "Review this repository" > events.jsonl
```

JSON mode writes the session header and then every agent and session event, one
JSON object per line. It is an event stream — not a single JSON result, and
not a constraint on what the model writes. Fields are camelCase, as in the
session files ([message-types.md](message-types.md)).

- All prompts are given at startup. The process streams that run and exits; it
  takes no further input.
- A failed or aborted response shows up in the stream but does not by itself
  exit nonzero; inspect the events when success matters. An invocation that
  raises still exits nonzero.
- `message_update` records carry deltas (`assistantMessageEvent`) and the
  running usage, not a growing snapshot. Build live output from the deltas,
  then replace it with the authoritative message from `message_end`.
- `agent_end` can be followed by automatic retries or queued work;
  `agent_settled` marks the end of automatic work for the run.
- Stdout is reserved for JSONL; diagnostics and logging go to stderr.

Streaming just the text:

```bash
pidrei --mode json "Explain this repo" \
  | jq -rj 'select(.type == "message_update") | .assistantMessageEvent
            | select(.type == "text_delta") | .delta'
```
