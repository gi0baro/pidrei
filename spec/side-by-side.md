# Example: side-by-side responsiveness comparison

*This is an example of how PiDrei's responsiveness can be compared with
Pi's by eye, not a required step. It has been used before landing changes to
hot paths (the render loop, the streaming path), where a regression shows up
as feel rather than as a failing test. Adapt it, or use something else, as
the change calls for.*

The idea: run Pi and PiDrei next to each other on the same task, with a
model fast enough that the *client* is the bottleneck, and watch where one
lags.

## Principles

- **Use a very fast model.** Pick a Cerebras- or Groq-served model on
  OpenRouter (e.g. `openai/gpt-oss-120b`, roughly 1500–3000 tokens/s). At
  that throughput the TUI renderer and the event loop become the bottleneck,
  which is what is being compared.
- **Pin the provider.** OpenRouter's routing variance (which backend, its
  load minute to minute) would otherwise swamp any difference between the
  two clients. `:nitro` helps, but explicit provider pinning keeps both runs
  on the same backend.
- **Identical sandboxes.** Seed one directory, copy it, so both agents work
  on byte-identical data.
- **A near-deterministic tool sequence.** The prompt is numbered steps, one
  tool call at a time, so both runs do roughly the same work.

## What to watch, most useful first

1. **Input latency during streaming.** Scroll or hold a key in both TUIs
   mid-stream; the laggy one is obvious within seconds. This is the sharpest
   discriminator and the best probe of the runtime's responsiveness.
2. **Stream smoothness.** The long markdown and code-block output at
   2000 tokens/s is where renderers stutter.
3. **Large tool-output rendering.** The 8,000-line file read: does the TUI
   freeze while ingesting and collapsing it?
4. **Tool-call turnaround.** The dead gap between a tool result landing and
   the next tokens streaming.
5. **Wall time.** Last, since it is the noisiest. If you want a number,
   `time` the run and note the time to first token.

## Setup

Seed one sandbox deterministically (no `$RANDOM`, so it is reproducible
across machines), then copy it:

```bash
mkdir -p /tmp/bench-pi/data
for i in $(seq -w 1 12); do
  { echo "id,value"; for j in $(seq 1 200); do echo "$j,$(( (10#$i * 7919 + j * 104729) % 100 ))"; done; } \
    > /tmp/bench-pi/data/f$i.csv
done
seq 1 8000 > /tmp/bench-pi/data/big.txt
cp -r /tmp/bench-pi /tmp/bench-pidrei
```

`10#$i` forces base 10 for the zero-padded index (`08` and `09` would
otherwise be invalid octal). Bash or zsh.

## The prompt

Start each client in its own sandbox (`/tmp/bench-pi` and
`/tmp/bench-pidrei`) and paste the same prompt into both:

> Do exactly the following, step by step, one tool call at a time (no
> parallel calls):
> 1. List the files in ./data.
> 2. Read each of the 12 CSV files, one read per file.
> 3. Read ./data/big.txt in full.
> 4. For each CSV file, compute the sum of the `value` column with an awk
>    one-liner in bash.
> 5. Write `summary.md` containing a markdown table of filename → sum.
> 6. `cat summary.md` to verify.
> 7. Finally, write out a full report in your response: the table again, a
>    short analysis, and a complete ~100-line Python script in a code block
>    that reproduces the computation.

That is about 28 predictable tool calls mixing file reads, bash and a write
(agent-loop and I/O overhead), one large tool output (renderer ingest), and
a long uninterrupted stream at the end (renderer throughput).

## Running it

1. Two terminal panes, one client each, each started in its sandbox.
2. Same model, same pinned provider, temperature 0 if configurable.
3. Paste the prompt into both and press Enter at the same moment.
4. While they stream, do the scroll and keypress test in each pane.

## Variance

- **Throw away the first run of each client**: it pays for TLS and
  connection warm-up.
- **Run two or three times, alternating which client goes first**: provider
  load drifts minute to minute.
- Both clients hitting the same provider at once is fine (that is the point
  of side by side), but if results look odd, also try sequential runs to
  rule out rate limiting.
- Re-seed the sandboxes between runs (`rm -rf /tmp/bench-pi*`, then Setup
  again) so a leftover `summary.md` doesn't change the next run.
