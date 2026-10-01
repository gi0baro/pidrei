# Debugging hangs, flakes and platform differences

Techniques that have found real defects in PiDrei, for when a test hangs, a
failure only shows up on CI, or the TUI freezes. The rules they serve are in
`AGENTS.md` (tests, the TonIO contract); this is the how.

## Hangs

- **Find the test.** Locally, `-v` prints a test's name when it starts, so
  the last name on screen is the stuck one. In CI logs a line appears only
  once it is complete, so the stuck test is the one *after* the last finished
  test, in collection order (`pytest --collect-only -q` gives the order).
- **Dump the stacks.** `-o faulthandler_timeout=N` (N well under 60) dumps
  every thread's stack when a test runs too long. For a manual capture of a
  live process: `py-spy dump`, per-thread CPU from `/proc/<pid>/task/*/stat`,
  and `wchan`.
- **Read the dump.** All threads parked with no Python frames means the
  runtime is idle: something awaits an Event, channel or handle that never
  fires. If the test itself is on the stack, the test awaits something that
  never happens; look for a fake with an outdated signature, or a child that
  raised while a sibling waits.
- **Crashed coroutines are invisible by default.** TonIO reports an
  unretrieved coroutine exception as an `UNHANDLED ...` line on stdout, which
  pytest's capture swallows. Run with `-s` and grep for `UNHANDLED` when a
  detached coroutine may have died; CI already runs with `-s`.
- **Manual clocks.** A clock seeded from the real monotonic time and
  advanced by *exactly* a threshold can fall short by float rounding, and a
  loop waiting on it then never sees the threshold pass. The failure depends
  on the host's uptime, so it looks intermittent. Advance past thresholds with
  margin.

## Proving a suspected race

Rerunning a flaky test hundreds of times rarely reproduces a CI-only
failure. Force the suspected interleaving instead:

- Insert a temporary `time.sleep` at the racy point (between a listener's
  settling call and its next line, or between a check and its act) and run
  the test once. If the race is real, the failure becomes deterministic.
- For a detached piece of work suspected of running at the wrong moment,
  gate it in a throwaway test with an Event the test controls, and order it
  explicitly against the test's own steps.

Then fix the race in production code if it is one there; if the race is in
the test, synchronize the test with an Event the fake sets.

## macOS CI

macOS CI fails where Linux passes for platform reasons. Each of these has
turned a latent race into a failure:

- **Pipe reads are line-sized.** Linux returns ~64 KiB chunks from a child's
  stdout; macOS can deliver a shell loop as thousands of ~10-byte chunks, so
  any per-chunk cost is multiplied by a thousand. Reproduce on Linux by
  narrowing `SPILL_CHANNEL_SIZE` or `EXIT_STDIO_GRACE_SECONDS` in the agent
  harness's `local.py`, stalling the spill with a slow `create_temp_file`
  override (`SlowSpillExecutionEnv` in `test_local_env.py`), or patching the
  stream's `receive_some` to 16 bytes in a throwaway test.
- **Paths are long.** `$TMPDIR` is 49 bytes and `sun_path` is 104, so socket
  paths overflow. Socket tests use the `sock_dir` fixture, which budgets for
  the longest socket name; widen its budget when a longer name appears.
  Reproduce with `TMPDIR` set to a 48-byte directory.
- **Blocking-pool I/O is slow** relative to short windows, so a pool hop
  between a pinned read and a guarded write can straddle a whole concurrent
  reload. Re-check published state under the guard. Reproduce by sleeping
  inside the relevant test seam for one reader.
- **Startup windows are reachable.** Interactive startup is slow enough that
  input can arrive before the editor's submit handler is wired. Tests typing
  into the real TUI tolerate "Startup is still in progress" (`_submit_until`
  in `test_boot_smoke.py`).
- **Spawned follow-ups lose races that render waits hide.** Work spawned
  after a key or mouse event can still be pending when a frame wait returns.
  Wait on an Event the fake sets, never on the frame.
- **A second spawned caller may not have run yet.** With two spawned callers
  of shared work, "the work started" only proves the first one joined.
  Wait until every caller is counted before cancelling one.
- **A ported `void f()` reads state late.** Pi's un-awaited call runs `f`'s
  body up to its first await immediately; a spawned coroutine first runs
  after later input may have changed the state it reads. Capture the
  arguments at the event and spawn only the I/O. This is a production fix,
  not a test fix.

Read CI failures with `gh run view <id> --log-failed`, and grep the log for
`UNHANDLED`.

## TUI freezes: the pty stress driver

A frozen TUI (process alive, input dead) has two known shapes: a
long-lived loop died from an exception nothing caught, or an fd pump parked
on a readiness edge that was lost. Unit tests with fakes don't reach either.
A pty stress driver does:

- Spawn the app on a pty and play the terminal: answer the capability
  queries (Kitty-capable replies), then hammer it with key bursts, SIGWINCH
  storms and escape sequences split across writes while output streams.
- A freeze shows as output silence following keys.
- `uv run X` starts X as a child without exec, so killing `uv` orphans the
  app. Start the driver's child with `start_new_session=True` and kill it
  with `os.killpg`.

The in-repo nets are the pty tests in `packages/tui/tests/test_terminal_pty.py`
(`test_pty_input_survives_a_raising_input_handler`,
`test_pty_tui_survives_an_input_storm_with_concurrent_mutations`); extend
them, or drive the real app from a throwaway script outside the repo.
