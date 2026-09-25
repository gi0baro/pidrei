"""One task owns a mutable aggregate (concurrency audit §4.3/§4.4).

No pi counterpart: pi's single JS thread is the owner of everything — input
handlers, `setTimeout` callbacks and promise continuations never overlap.
Here the equivalents run on different tonio tasks, so the TUI's input state
(`StdinBuffer`, keyboard-protocol negotiation, `editor._state`, focus and
overlays) gets one owner task instead of a lock per site:

- `run(fn)` / `post(fn)` hand work to the owner; it runs serially, in order.
  `post` always enqueues (a post before `start()` runs at start; an owner
  that never starts never runs it — tests start one or stub the seam).
- `after(delay_ms, fn)` / `every(delay_ms, fn)` are `setTimeout`/`setInterval`
  whose fires are delivered as posted work, so a callback runs on the owner
  too. `cancel()` is exact by construction: the cancel and the fire are
  ordered on the same task, so a cancelled timer never runs — none of the
  identity re-checks the detached `_timers` needed.
- The timer tasks are children of the scope passed to `start()`, so stopping
  the owner reaps them; nothing ticks after `close()`.
- One queue for the owner's lifetime: posted work the consumer did not reach
  before a `close()` (behind the shutdown sentinel, or left by a cancelled
  consumer) runs in order after the next `start()` — the TUI suspend/resume
  and renderer switches stop and restart the same owner.

An owner that was never started (a TUI that is never `start()`ed — tests)
runs `run` work on the caller and timer fires inline; `post`ed work waits in
the queue for a `start()` that may never come.
"""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import tonio.colored as tonio
from tonio.colored.sync import channel


type Thunk = Callable[[], Awaitable[None]]


class OwnerStopped(Exception):
    """The owner stopped before running a `run()` job.

    A `run()` waiter is settled with this — never left parked — whenever the
    consumer exits for any reason: a clean `close()` whose sentinel the job
    landed behind, cancellation unwinding it, or a crash (chained as
    `__cause__`). "Processed, or the queue stopped for good" is the contract;
    a silent hang is not an outcome.
    """


@dataclass(slots=True, eq=False)  # eq=False: identity-hashed for `_pending`
class _Job:
    fn: Thunk
    done: tonio.Event | None = None
    error: BaseException | None = None


@dataclass(slots=True, eq=False)
class TimerHandle:
    """A scheduled fire; `cancel()` guarantees `fn` does not run afterwards."""

    _wake: tonio.Event = field(default_factory=tonio.Event)
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True
        self._wake.set()


class OwnerTask:
    def __init__(self, on_error: Callable[[BaseException], None] | None = None) -> None:
        self._sender, self._receiver = channel.unbounded()
        self._scope = None
        self._closed = False
        #: published (before the pending sweep) only by the consumer's
        #  shutdown path — unlike `_closed`, which a clean `close()` sets
        #  while the consumer is still draining jobs ahead of the sentinel.
        self._stopped = False
        #: the exception that killed the consumer, when it died of one.
        #  A crashed owner is an error state, not headless mode: `run` raises
        #  instead of degrading to the never-started inline fallback.
        self._crashed: BaseException | None = None
        #: `run()` jobs nobody has claimed yet. Leaving the set under the
        #  guard is the claim: the consumer takes a job to run it, a stopped
        #  owner's sweep or `run()` itself to settle it with `OwnerStopped` —
        #  never both, even when a restarted consumer meets a job a stopped
        #  one left queued.
        self._pending: set[_Job] = set()
        #: sync sections only: sends, the restart's queue swap, the claim
        #  set and the stop/generation state the shutdown sweep publishes.
        self._guard = threading.Lock()
        #: bumped by each `start()`; a consumer's shutdown path touches the
        #  owner's state only while it is still the current generation.
        self._generation = 0
        self._timers: set[TimerHandle] = set()
        # Called with an exception escaping posted work (a timer callback,
        # a fire-and-forget mutation). `None` lets it kill the owner.
        self.on_error = on_error

    @property
    def started(self) -> bool:
        return self._scope is not None and not self._closed

    @property
    def serving(self) -> bool:
        """Started and the consumer is still alive (not stopped, not crashed).

        The predicate for routing *new* work from ambient callers (the
        timers module): `started` alone reflects only what `start()`/
        `close()` recorded, so an owner abandoned without `close()` reads
        started forever even though nothing will ever drain its queue.
        """
        return self.started and not self._stopped and self._crashed is None

    def start(self, scope) -> None:
        """Run the owner loop as a child of `scope` (restartable after `close()`).

        A restart keeps the queue: work the previous consumer did not reach
        runs first, in post order. Callers restart only after the scope that
        ran the previous consumer has exited.
        """
        with self._guard:
            if self._closed:
                self._closed = False
                self._stopped = False
                self._crashed = None
                self._requeue()
            self._generation += 1
            generation = self._generation
            receiver = self._receiver
            self._scope = scope
        scope.spawn(self._consume(receiver, generation))

    def _requeue(self) -> None:
        """Carry what the stopped consumer left queued over to a fresh channel.

        A fresh channel, not the old one: a consumer unwound by cancellation
        is not known to have released its receive, and its sentinel may still
        be queued. Under `_guard`, so no send lands in the old channel after
        the move.
        """
        stale = self._receiver
        self._sender, self._receiver = channel.unbounded()
        while True:
            item = stale.receive_nowait()
            if item is stale.Empty or item is stale.Closed:
                return
            if item is not None:  # a previous generation's sentinel
                self._sender.send(item)

    def close(self) -> None:
        """Stop after the work already queued; cancel every live timer."""
        with self._guard:
            if self._closed:
                return
            self._closed = True
            if self._scope is not None:
                self._sender.send(None)
        for handle in list(self._timers):
            handle.cancel()
        self._timers.clear()

    def _send(self, item: _Job) -> None:
        with self._guard:
            self._sender.send(item)

    def post(self, fn: Thunk) -> None:
        """Run `fn` on the owner, fire-and-forget, in post order.

        Always enqueues: work posted before `start()` runs when the owner
        starts, still in order. An owner that never starts never runs it —
        tests drive a started owner or stub the posting seam; production
        owners span the terminal's lifetime. After `close()` the job waits
        behind the shutdown sentinel and runs after the next `start()`.
        """
        self._send(_Job(fn))

    async def run(self, fn: Thunk) -> None:
        """Run `fn` on the owner and wait for it; its error surfaces here.

        Never parks forever: if the owner stops before running the job — a
        `close()` sentinel it landed behind, cancellation, a consumer crash —
        the job is settled with `OwnerStopped` (crash chained as `__cause__`)
        by the consumer's shutdown path, and raised here. Only a never-started
        or cleanly-closed owner runs `fn` inline (the headless contract); a
        *crashed* owner raises instead — inline mutation on the caller's task
        after the owner died would be an ownership violation, not a fallback.
        """
        if self._crashed is not None:
            raise self._stop_error()
        if not self.started:
            await fn()
            return
        job = _Job(fn, done=tonio.Event())
        # Enrolled before the send so the consumer's shutdown sweep can never
        # miss it: either we observe the stop below, or the sweep — which
        # publishes the stop under the same guard — observes the job.
        with self._guard:
            self._pending.add(job)
            self._sender.send(job)
        with self._guard:
            # The consumer stopped between our `started` check and the send:
            # settle here instead of parking until a restart. The job stays
            # queued; a restarted consumer finds it claimed and skips it.
            stranded = self._stopped and job in self._pending
            if stranded:
                self._pending.discard(job)
        if stranded:
            job.error = self._stop_error()
            job.done.set()
        await job.done.wait(None)
        if job.error is not None:
            raise job.error

    def _stop_error(self) -> OwnerStopped:
        error = OwnerStopped()
        if self._crashed is not None:
            error.__cause__ = self._crashed
        return error

    def spawn(self, coro) -> None:
        """Run `coro` concurrently (off the owner) as a child of its scope.

        For work the owner kicks off but must not wait for — a provider
        request whose result comes back through `run`/`post`. Reaped with
        the scope instead of outliving the TUI.
        """
        if self.started:
            self._scope.spawn(coro)
        else:
            tonio.spawn.without_tracking(coro)

    def after(self, delay_ms: float, fn: Thunk) -> TimerHandle:
        """`setTimeout`: run `fn` on the owner after `delay_ms` unless cancelled."""
        return self._schedule(delay_ms, fn, repeat=False)

    def every(self, delay_ms: float, fn: Thunk) -> TimerHandle:
        """`setInterval`: run `fn` on the owner every `delay_ms` until cancelled."""
        return self._schedule(delay_ms, fn, repeat=True)

    def _schedule(self, delay_ms: float, fn: Thunk, *, repeat: bool) -> TimerHandle:
        handle = TimerHandle()
        if self._closed:
            handle.cancel()
            return handle
        self._timers.add(handle)
        timer = self._timer(handle, delay_ms / 1000, fn, repeat)
        if self._scope is not None:
            self._scope.spawn(timer)
        else:
            tonio.spawn.without_tracking(timer)
        return handle

    async def _timer(self, handle: TimerHandle, delay_s: float, fn: Thunk, repeat: bool) -> None:
        try:
            while True:
                await handle._wake.wait(delay_s)
                if handle.cancelled:
                    return
                if self.started:
                    self._send(_Job(self._fire(handle, fn)))
                elif not self._closed:
                    await fn()
                if not repeat:
                    return
        finally:
            self._timers.discard(handle)

    @staticmethod
    def _fire(handle: TimerHandle, fn: Thunk) -> Thunk:
        async def fire() -> None:
            # Ordered after any `cancel()` the owner's earlier work made.
            if not handle.cancelled:
                await fn()

        return fire

    def _claim(self, job: _Job) -> bool:
        with self._guard:
            if job not in self._pending:
                return False
            self._pending.discard(job)
            return True

    async def _consume(self, receiver, generation: int) -> None:
        crash: BaseException | None = None
        try:
            while True:
                job = await receiver.receive()
                if job is None:
                    return
                if job.done is None:
                    try:
                        await job.fn()
                    except BaseException as error:
                        # BaseException: a pyo3 PanicException escaping here
                        # would kill the owner — input and timers — silently.
                        if isinstance(error, GeneratorExit) or self.on_error is None:
                            raise
                        self.on_error(error)
                    continue
                if not self._claim(job):
                    continue  # settled `OwnerStopped` while no consumer served it
                try:
                    await job.fn()
                except BaseException as error:
                    job.error = error
                finally:
                    job.done.set()
        except BaseException as error:
            crash = error
            if isinstance(error, GeneratorExit):
                raise
            # Swallowed, not re-raised: the sweep below fully accounts for
            # the crash (`_crashed`, `OwnerStopped` settlements, `serving`
            # off). Escaping further would only reach tonio's
            # unhandled-coroutine printer on stdout — which the TUI may be
            # holding in non-blocking mode.
        finally:
            # Shutdown sweep — sync only (a cancelled child unwinds but cannot
            # await): whatever stopped this loop, no `run()` waiter is left
            # parked on a queue nobody drains. Publish the stop first, then
            # settle; `run()` enrolls before sending, so one side always sees
            # the job. Runs on the clean sentinel exit too, settling `run`
            # jobs that landed behind it. A consumer unwound by cancellation
            # may get here only after a restart: the owner is then another
            # generation's, and its state and claims are left alone.
            with self._guard:
                current = generation == self._generation
                if current:
                    if crash is not None and not isinstance(crash, GeneratorExit):
                        self._crashed = crash
                    self._closed = True
                    self._stopped = True
                    stranded = list(self._pending)
                    self._pending.clear()
                else:
                    stranded = []
            for job in stranded:
                job.error = self._stop_error()
                job.done.set()
