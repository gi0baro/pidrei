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

An owner that was never started (a TUI that is never `start()`ed — tests)
runs `run` work on the caller and timer fires inline; `post`ed work waits in
the queue for a `start()` that may never come.
"""

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
        self._pending: set[_Job] = set()
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
        """Run the owner loop as a child of `scope` (restartable after `close()`)."""
        if self._closed:
            self._sender, self._receiver = channel.unbounded()
            self._closed = False
            self._stopped = False
            self._crashed = None
        self._scope = scope
        scope.spawn(self._consume(self._receiver))

    def close(self) -> None:
        """Stop after the work already queued; cancel every live timer."""
        if self._closed:
            return
        self._closed = True
        for handle in list(self._timers):
            handle.cancel()
        self._timers.clear()
        if self._scope is not None:
            self._sender.send(None)

    def post(self, fn: Thunk) -> None:
        """Run `fn` on the owner, fire-and-forget, in post order.

        Always enqueues: work posted before `start()` runs when the owner
        starts, still in order. An owner that never starts never runs it —
        tests drive a started owner or stub the posting seam; production
        owners span the terminal's lifetime. After `close()` the job lands
        behind the shutdown sentinel and is dropped with the channel.
        """
        self._sender.send(_Job(fn))

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
        # runs after the stop is published — observes the job.
        self._pending.add(job)
        self._sender.send(job)
        if self._stopped and not job.done.is_set():
            # The consumer stopped between our `started` check and the send;
            # nothing will drain the channel again. (The sweep may settle the
            # job concurrently — both sides write the same outcome.)
            self._pending.discard(job)
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
                    self._sender.send(_Job(self._fire(handle, fn)))
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

    async def _consume(self, receiver) -> None:
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
                try:
                    await job.fn()
                except BaseException as error:
                    job.error = error
                finally:
                    self._pending.discard(job)
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
            # jobs that landed behind it.
            if crash is not None and not isinstance(crash, GeneratorExit):
                self._crashed = crash
            self._closed = True
            self._stopped = True
            for job in list(self._pending):
                self._pending.discard(job)
                if job.error is None:
                    job.error = self._stop_error()
                job.done.set()
