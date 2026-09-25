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
- `close()` cancels every live timer handle, which ends its task; nothing
  ticks after `close()`.
- One queue and one consumer for the owner's lifetime: the first `start()`
  spawns the consumer (later calls do nothing), so posted work keeps being
  served across the TUI's suspend/resume and renderer switches. `close()` is
  final: it closes the queue from the sender side, and the consumer drains it
  and returns.

An owner that was never started (a TUI that is never `start()`ed — tests)
runs `run` work on the caller and timer fires inline; `post`ed work waits in
the queue for a `start()` that may never come.
"""

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import tonio.colored as tonio
from tonio.colored.sync import channel


type Thunk = Callable[[], Awaitable[None]]


class OwnerStopped(Exception):
    """The owner stopped before running a `run()` job.

    A `run()` waiter is settled with this — never left parked — whenever the
    consumer exits before running the job: a `close()` the job raced, or a
    crash (chained as `__cause__`). "Processed, or the queue stopped for
    good" is the contract;
    a silent hang is not an outcome.
    """


@dataclass(slots=True, eq=False)  # eq=False: identity-hashed for `_pending`
class _Job:
    fn: Thunk
    done: tonio.Event | None = None
    error: BaseException | None = None


@dataclass(slots=True, eq=False)
class _ConsumerState:
    """The part of an `OwnerTask` its consumer works with — all it holds.

    Not the owner: the consumer never reaches the sender, the timers or
    whatever the owner's users hang on it. The owner holds this; nothing
    here points back.
    """

    receiver: Any
    # Called with an exception escaping posted work (a timer callback,
    # a fire-and-forget mutation). `None` lets it kill the owner.
    on_error: Callable[[BaseException], None] | None
    #: sync sections only: the claim set and the stop state the shutdown
    #  sweep publishes.
    guard: threading.Lock = field(default_factory=threading.Lock)
    #: `run()` jobs nobody has claimed yet. Leaving the set under the
    #  guard is the claim: the consumer takes a job to run it, the
    #  shutdown sweep or `run()` itself to settle it with `OwnerStopped` —
    #  never both.
    pending: set[_Job] = field(default_factory=set)
    closed: bool = False
    #: published (before the pending sweep) only by the consumer's
    #  shutdown path — unlike `closed`, which `close()` sets while the
    #  consumer is still draining the queue.
    stopped: bool = False
    #: the exception that killed the consumer, when it died of one.
    #  A crashed owner is an error state, not headless mode: `run` raises
    #  instead of degrading to the never-started inline fallback.
    crashed: BaseException | None = None
    #: set by the first `start()`, which spawns the consumer.
    spawned: bool = False
    #: set when the consumer returns.
    done: tonio.Event = field(default_factory=tonio.Event)

    def claim(self, job: _Job) -> bool:
        with self.guard:
            if job not in self.pending:
                return False
            self.pending.discard(job)
            return True

    def stop_error(self) -> OwnerStopped:
        error = OwnerStopped()
        if self.crashed is not None:
            error.__cause__ = self.crashed
        return error


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
        self._sender, receiver = channel.unbounded()
        self._state = _ConsumerState(receiver, on_error)
        self._timers: set[TimerHandle] = set()

    @property
    def on_error(self) -> Callable[[BaseException], None] | None:
        return self._state.on_error

    @on_error.setter
    def on_error(self, handler: Callable[[BaseException], None] | None) -> None:
        self._state.on_error = handler

    @property
    def started(self) -> bool:
        return self._state.spawned and not self._state.closed

    @property
    def serving(self) -> bool:
        """Started and the consumer is still alive (not stopped, not crashed).

        The predicate for routing *new* work from ambient callers (the
        timers module): `started` alone reflects only what `start()`/
        `close()` recorded, so an owner abandoned without `close()` reads
        started forever even though nothing will ever drain its queue.
        """
        state = self._state
        return self.started and not state.stopped and state.crashed is None

    def start(self) -> None:
        """Start the consumer, which serves the queue until `close()`.

        A TUI restarting (suspend/resume, renderer switch) calls this again:
        it does nothing, the consumer and the queue carry on.
        """
        state = self._state
        if not state.spawned:
            state.spawned = True
            tonio.spawn.without_tracking(self._consume(state))

    def close(self) -> None:
        """Final: stop after the work already queued; cancel every live timer.

        Closes the queue from the sender side: the consumer drains it, then
        its receive raises and it returns (`join()` waits for that).
        """
        state = self._state
        with state.guard:
            if state.closed:
                return
            state.closed = True
        self._sender.close()
        for handle in list(self._timers):
            handle.cancel()
        self._timers.clear()

    async def join(self) -> None:
        """Wait for the consumer to return (after `close()`, or a crash)."""
        state = self._state
        if state.spawned:
            await state.done.wait(None)

    def _send(self, item: _Job) -> bool:
        try:
            self._sender.send(item)
        except BrokenPipeError:
            return False  # closed for good
        return True

    def post(self, fn: Thunk) -> None:
        """Run `fn` on the owner, fire-and-forget, in post order.

        Always enqueues: work posted before `start()` runs when the owner
        starts, still in order. An owner that never starts never runs it —
        tests drive a started owner or stub the posting seam; production
        owners span the app's lifetime. After `close()` it is dropped.
        """
        self._send(_Job(fn))

    async def run(self, fn: Thunk) -> None:
        """Run `fn` on the owner and wait for it; its error surfaces here.

        Never parks forever: if the owner stops before running the job — a
        `close()` it raced, a consumer crash — the job is settled with
        `OwnerStopped` (crash chained as `__cause__`) by the consumer's
        shutdown path or here, and raised here. Only a never-started or
        closed owner runs `fn` inline (the headless contract); a
        *crashed* owner raises instead — inline mutation on the caller's task
        after the owner died would be an ownership violation, not a fallback.
        """
        state = self._state
        if state.crashed is not None:
            raise state.stop_error()
        if not self.started:
            await fn()
            return
        job = _Job(fn, done=tonio.Event())
        # Enrolled before the send so the consumer's shutdown sweep can never
        # miss it: either we observe the stop below, or the sweep — which
        # publishes the stop under the same guard — observes the job.
        with state.guard:
            state.pending.add(job)
        sent = self._send(job)
        with state.guard:
            # Closed or stopped between our `started` check and the send:
            # nothing will run it, so settle it here.
            stranded = (not sent or state.stopped) and job in state.pending
            if stranded:
                state.pending.discard(job)
        if stranded:
            job.error = state.stop_error()
            job.done.set()
        await job.done.wait(None)
        if job.error is not None:
            raise job.error

    def spawn(self, coro) -> None:
        """Run `coro` concurrently (off the owner), fire-and-forget.

        For work the owner kicks off but must not wait for — a provider
        request whose result comes back through `run`/`post`.
        """
        tonio.spawn.without_tracking(coro)

    def after(self, delay_ms: float, fn: Thunk) -> TimerHandle:
        """`setTimeout`: run `fn` on the owner after `delay_ms` unless cancelled."""
        return self._schedule(delay_ms, fn, repeat=False)

    def every(self, delay_ms: float, fn: Thunk) -> TimerHandle:
        """`setInterval`: run `fn` on the owner every `delay_ms` until cancelled."""
        return self._schedule(delay_ms, fn, repeat=True)

    def _schedule(self, delay_ms: float, fn: Thunk, *, repeat: bool) -> TimerHandle:
        handle = TimerHandle()
        if self._state.closed:
            handle.cancel()
            return handle
        self._timers.add(handle)
        # Ended by its handle (`cancel()`, or `close()` cancelling them all).
        tonio.spawn.without_tracking(self._timer(handle, delay_ms / 1000, fn, repeat))
        return handle

    async def _timer(self, handle: TimerHandle, delay_s: float, fn: Thunk, repeat: bool) -> None:
        try:
            while True:
                await handle._wake.wait(delay_s)
                if handle.cancelled:
                    return
                if self.started:
                    self._send(_Job(self._fire(handle, fn)))
                elif not self._state.closed:
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

    @staticmethod
    async def _consume(state: _ConsumerState) -> None:
        crash: BaseException | None = None
        try:
            while True:
                try:
                    job = await state.receiver.receive()
                except BrokenPipeError:
                    return  # closed from the sender side, and drained
                if job.done is None:
                    try:
                        await job.fn()
                    except BaseException as error:
                        # BaseException: a pyo3 PanicException escaping here
                        # would kill the owner — input and timers — silently.
                        on_error = state.on_error
                        if on_error is None:
                            raise
                        on_error(error)
                    continue
                if not state.claim(job):
                    continue  # already settled by `run()` (it raced a close)
                try:
                    await job.fn()
                except BaseException as error:
                    job.error = error
                finally:
                    job.done.set()
        except BaseException as error:
            crash = error
            # Swallowed, not re-raised: the sweep below fully accounts for
            # the crash (`_crashed`, `OwnerStopped` settlements, `serving`
            # off). Escaping further would only reach tonio's
            # unhandled-coroutine printer on stdout — which the TUI may be
            # holding in non-blocking mode.
        finally:
            # Shutdown sweep — sync: whatever stopped this loop (the close,
            # a crash), no `run()` waiter is left parked on a queue nobody
            # drains. Publish the stop first, then settle; `run()` enrolls
            # before sending, so one side always sees the job.
            with state.guard:
                if crash is not None:
                    state.crashed = crash
                state.closed = True
                state.stopped = True
                stranded = list(state.pending)
                state.pending.clear()
            for job in stranded:
                job.error = state.stop_error()
                job.done.set()
            state.done.set()
