"""Mirror of pi coding-agent src/utils/fs-watch.ts.

pi wraps node's ``fs.watch`` with error handling. Python has no stdlib
directory-watch primitive, so the watcher polls: it snapshots entry mtimes
and reports changes as ``(event_type, filename)`` callbacks — the same
listener shape pi consumers use (they debounce and re-read on their side,
so poll granularity only affects reload latency).

The poll is a TUI timer (`pidrei_tui._timers.Interval`): each tick runs the
scan on the blocking pool and calls the listener on the UI owner task, the
way node delivers watcher events on its one thread. Listeners therefore run
where the state they touch lives (theme reload, footer refresh), and the
watchers are reaped with the TUI's scope. A watcher needs a tonio runtime.
"""

import os

import tonio.colored as tonio

from pidrei_tui._timers import Interval


FS_WATCH_RETRY_DELAY_MS = 5000

_POLL_INTERVAL_MS = 200


class FsWatcher:
    """Polling stand-in for node's FSWatcher (directory or file watch)."""

    def __init__(self, path: str, listener, on_error) -> None:
        self._path = path
        self._listener = listener
        self._on_error = on_error
        self._is_dir = os.path.isdir(path)
        if not self._is_dir and not os.path.exists(path):
            raise OSError(f"path does not exist: {path}")
        # The baseline is taken here so that every change after the watcher
        # exists is reported (node's fs.watch has the same contract).
        self._snapshot = self._scan()
        self._closed = False
        self._scanning = False
        self._interval = Interval(_POLL_INTERVAL_MS, self._tick)

    def _scan(self) -> dict:
        if not self._is_dir:
            return {os.path.basename(self._path): os.stat(self._path).st_mtime_ns}
        entries = {}
        for name in os.listdir(self._path):
            try:
                entries[name] = os.stat(os.path.join(self._path, name)).st_mtime_ns
            except OSError:
                continue
        return entries

    async def _tick(self) -> None:
        if self._closed or self._scanning:
            return  # a scan slower than the poll interval does not pile up
        self._scanning = True
        try:
            current = await tonio.spawn_blocking(self._scan)
        except OSError:
            # Watched directory disappeared or became unreadable — mirrors
            # the watcher "error" event path.
            if not self._closed:
                self.close()
                self._on_error()
            return
        finally:
            self._scanning = False
        previous = self._snapshot
        self._snapshot = current
        if self._closed:
            return
        for name in current.keys() | previous.keys():
            if current.get(name) == previous.get(name):
                continue
            event = "change" if name in current and name in previous else "rename"
            if self._closed:
                return
            self._listener(event, name)

    def close(self) -> None:
        self._closed = True
        self._interval.cancel()


def close_watcher(watcher) -> None:
    if watcher is None:
        return
    try:
        watcher.close()
    except Exception:
        # Ignore watcher close errors
        pass


def watch_with_error_handler(path: str, listener, on_error):
    try:
        return FsWatcher(path, listener, on_error)
    except OSError:
        on_error()
        return None


# ----------------------------------------------------------------------------
# node fs.watchFile / fs.unwatchFile equivalents (stat polling on one file)
# ----------------------------------------------------------------------------

_MISSING_STAT = {"mtimeMs": 0.0, "ctimeMs": 0.0, "size": 0}


def _stat_record(path: str) -> dict:
    try:
        st = os.stat(path)
    except OSError:
        # node reports zeroed Stats for missing files
        return dict(_MISSING_STAT)
    return {"mtimeMs": st.st_mtime_ns / 1e6, "ctimeMs": st.st_ctime_ns / 1e6, "size": st.st_size}


class _FileStatPoller:
    def __init__(self, path: str, interval_ms: float, listener) -> None:
        self._path = path
        self.listener = listener
        self._previous = _stat_record(path)
        self._stopped = False
        self._polling = False
        self._interval = Interval(max(interval_ms, 1), self._tick)

    async def _tick(self) -> None:
        if self._stopped or self._polling:
            return
        self._polling = True
        try:
            current = await tonio.spawn_blocking(_stat_record, self._path)
        finally:
            self._polling = False
        previous = self._previous
        self._previous = current
        if current != previous and not self._stopped:
            self.listener(current, previous)

    def stop(self) -> None:
        self._stopped = True
        self._interval.cancel()


_file_pollers: dict = {}


def watch_file(path: str, interval_ms: float, listener) -> None:
    """node ``fs.watchFile``: poll a file's stats, call ``listener(curr, prev)``."""
    _file_pollers.setdefault(path, []).append(_FileStatPoller(path, interval_ms, listener))


def unwatch_file(path: str, listener=None) -> None:
    """node ``fs.unwatchFile``: remove one listener, or all for the path."""
    pollers = _file_pollers.get(path, [])
    remaining = []
    for poller in pollers:
        if listener is None or poller.listener is listener:
            poller.stop()
        else:
            remaining.append(poller)
    if remaining:
        _file_pollers[path] = remaining
    else:
        _file_pollers.pop(path, None)
