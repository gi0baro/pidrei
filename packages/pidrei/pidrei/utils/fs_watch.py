"""Mirror of pi coding-agent src/utils/fs-watch.ts.

pi wraps node's ``fs.watch`` with error handling. Python has no stdlib
directory-watch primitive, so the watcher is a polling daemon thread that
snapshots entry mtimes and reports changes as ``(event_type, filename)``
callbacks — the same listener shape pi consumers use (they debounce and
re-read on their side, so poll granularity only affects reload latency).
"""

import os
import threading


FS_WATCH_RETRY_DELAY_MS = 5000

_POLL_INTERVAL_SECONDS = 0.2


class FsWatcher:
    """Polling stand-in for node's FSWatcher (directory or file watch)."""

    def __init__(self, path: str, listener, on_error) -> None:
        self._path = path
        self._listener = listener
        self._on_error = on_error
        self._is_dir = os.path.isdir(path)
        if not self._is_dir and not os.path.exists(path):
            raise OSError(f"path does not exist: {path}")
        self._stop = threading.Event()
        self._snapshot = self._scan()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

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

    def _run(self) -> None:
        while not self._stop.wait(_POLL_INTERVAL_SECONDS):
            try:
                current = self._scan()
            except OSError:
                # Watched directory disappeared or became unreadable —
                # mirrors the watcher "error" event path.
                if not self._stop.is_set():
                    self._on_error()
                return
            previous = self._snapshot
            self._snapshot = current
            for name in current.keys() | previous.keys():
                if current.get(name) == previous.get(name):
                    continue
                event = "change" if name in current and name in previous else "rename"
                if self._stop.is_set():
                    return
                self._listener(event, name)

    def close(self) -> None:
        self._stop.set()


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
        self._interval = max(interval_ms, 1) / 1000
        self.listener = listener
        self._stop = threading.Event()
        self._previous = _stat_record(path)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            current = _stat_record(self._path)
            previous = self._previous
            self._previous = current
            if current != previous and not self._stop.is_set():
                self.listener(current, previous)

    def stop(self) -> None:
        self._stop.set()


_file_pollers_lock = threading.Lock()
_file_pollers: dict = {}


def watch_file(path: str, interval_ms: float, listener) -> None:
    """node ``fs.watchFile``: poll a file's stats, call ``listener(curr, prev)``."""
    with _file_pollers_lock:
        _file_pollers.setdefault(path, []).append(_FileStatPoller(path, interval_ms, listener))


def unwatch_file(path: str, listener=None) -> None:
    """node ``fs.unwatchFile``: remove one listener, or all for the path."""
    with _file_pollers_lock:
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
