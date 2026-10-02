"""Loader that can be cancelled with Escape (port of pi tui ``components/cancellable-loader.ts``).

Extends Loader with a cancel token for cancelling async operations. pi uses a
DOM ``AbortController``; pidrei uses the shared ``pidrei_utils.cancel``
token, so loader signals go straight to the ai layer.

Example::

    loader = CancellableLoader(tui, cyan, dim, "Working...")
    loader.on_abort = lambda: done(None)
    do_work(loader.signal)
"""

from pidrei_utils.cancel import CancelToken

from ..keybindings import get_keybindings
from .loader import Loader


__all__ = ["CancellableLoader"]


class CancellableLoader(Loader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._token = CancelToken()
        # Called when user presses Escape
        self.on_abort = None

    @property
    def signal(self) -> CancelToken:
        """Token that is cancelled when user presses Escape."""
        return self._token

    @property
    def aborted(self) -> bool:
        """Whether the loader was aborted."""
        return self._token.cancelled

    def handle_input(self, data: str) -> None:
        kb = get_keybindings()
        if kb.matches(data, "tui.select.cancel"):
            self._token.cancel()
            if self.on_abort is not None:
                self.on_abort()

    def dispose(self) -> None:
        self.stop()
