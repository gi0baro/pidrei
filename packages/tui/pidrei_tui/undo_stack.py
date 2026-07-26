"""Generic undo stack with clone-on-push semantics (port of pi tui ``undo-stack.ts``).

Stores deep copies of state snapshots. Popped snapshots are returned
directly (no re-cloning) since they are already detached.
"""

import copy


__all__ = ["UndoStack"]


class UndoStack:
    __slots__ = ("_stack",)

    def __init__(self) -> None:
        self._stack: list = []

    def push(self, state) -> None:
        """Push a deep copy of the given state onto the stack."""
        self._stack.append(copy.deepcopy(state))

    def pop(self):
        """Pop and return the most recent snapshot, or None if empty."""
        return self._stack.pop() if self._stack else None

    def clear(self) -> None:
        """Remove all snapshots."""
        self._stack.clear()

    @property
    def length(self) -> int:
        return len(self._stack)
