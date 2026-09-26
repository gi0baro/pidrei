"""Harness for interactive-mode handlers that spawn their flows.

A handler does pi's part before its first await synchronously and spawns the
rest with `InteractiveMode._spawn_flow`. A fake context records the spawned
flows with `SpawnedFlows`, and the test awaits them where pi's test awaits the
handler's promise.
"""


class SpawnedFlows(list):
    """Stands in for `InteractiveMode._spawn_flow` on a fake context."""

    def __call__(self, flow) -> None:
        self.append(flow)

    async def finish(self) -> None:
        """Run the recorded flows (and any they spawn) to completion, in order."""
        while self:
            await self.pop(0)
