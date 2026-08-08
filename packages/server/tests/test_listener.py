"""Port of pi server `test/listener.test.ts`."""

import pytest

from pidrei_server.testing import TestServerOptions, create_test_server


class RecordingListener:
    def __init__(self, address, start_error=None):
        self.address = address
        self.accept = None
        self.start_count = 0
        self.close_count = 0
        self.start_error = start_error

    async def start(self, accept):
        self.start_count += 1
        self.accept = accept
        if self.start_error is not None:
            raise self.start_error

    async def close(self):
        self.close_count += 1
        self.address = None


@pytest.mark.tonio
async def test_starts_and_closes_every_configured_listener():
    first = RecordingListener("first")
    second = RecordingListener("second")
    server = create_test_server(TestServerOptions(listeners=[first, second])).server

    await server.start()
    assert server.addresses == ["first", "second"]
    assert callable(first.accept)
    assert callable(second.accept)

    await server.close()
    assert first.close_count == 1
    assert second.close_count == 1
    assert server.addresses == []


@pytest.mark.tonio
async def test_closes_previously_started_listeners_when_startup_fails():
    first = RecordingListener("first")
    failure = Exception("listener failed")
    second = RecordingListener("second", failure)
    server = create_test_server(TestServerOptions(listeners=[first, second])).server

    with pytest.raises(Exception, match="listener failed"):
        await server.start()
    assert first.close_count == 1
    assert second.close_count == 0
