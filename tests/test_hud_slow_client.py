"""One stalled HUD client must not slow the stream for the others.

Measured on the host: with a client that stopped reading, a healthy client got
0.47 updates/s instead of ~4.3. Sends are now concurrent and time-limited.
"""

import asyncio
import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import core.web_bridge as wb


class FakeWS:
    def __init__(self, stall=False, fail=False):
        self.stall, self.fail, self.closed, self.got = stall, fail, False, []

    async def send_str(self, data):
        if self.fail:
            raise ConnectionResetError("gone")
        if self.stall:
            await asyncio.sleep(3600)
        self.got.append(data)

    async def close(self):
        self.closed = True


def test_a_stalled_client_is_dropped_and_others_are_served_promptly(monkeypatch):
    monkeypatch.setattr(wb, "WS_SEND_TIMEOUT_SEC", 0.2)
    bridge = wb.KaiaWebBridge.__new__(wb.KaiaWebBridge)
    healthy, stalled, broken = FakeWS(), FakeWS(stall=True), FakeWS(fail=True)
    bridge.ws_clients = {healthy, stalled, broken}

    async def run():
        loop = asyncio.get_running_loop()
        start = loop.time()
        await bridge._broadcast("frame-1")
        elapsed = loop.time() - start
        await bridge._broadcast("frame-2")
        return elapsed

    elapsed = asyncio.run(run())
    assert elapsed < 1.0                              # bounded by the timeout, not by the stalled client
    assert healthy.got == ["frame-1", "frame-2"]
    assert bridge.ws_clients == {healthy}
