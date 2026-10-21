"""Regression tests for the privileged web command endpoint guards."""

import pytest
from aiohttp.test_utils import TestClient, TestServer
from core.web_bridge import KaiaWebBridge


@pytest.mark.asyncio
async def test_command_endpoint_rejects_non_object_json(monkeypatch):
    """Malformed JSON shapes must fail closed rather than raising in the handler."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        resp = await client.post("/api/command", json=["status"])
        assert resp.status == 400
        assert "object" in (await resp.json())["error"].lower()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_command_endpoint_rejects_untrusted_host(monkeypatch):
    """A loopback-bound service must reject Host headers used for DNS rebinding."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        resp = await client.post(
            "/api/command", json={"command": "status"},
            headers={"Host": "attacker.example"},
        )
        assert resp.status == 403
        assert "Host" in (await resp.json())["error"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_command_endpoint_rejects_cross_origin(monkeypatch):
    """Browser-originated commands must be restricted to the local dashboard origin."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        resp = await client.post(
            "/api/command", json={"command": "status"},
            headers={"Origin": "https://attacker.example"},
        )
        assert resp.status == 403
        assert "Origin" in (await resp.json())["error"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_command_endpoint_rejects_oversized_command(monkeypatch):
    """Reject oversized input before it reaches the command dispatcher."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        resp = await client.post("/api/command", json={"command": "x" * 2049})
        assert resp.status == 413
    finally:
        await client.close()
