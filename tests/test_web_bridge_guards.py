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



@pytest.mark.asyncio
async def test_command_endpoint_rejects_origin_with_implicit_default_port(monkeypatch):
    """An origin with no port means 80/443, not the HUD's non-default port."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        local_url = client.make_url("/")
        assert local_url.port is not None
        origin = f"http://{local_url.host}"  # implied port 80
        resp = await client.post(
            "/api/command",
            json={"command": "status"},
            headers={"Origin": origin},
        )
        assert resp.status == 403
        assert "port" in (await resp.json())["error"].lower()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_command_endpoint_accepts_exact_same_origin(monkeypatch):
    """The host and explicit port of a same-origin HUD request remain valid."""
    import core.web_bridge as web_bridge_module

    monkeypatch.setattr(
        web_bridge_module,
        "dispatch_operator_command",
        lambda command: {"success": True, "command": command, "lines": []},
    )
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        local_url = client.make_url("/")
        origin = f"{local_url.scheme}://{local_url.host}:{local_url.port}"
        resp = await client.post(
            "/api/command",
            json={"command": "status"},
            headers={"Origin": origin},
        )
        assert resp.status == 200
        assert (await resp.json())["success"] is True
    finally:
        await client.close()
