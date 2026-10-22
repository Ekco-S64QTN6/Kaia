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
    """The Host and Origin of a same-origin HUD request remain valid."""
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


@pytest.mark.asyncio
async def test_command_endpoint_rejects_host_authority_with_path_query_or_fragment():
    """Host is an authority, not a URL; suffix components must not be accepted."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        local_url = client.make_url("/")
        port = local_url.port
        for suffix in ("/path", "?query=1", "#fragment"):
            resp = await client.post(
                "/api/command",
                json={"command": "status"},
                headers={"Host": f"127.0.0.1:{port}{suffix}"},
            )
            assert resp.status == 403
            assert "Host" in (await resp.json())["error"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_command_endpoint_rejects_host_port_not_matching_listener():
    """A matching loopback name is insufficient if Host names another port."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        local_url = client.make_url("/")
        other_port = 80 if local_url.port != 80 else 443
        resp = await client.post(
            "/api/command",
            json={"command": "status"},
            headers={"Host": f"127.0.0.1:{other_port}"},
        )
        assert resp.status == 403
        assert "port" in (await resp.json())["error"].lower()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_command_endpoint_rejects_origin_scheme_mismatch():
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        local_url = client.make_url("/")
        origin = f"https://{local_url.host}:{local_url.port}"
        resp = await client.post(
            "/api/command",
            json={"command": "status"},
            headers={"Origin": origin},
        )
        assert resp.status == 403
        assert "scheme" in (await resp.json())["error"].lower()
    finally:
        await client.close()



@pytest.mark.asyncio
async def test_command_endpoint_checks_origin_before_parsing_malformed_body():
    """Reject cross-origin requests before their body is parsed."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        resp = await client.post(
            "/api/command",
            data="{not-json",
            headers={"Origin": "https://attacker.example"},
        )
        assert resp.status == 403
        assert "Origin" in (await resp.json())["error"]
    finally:
        await client.close()

@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/", "/api/health", "/api/snapshot"])
async def test_read_only_routes_reject_untrusted_host_for_dns_rebinding(path):
    """Telemetry endpoints must not trust arbitrary Host values just because they are GETs."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        resp = await client.get(path, headers={"Host": "attacker.example"})
        assert resp.status == 403
        assert "Host" in (await resp.json())["error"]
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/", "/api/health", "/api/snapshot"])
async def test_read_only_routes_reject_cross_origin_requests(path):
    """All HTTP routes should use the same browser-origin guard."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        resp = await client.get(path, headers={"Origin": "https://attacker.example"})
        assert resp.status == 403
        assert "Origin" in (await resp.json())["error"]
    finally:
        await client.close()
