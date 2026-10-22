#!/usr/bin/env python3
"""
Unit and Integration Tests for Kaia Cyber-Sentinel Web Bridge Subsystem
=======================================================================
Verifies zero-trust invariants, snapshot serialization, CSP security headers,
HTTP endpoints, and WebSocket telemetry streaming.
"""

import os
import sys
import json
import pytest
from aiohttp.client_exceptions import WSServerHandshakeError
import asyncio
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop, TestClient, TestServer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.web_bridge import (
    KaiaWebBridge,
    serialize_snapshot,
    dispatch_operator_command,
    ProcessSyscallSampler,
    DEFAULT_HOST,
    DEFAULT_PORT,
)
from kaia_dashboard import CollectorManager, KaiamonSnapshot, LogEntry


class TestZeroTrustInvariants:
    def test_loopback_isolation_invariant(self):
        """INV-Loopback: Reject non-loopback bindings (0.0.0.0, external IPs)."""
        with pytest.raises(ValueError, match="Security Invariant Violation"):
            KaiaWebBridge(host="0.0.0.0")

        with pytest.raises(ValueError, match="Security Invariant Violation"):
            KaiaWebBridge(host="192.168.1.100")

        # Loopback must succeed
        bridge1 = KaiaWebBridge(host="127.0.0.1")
        assert bridge1.host == "127.0.0.1"

        bridge2 = KaiaWebBridge(host="localhost")
        assert bridge2.host == "localhost"

    def test_snapshot_serialization_structure(self):
        """Verify that serialize_snapshot generates all required HUD telemetry fields."""
        cm = CollectorManager()
        snap = cm.take_snapshot()
        data = serialize_snapshot(snap)

        required_keys = [
            "timestamp_iso",
            "timestamp_unix",
            "lps",
            "lps_scale",
            "filter_mode",
            "paused",
            "threat_level",
            "sentinel_state",
            "logs",
            "threat_intel",
            "containment",
            "system_security",
            "host_telemetry",
            "services",
            "ebpf_waterfall",
        ]
        for k in required_keys:
            assert k in data, f"Missing required telemetry key: {k}"

        assert data["sentinel_state"] in [
            "SentinelWatch", "PolicyEvaluating", "AnomalyDetected",
            "ContainmentAlert", "LockdownMode", "AuditSurge", "SleepyIdle"
        ]
        assert data["threat_level"] in ["LOW", "ELEVATED", "CRITICAL"]
        assert isinstance(data["host_telemetry"], dict)
        assert "cpu_pct" in data["host_telemetry"]
        assert "ram_used_gb" in data["host_telemetry"]
        assert isinstance(data["services"], list)
        assert isinstance(data["ebpf_waterfall"], list)

    def test_process_syscall_sampler(self):
        """Verify fallback userspace sampler produces sys_enter_execve event objects."""
        sampler = ProcessSyscallSampler()
        events = sampler.sample()
        assert isinstance(events, list)
        for ev in events:
            assert "syscall" in ev
            assert "pid" in ev
            assert "comm" in ev

    def test_command_dispatcher_help_and_status(self):
        """Verify operator command dispatch validates commands and returns structured lines."""
        res_help = dispatch_operator_command("help")
        assert res_help["success"] is True
        assert any("KAIA CYBER-SENTINEL" in line for line in res_help["lines"])

        res_status = dispatch_operator_command("status")
        assert res_status["success"] is True
        assert any("Policy Gate Status" in line for line in res_status["lines"])

        res_invalid = dispatch_operator_command("nonexistent_command_xyz")
        assert res_invalid["success"] is False


@pytest.mark.asyncio
async def test_web_bridge_http_and_security_headers():
    """Verify HTTP endpoints and strict Content-Security-Policy headers."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    app = bridge.create_app()

    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        # 1. Index route
        resp = await client.get("/")
        assert resp.status == 200
        text = await resp.text()
        assert "KAIA //" in text
        assert "Content-Security-Policy" in resp.headers
        csp = resp.headers["Content-Security-Policy"]
        assert "default-src 'self'" in csp
        assert "script-src 'self'" in csp
        assert "style-src 'self'" in csp
        assert "font-src 'self'" in csp
        assert "font-src 'self' data:" not in csp
        assert "'unsafe-inline'" not in csp
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"

        # 2. Health check
        resp_health = await client.get("/api/health")
        assert resp_health.status == 200
        health_json = await resp_health.json()
        assert health_json["status"] == "healthy"
        assert health_json["loopback_isolated"] is True

        # 3. Snapshot endpoint
        resp_snap = await client.get("/api/snapshot")
        assert resp_snap.status == 200
        snap_json = await resp_snap.json()
        assert "sentinel_state" in snap_json
        assert "host_telemetry" in snap_json

        # 4. Command endpoint
        resp_cmd = await client.post("/api/command", json={"command": "help"})
        assert resp_cmd.status == 200
        cmd_json = await resp_cmd.json()
        assert cmd_json["success"] is True

        # Bad command request
        resp_bad = await client.post("/api/command", data="not json")
        assert resp_bad.status == 400

        # 5. Static assets
        resp_css = await client.get("/static/css/sentinel.css")
        assert resp_css.status == 200
        css_text = await resp_css.text()
        assert "--cyan: #00f0ff" in css_text

        resp_js = await client.get("/static/js/app.js")
        assert resp_js.status == 200

    finally:
        await client.close()


@pytest.mark.asyncio
async def test_websocket_rejects_cross_origin_browser_connection():
    """A foreign site must not be able to read local host telemetry over WS."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        with pytest.raises(WSServerHandshakeError):
            await client.ws_connect("/ws", headers={"Origin": "https://attacker.example"})
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_websocket_telemetry_stream():
    """Verify WebSocket connection receives live JSON telemetry snapshots."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    app = bridge.create_app()

    client = TestClient(TestServer(app))
    await client.start_server()

    try:
        ws = await client.ws_connect("/ws")
        msg = await ws.receive_str()
        data = json.loads(msg)
        assert "sentinel_state" in data
        assert "host_telemetry" in data
        assert "timestamp_iso" in data

        # Send ping, expect pong or safe ignore
        await ws.send_str("ping")
        resp = await ws.receive_str()
        assert resp == "pong"

        await ws.close()
    finally:
        await client.close()


class TestSentinelPosture:
    """Posture must follow *recent* activity, never lifetime counters."""

    @staticmethod
    def _snap(rejected=0, alerts=0, logs=(), gate=True):
        from kaia_dashboard import KaiamonSnapshot
        return KaiamonSnapshot(
            logs=tuple(logs),
            system_security={"tokens_rejected": rejected, "gate_running": gate, "lockdown_active": False},
            containment={"sentinel_alerts": alerts},
        )

    def test_lifetime_rejections_do_not_pin_alert(self):
        from core.web_bridge import SentinelStateTracker
        tracker = SentinelStateTracker()
        recent = tracker.observe(self._snap(rejected=172, alerts=3), now=1000.0)
        assert recent["alerting"] is False
        assert recent["rejections_10m"] == 0
        data = serialize_snapshot(self._snap(rejected=172, alerts=3))
        assert data["sentinel_state"] != "ContainmentAlert"

    def test_new_rejection_alerts_then_decays(self):
        from core.web_bridge import SentinelStateTracker
        tracker = SentinelStateTracker()
        tracker.observe(self._snap(rejected=10), now=1000.0)
        recent = tracker.observe(self._snap(rejected=12), now=1001.0)
        assert recent["alerting"] is True
        assert recent["rejections_10m"] == 2
        later = tracker.observe(self._snap(rejected=12), now=1001.0 + SentinelStateTracker.ALERT_HOLD_SEC + 1)
        assert later["alerting"] is False
        assert later["rejections_10m"] == 2   # still inside the 10-minute window

    def test_new_denied_log_line_alerts_and_approved_evaluates(self):
        from core.web_bridge import SentinelStateTracker
        tracker = SentinelStateTracker()
        base = [LogEntry("10:00:00", "SUCCESS", "✓ BOOT │ approved │ (policy_gate)")]
        tracker.observe(self._snap(logs=base), now=1000.0)
        denied = base + [LogEntry("10:00:05", "ERROR", "✗ FILE_WRITE_VIOLATION │ denied │ hash:ab (policy_gate)")]
        assert tracker.observe(self._snap(logs=denied), now=1005.0)["alerting"] is True
        approved = denied + [LogEntry("10:01:00", "SUCCESS", "✓ BLOCK_IP │ approved │ hash:cd (policy_gate)")]
        recent = tracker.observe(self._snap(logs=approved), now=1100.0)
        assert recent["evaluating"] is True and recent["alerting"] is False


class TestHonestTelemetry:
    def test_sampled_processes_are_observed_not_safe(self):
        sampler = ProcessSyscallSampler()
        sampler._known_pids = set()          # treat every running process as new
        events = sampler.sample()
        assert events, "sampler should see at least this test process"
        assert {e["disposition"] for e in events} == {"observed"}

    def test_host_introspector_shapes(self):
        from core.web_bridge import HostIntrospector
        h = HostIntrospector()
        top = h.top_processes()
        assert top and {"pid", "name", "user", "cpu", "memory_mb"} <= set(top[0])
        for sock in h.listening_sockets():
            assert sock["proto"] in ("tcp", "udp")
            assert isinstance(sock["exposed"], bool)

    def test_snapshot_exposes_source_and_host(self):
        cm = CollectorManager()
        data = serialize_snapshot(cm.take_snapshot())
        assert data["process_source"]
        assert data["host"]
        assert "recent_activity" in data


@pytest.mark.asyncio
async def test_static_ui_has_no_inline_css_or_javascript():
    """The CSP must protect both the primary dashboard and the avatar preview."""
    bridge = KaiaWebBridge(host="127.0.0.1", port=8765)
    client = TestClient(TestServer(bridge.create_app()))
    await client.start_server()
    try:
        for path in ("/", "/static/kaia-preview.html"):
            resp = await client.get(path)
            assert resp.status == 200
            html = await resp.text()
            assert "<style" not in html.lower()
            assert "<script>" not in html.lower()
            assert " style=" not in html.lower()
            assert "style-src 'self'" in resp.headers["Content-Security-Policy"]
        css_resp = await client.get("/static/css/kaia-preview.css")
        js_resp = await client.get("/static/js/kaia/kaia-preview.js")
        assert css_resp.status == 200
        assert js_resp.status == 200
    finally:
        await client.close()


def test_terminal_colors_use_stylesheet_classes_under_strict_csp():
    from pathlib import Path
    static = Path(__file__).resolve().parents[1] / "core" / "web" / "static"
    terminal_js = (static / "js" / "terminal.js").read_text()
    stylesheet = (static / "css" / "sentinel.css").read_text()

    assert "style=" not in terminal_js
    for class_name in (
        "term-cyan", "term-dim", "term-line-bright", "term-green",
        "term-red", "term-yellow", "term-text", "term-text-bright", "term-bold",
    ):
        assert f".{class_name}" in stylesheet
        assert f'class="{class_name}' in terminal_js or class_name == "term-bold"


def test_dashboard_dynamic_status_colours_use_css_classes_under_strict_csp():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    app = (root / "core" / "web" / "static" / "js" / "app.js").read_text()
    css = (root / "core" / "web" / "static" / "css" / "sentinel.css").read_text()

    assert ".style" not in app
    assert "state-good" in app and "state-bad" in app
    assert "#policy-gate-badge.state-good" in css
    assert "#policy-gate-badge.state-bad" in css
    assert "#host-meta.state-bad" in css
