from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC = REPO_ROOT / "core" / "web" / "static"


def test_dashboard_wires_live_visual_widgets():
    index = (STATIC / "index.html").read_text()
    app = (STATIC / "js" / "app.js").read_text()

    for element_id in ("waterfall-canvas", "network-radar", "radar-risk", "token-ledger"):
        assert f'id="{element_id}"' in index
    for asset in ("js/waterfall.js", "js/radar.js", "js/tokens.js"):
        assert f'/static/{asset}' in index
        assert (STATIC / asset).is_file()
    assert "SyscallWaterfallCanvas" in app
    assert "NetworkRadarCanvas" in app
    assert "CapabilityTokenLedger" in app
    assert "waterfall.update(events" in app
    assert "networkRadar.update(ti)" in app
    assert "tokenLedger.render(sec.token_cards" in app


def test_lps_chart_exposes_surge_threshold_and_area_fill():
    app = (STATIC / "js" / "app.js").read_text()
    charts = (STATIC / "js" / "charts.js").read_text()
    assert "threshold: 35" in app
    assert "areaFill: true" in app
    assert "AUDIT SURGE 35 LPS" in app
    assert "frameIntervalMs: 16" in app
    assert "o.thresholdLabel" in charts
    assert "if (o.areaFill)" in charts


def test_radar_uses_observed_reputation_scores():
    dashboard = (REPO_ROOT / "kaia_dashboard.py").read_text()
    radar = (STATIC / "js" / "radar.js").read_text()
    assert '"reputation_score": int(max(0, min(100, rep_score))) if has_rated_score else None' in dashboard
    assert 'bool(rep.get("cached"))' in dashboard
    assert "block.reputation_score" in radar
    assert "score <= 30" in radar
    assert "score <= 65" in radar
    assert "UNRATED" in radar


def test_openat2_perf_buffer_reaches_the_waterfall():
    ebpf = (REPO_ROOT / "security" / "ebpf_telemetry.py").read_text()
    bridge = (REPO_ROOT / "core" / "web_bridge.py").read_text()
    waterfall = (STATIC / "js" / "waterfall.js").read_text()

    assert 'self.bpf["openat2_events"].open_perf_buffer(self._handle_openat2_event)' in ebpf
    assert 'openat2_events.perf_submit(args, &data, sizeof(data))' in ebpf
    assert "def get_recent_opens" in ebpf
    assert "raw_opens = ebpf_engine.get_recent_opens(30)" in bridge
    assert '"syscall": ev.get("syscall", "sys_enter_openat")' in bridge
    assert "/openat2/i" in waterfall


def test_snapshot_contains_openat2_events_from_telemetry_engine():
    from core.web_bridge import serialize_snapshot
    from kaia_dashboard import CollectorManager
    import time

    class FakeEBPFEngine:
        bpf = object()

        def get_recent_execs(self, n=20):
            return []

        def get_recent_opens(self, n=20):
            return [{
                "syscall": "sys_enter_openat2",
                "pid": 4321,
                "comm": "test-process",
                "filename": "/tmp/test-file",
                "timestamp": time.time(),
            }]

    snapshot = CollectorManager().take_snapshot()
    data = serialize_snapshot(snapshot, ebpf_engine=FakeEBPFEngine(), proc_sampler=None)
    event = next(item for item in data["ebpf_waterfall"] if item["syscall"] == "sys_enter_openat2")

    assert event["pid"] == 4321
    assert event["filename"] == "/tmp/test-file"
    assert event["disposition"] == "observed"
