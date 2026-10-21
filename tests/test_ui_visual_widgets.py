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
    assert '"reputation_score": int(max(0, min(100, rep.get("score", 0))))' in dashboard
    assert "block.reputation_score" in radar
    assert "NO SCORE" in radar
