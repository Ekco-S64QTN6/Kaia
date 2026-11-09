"""The HUD feed shows each security event's level and reason, and filters to High+.

The feed parsed levels and sources out of the message text; the four-column
message format from alert levels shifted its columns. Security events now carry
structured fields end to end.
"""

import os
import sqlite3
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from kaia_dashboard import AuditLogCollector, KaiamonSnapshot, security_event_entry
from core.web_bridge import serialize_snapshot

ROOT = Path(__file__).resolve().parent.parent


def test_entry_carries_level_reason_status_and_detail():
    e = security_event_entry("12:00:00", "privilege_escalation_unexpected", "ebpf_telemetry",
                             "exploit (PID 7)", "uid 1000->0", "observed")
    assert (e.event, e.alert_level, e.status) == ("privilege_escalation_unexpected", "critical", "observed")
    assert "outside execve" in e.reason and "uid 1000->0" in e.detail and e.level == "CRITICAL"


def test_collector_and_bridge_pass_the_fields_through(tmp_path):
    db = tmp_path / "e.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE security_events (event_id TEXT PRIMARY KEY, timestamp TEXT, type TEXT,
            source TEXT, actor TEXT, payload_hash TEXT, disposition TEXT, session_id TEXT)""")
        conn.execute("INSERT INTO security_events VALUES ('e1','2026-11-09T17:00:00Z','tamper_detected',"
                     "'tamper_detector','/opt/kaia/core/config.py','0123456789abcdef','blocked','s')")
    c = AuditLogCollector.__new__(AuditLogCollector)
    c._security_db_path, c._last_seen_rowid = str(db), 0
    (entry,) = c._poll_security_db()
    log = serialize_snapshot(KaiamonSnapshot(logs=(entry,)))["logs"][0]
    assert log["event"] == "tamper_detected" and log["alert_level"] == "critical"
    assert log["status"] == "denied" and "trusted baseline" in log["reason"]


def test_hud_renders_level_column_reason_title_and_high_filter():
    js = (ROOT / "core/web/static/js/app.js").read_text()
    panels = (ROOT / "core/web/static/js/panels.js").read_text()
    html = (ROOT / "core/web/static/index.html").read_text()
    assert "if (log.event) {" in panels and "alertLevel: String(log.alert_level" in panels
    assert "levelTag(p.alertLevel)" in js and 'title="${esc(p.reason' in js
    assert "logFilter === 'high' && isHigh(p)" in js and 'data-filter="high"' in html
    assert "<th>LEVEL</th>" in html and "new FitTable('audit-body', 5)" in js


def test_every_hud_script_parses():
    import shutil
    import subprocess
    import pytest
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    for js in sorted((ROOT / "core/web/static/js").glob("*.js")):
        r = subprocess.run([node, "--check", str(js)], capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f"{js.name}: {r.stderr}"
