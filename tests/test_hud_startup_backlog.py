"""HUD start: history is shown, not counted as live activity; the ledger is followed, not re-read.

Every HUD start replayed the event database from row 0 and the whole ledger
(18 MB on the host), 50 rows per 250 ms, all counted as live log rate: the HUD
showed AuditSurge / ELEVATED for minutes (seen in a headless screenshot,
2026-11-09). The ledger was also re-read in full on every change.
"""

import json
import os
import sqlite3

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import threading

from kaia_dashboard import AuditLogCollector


def _row(i, action="diagnostics"):
    return json.dumps({"timestamp": f"2026-11-09T10:00:{i % 60:02d}Z", "result": "approved",
                       "request": {"action": action}, "reason": f"r{i}"}) + "\n"


def _collector(tmp_path, ledger_lines=0, db_rows=0):
    ledger = tmp_path / "audit.json"
    ledger.write_text("".join(_row(i) for i in range(ledger_lines)))
    db = tmp_path / "events.db"
    with sqlite3.connect(db) as conn:
        conn.execute("""CREATE TABLE security_events (event_id TEXT PRIMARY KEY, timestamp TEXT, type TEXT,
            source TEXT, actor TEXT, payload_hash TEXT, disposition TEXT, session_id TEXT)""")
        for i in range(db_rows):
            conn.execute("INSERT INTO security_events VALUES (?,?,?,?,?,?,?,?)",
                         (f"e{i}", "2026-11-09T10:00:00Z", "block_ip", "g", "a", "", "approved", "s"))
    c = AuditLogCollector(threading.Event())
    c._audit_ledger_path, c._security_db_path = str(ledger), str(db)
    return c, ledger


def test_first_poll_shows_a_bounded_tail_then_only_new_lines(tmp_path):
    c, ledger = _collector(tmp_path, ledger_lines=5000)
    first = c._poll_audit_ledger()
    assert len(first) == 50 and c._backlog and c._ledger_offset == ledger.stat().st_size
    assert c._poll_audit_ledger() == []
    with open(ledger, "a") as f:
        f.write(_row(1, "block_ip") + _row(2, "unblock_ip")[:20])      # one full line, one partial
    (entry,) = c._poll_audit_ledger()
    assert "BLOCK_IP" in entry.message
    with open(ledger, "a") as f:
        f.write(_row(2, "unblock_ip")[20:])
    (entry,) = c._poll_audit_ledger()
    assert "UNBLOCK_IP" in entry.message


def test_rotation_is_followed_from_the_new_file_start(tmp_path):
    c, ledger = _collector(tmp_path, ledger_lines=10)
    c._poll_audit_ledger()
    rotated = tmp_path / "new.json"
    rotated.write_text(_row(1, "restart_service"))
    os.replace(rotated, ledger)
    (entry,) = c._poll_audit_ledger()
    assert "RESTART_SERVICE" in entry.message


def test_event_db_starts_near_the_end(tmp_path):
    c, _ = _collector(tmp_path, db_rows=400)
    assert len(c._poll_security_db()) == 50 and c._backlog
    assert c._poll_security_db() == []


def test_backlog_is_not_counted_as_live_log_rate(tmp_path):
    c, ledger = _collector(tmp_path, ledger_lines=3000, db_rows=400)
    polls = iter(range(3))

    class Stop:
        def is_set(self):
            return False

        def wait(self, _):
            if next(polls, None) is None:
                Stop.is_set = lambda self: True

    c._stop = Stop()
    c._run()
    assert len(c._logs) == 100                      # 50 rows + 50 ledger lines displayed
    assert c._calculate_logs_per_second() == 0      # none counted as activity
