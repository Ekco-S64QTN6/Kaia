"""A security event is never dropped silently.

Event ids were 32 random bits under a PRIMARY KEY (collisions likely within
tens of thousands of events, each one a lost detection); the connect ran
outside the try, so an unopenable database raised into callers that swallow
exceptions; and the failure log omitted the event itself.
"""

import logging
import os
import sqlite3
import uuid

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

import config
import security.db as db
from security import control_health


@pytest.fixture
def events_db(tmp_path, monkeypatch):
    path = tmp_path / "security_events.db"
    monkeypatch.setattr(config, "SECURITY_DB_PATH", str(path))
    db._db_state["ok"] = None
    db.initialize_db()
    return path


@pytest.fixture
def health(monkeypatch):
    seen = []
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.append((name, state, detail)))
    db._db_state["ok"] = None
    return seen


def _rows(path):
    with sqlite3.connect(path) as conn:
        return conn.execute("SELECT event_id, type FROM security_events ORDER BY rowid").fetchall()


def test_ids_are_128_bit(events_db):
    eid = db.log_security_event("tamper_detected", "t", "a", "", "blocked", "s")
    assert eid.startswith("evt_") and len(eid) == 4 + 32
    assert _rows(events_db)[0][0] == eid


def test_an_id_collision_retries_instead_of_dropping_the_event(events_db, monkeypatch):
    ids = iter([uuid.UUID(int=1), uuid.UUID(int=1), uuid.UUID(int=2)])
    monkeypatch.setattr(db.uuid, "uuid4", lambda: next(ids))
    first = db.log_security_event("honeypot_canary_read", "h", "a", "", "blocked", "s")
    second = db.log_security_event("honeypot_canary_read", "h", "b", "", "blocked", "s")
    assert first != second
    assert [t for _, t in _rows(events_db)] == ["honeypot_canary_read", "honeypot_canary_read"]


def test_unopenable_database_never_raises_logs_the_event_and_fails_health(tmp_path, monkeypatch, caplog, health):
    monkeypatch.setattr(config, "SECURITY_DB_PATH", str(tmp_path / "missing-dir" / "x.db"))
    with caplog.at_level(logging.ERROR, logger="security.db"):
        db.log_security_event("privilege_escalation_unexpected", "ebpf_telemetry",
                              "exploit (PID 7)", "uid 1000->0", "observed", "s")
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "privilege_escalation_unexpected" in msg and "exploit (PID 7)" in msg and "uid 1000->0" in msg
    assert ("security_db", "failed") == health[-1][:2]


def test_health_recovers_and_reports_only_on_change(events_db, monkeypatch, health):
    good = config.SECURITY_DB_PATH
    db.log_security_event("block_ip", "g", "a", "", "approved", "s")
    db.log_security_event("block_ip", "g", "a", "", "approved", "s")
    monkeypatch.setattr(config, "SECURITY_DB_PATH", "/nonexistent-dir/x.db")
    db.log_security_event("block_ip", "g", "a", "", "approved", "s")
    db.log_security_event("block_ip", "g", "a", "", "approved", "s")
    monkeypatch.setattr(config, "SECURITY_DB_PATH", good)
    db.log_security_event("block_ip", "g", "a", "", "approved", "s")
    assert [s for n, s, _ in health if n == "security_db"] == ["active", "failed", "active"]


def test_no_writer_uses_short_ids():
    from pathlib import Path
    for path in ("security/db.py", "security/policy_gate.py"):
        assert "uuid.uuid4().hex[:8]" not in Path(path).read_text(), path
    assert "security_db" in control_health.GATE_CONTROLS
