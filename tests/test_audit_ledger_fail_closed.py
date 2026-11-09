"""Changes are refused when the audit ledger cannot record them.

The gate executed first and audited after; a failed ledger write was only
logged. A block applied while the ledger was unwritable was live but missing
from the record kaia-restore-blocks replays, so it vanished at the next reboot
while the caller was told "success". The ledger write also followed symlinks.
"""

import json
import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

import config
import security.policy_gate as pg
from security import control_health
from security.host_executor import HostExecutor


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", str(tmp_path / "audit_ledger.json"))
    monkeypatch.setattr(config, "SECURITY_DB_PATH", str(tmp_path / "security_events.db"))
    import security.db
    security.db.initialize_db()
    perms = {"diagnostics", "block_ip", "unblock_ip", "restart_service", "write_file", "run_script", "add_rule"}
    monkeypatch.setattr(config, "GLOBAL_PERMISSIONS", perms)
    monkeypatch.setattr(config, "WORKSPACE_PERMISSIONS", perms)
    events = []
    monkeypatch.setattr(pg, "log_security_event", lambda *a, **k: events.append(a[0] if a else k.get("event_type")))
    health = []
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: health.append((name, state)))
    g = pg.PolicyGate()
    g.events, g.health = events, health
    return g


def _block(target="203.0.113.9"):
    return {"action": "block_ip", "target_ip": target, "protocol": "all", "justification": "test block",
            "session_id": "t",
            "capability_token": pg.generate_capability_token("block_ip", target, duration_seconds=60)}


def _break_ledger(path):
    """Make the ledger path unappendable: a directory sits where the file should be."""
    os.mkdir(path)


@pytest.mark.parametrize("payload", [
    _block(),
    {"action": "unblock_ip", "target_ip": "203.0.113.9", "protocol": "all", "justification": "x", "session_id": "t"},
    {"action": "restart_service", "service_name": "ollama", "justification": "x", "session_id": "t"},
    {"action": "write_file", "path": "notes.txt", "content": "x", "justification": "x", "session_id": "t"},
    {"action": "run_script", "script_name": "free-space.sh", "justification": "x", "session_id": "t"},
    {"action": "add_rule", "rule_name": "abc", "justification": "x", "session_id": "t"},
])
def test_every_mutating_action_is_refused_before_execution(gate, monkeypatch, payload):
    _break_ledger(config.AUDIT_LOG_PATH)
    for name in ("execute_mitigation", "execute_unmitigation", "execute_service_control",
                 "execute_state_modification", "execute_script"):
        monkeypatch.setattr(HostExecutor, name, staticmethod(lambda *a, **k: pytest.fail("executed unaudited")))
    result = gate.evaluate_and_execute(dict(payload))
    assert result["status"] == "denied" and "audit ledger" in result["message"]
    assert "audit_ledger_unavailable" in gate.events
    assert ("audit_ledger", "failed") in gate.health


def test_read_only_diagnostics_and_lockdown_still_work_without_the_ledger(gate, monkeypatch):
    _break_ledger(config.AUDIT_LOG_PATH)
    monkeypatch.setattr(HostExecutor, "execute_diagnostics", staticmethod(lambda *a, **k: (True, "ok", "")))
    r = gate.evaluate_and_execute({"action": "diagnostics", "query_type": "nft_list", "args": [],
                                   "justification": "x", "session_id": "t"})
    assert r["status"] == "success"
    fired = []
    monkeypatch.setattr(pg, "trigger_lockdown", lambda reason: fired.append(reason) or True)
    token = pg.generate_capability_token("diagnostics", "lockdown_confirm", duration_seconds=60)
    r = gate.evaluate_and_execute({"action": "lockdown", "capability_token": token, "session_id": "t"})
    assert fired == ["operator_command"] and r["status"] == "success"


def test_a_change_whose_audit_write_fails_is_reported_as_an_error(gate, monkeypatch):
    monkeypatch.setattr(HostExecutor, "execute_mitigation", staticmethod(lambda *a, **k: (True, "applied", "")))
    real_open = os.open

    def failing_open(path, flags, *a):
        if path == config.AUDIT_LOG_PATH and flags & os.O_APPEND and failing_open.armed:
            raise OSError(28, "No space left on device")
        return real_open(path, flags, *a)

    failing_open.armed = False
    monkeypatch.setattr(pg.PolicyGate, "audit_ledger_available", staticmethod(lambda: (True, None)))
    monkeypatch.setattr(pg.os, "open", failing_open)
    failing_open.armed = True
    result = gate.evaluate_and_execute(_block())
    assert result["status"] == "error" and "could not record" in result["message"]


def test_ledger_write_does_not_follow_a_planted_symlink(gate, tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep\n")
    os.symlink(victim, config.AUDIT_LOG_PATH)
    assert gate._log_audit({"action": "diagnostics", "session_id": "t"}, "approved") is False
    assert victim.read_text() == "keep\n"
    ok, why = pg.PolicyGate.audit_ledger_available()
    assert ok is False


def test_the_failure_log_never_contains_the_token(gate, tmp_path, caplog):
    _break_ledger(config.AUDIT_LOG_PATH)
    payload = _block()
    with caplog.at_level("CRITICAL"):
        gate._log_audit(dict(payload), "approved")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert "block_ip" in text
    assert payload["capability_token"] not in text and json.loads(payload["capability_token"])["signature"] not in text


def test_successful_audit_still_appends(gate):
    assert gate._log_audit({"action": "diagnostics", "session_id": "t"}, "approved") is True
    (line,) = open(config.AUDIT_LOG_PATH).read().splitlines()
    assert json.loads(line)["result"] == "approved"
