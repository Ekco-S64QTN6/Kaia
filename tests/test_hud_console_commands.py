"""Every command the HUD console's help advertises is implemented.

`show audit`, `add rule` and `list recent blocks` were listed in help but fell
through to "Unknown command".
"""

import json
import os
import re

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

import config
import core.web_bridge as wb


def _help_commands():
    lines = wb.dispatch_operator_command("help")["lines"]
    return [l.split(" - ")[0].strip() for l in lines if l.startswith("  ")]


def _example(usage: str) -> str:
    fill = {"<ip>": "203.0.113.9", "<service>": "ollama", "<name>": "demo_rule",
            "<ioc>": "evil.example", "[mitre]": "", "[--since <Nh>]": ""}
    for k, v in fill.items():
        usage = usage.replace(k, v)
    return " ".join(usage.split())


def test_help_and_dispatch_agree(monkeypatch):
    monkeypatch.setattr(wb, "send_policy_gate_ipc", lambda *a, **k: (False, {"message": "gate offline (test)"}))
    monkeypatch.setattr(wb, "query_gate_diagnostics", lambda *a, **k: (False, ""))
    for usage in _help_commands():
        cmd = _example(usage)
        if cmd in ("lockdown", "clear"):
            continue  # lockdown is exercised elsewhere; clear is handled by the browser
        out = wb.dispatch_operator_command(cmd)
        assert not out["lines"][0].startswith("Unknown command"), cmd


@pytest.fixture
def events_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SECURITY_DB_PATH", str(tmp_path / "e.db"))
    import security.db as db
    db.initialize_db()
    db.log_security_event("privilege_escalation_unexpected", "ebpf_telemetry", "exploit (PID 9)", "uid 1000->0", "observed", "s")
    return db


def test_show_audit_lists_events_with_level_and_reason(events_db):
    out = wb.dispatch_operator_command("show audit --since 2h")
    assert out["success"]
    line = next(l for l in out["lines"] if "privilege_escalation_unexpected" in l)
    assert "CRITICAL" in line and "outside execve" in line


@pytest.mark.parametrize("cmd", ["show audit --since 2d", "show audit --since -1h", "show audit extra", "show audit --since 9999h"])
def test_show_audit_rejects_malformed_windows(events_db, cmd):
    assert wb.dispatch_operator_command(cmd)["success"] is False


def test_list_recent_blocks_folds_the_ledger_like_boot_replay(tmp_path, monkeypatch):
    ledger = tmp_path / "audit.json"
    rows = [
        {"result": "approved", "request": {"action": "block_ip", "target_ip": "203.0.113.1"}},
        {"result": "approved", "request": {"action": "block_ip", "target_ip": "203.0.113.2"}},
        {"result": "approved", "request": {"action": "unblock_ip", "target_ip": "203.0.113.1", "protocol": "all"}},
        {"result": "denied", "request": {"action": "block_ip", "target_ip": "203.0.113.3"}},
    ]
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", str(ledger))
    lines = wb.dispatch_operator_command("list recent blocks")["lines"]
    assert any("203.0.113.2" in l for l in lines)
    assert not any("203.0.113.1 " in l or "203.0.113.3" in l for l in lines)


def test_add_rule_goes_through_the_gate_with_a_scoped_token(monkeypatch):
    sent = {}

    def fake_ipc(action, payload, token):
        sent.update(action=action, payload=payload, token=json.loads(token))
        return True, {"message": "YARA rule compiled, validated, and saved successfully."}

    monkeypatch.setattr(wb, "send_policy_gate_ipc", fake_ipc)
    out = wb.dispatch_operator_command("add rule demo_rule evil.example T1059")
    assert out["success"] and sent["action"] == "add_rule"
    assert sent["token"]["capability"] == "write_file" and sent["token"]["target"] == "rules/demo_rule.yar"
    assert sent["payload"]["mitre_framework_id"] == "T1059"
    assert len(sent["payload"]["threat_description"]) >= 10


def test_add_rule_denial_is_reported(monkeypatch):
    monkeypatch.setattr(wb, "send_policy_gate_ipc", lambda *a: (False, {"message": "Validation failed: bad name"}))
    out = wb.dispatch_operator_command("add rule ../x evil.example")
    assert out["success"] is False and "Validation failed" in out["lines"][0]
