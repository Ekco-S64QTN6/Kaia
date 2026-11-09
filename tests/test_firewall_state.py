"""The gate publishes the ruleset; dashboards read it instead of sending audited requests (#105).

Summarising audit rows would leave the trail incomplete; removing the requests
keeps every request audited and the ledger free of polling noise.
"""

import json
import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import control_health
from security import firewall_state as fs

RULESET = """table inet filter {
\tchain input {
\t\ttype filter hook input priority filter; policy drop;
\t}
}
table inet kaia_block {
\tchain input {
\t\ttype filter hook input priority -10; policy accept;
\t\tip saddr 203.0.113.9 drop
\t\tip6 saddr 2001:db8::7 tcp dport 22 drop
\t}
}
"""
LOCKED = "table inet kaia_lockdown {\n\tchain input {\n\t\ttype filter hook input priority filter; policy drop;\n\t}\n}\n"


def test_parsers_read_the_nft_output_format():
    assert fs.kaia_block_rule_count(RULESET) == 2
    assert fs.lockdown_state_from_ruleset(True, RULESET) == "inactive"
    assert fs.lockdown_state_from_ruleset(True, LOCKED) == "active"
    assert fs.lockdown_state_from_ruleset(False, LOCKED) == "unknown"


def test_round_trip_freshness_and_refusals(tmp_path, monkeypatch):
    path = str(tmp_path / fs.FILE_NAME)
    fs.write_state(path, True, RULESET, now=1000.0)
    assert (os.stat(path).st_mode & 0o777) == 0o640
    assert fs.read_state(path, now=1005.0) == (True, RULESET)
    assert fs.read_state(path, now=1000.0 + fs.MAX_AGE + 1) is None      # gate stopped publishing
    link = tmp_path / "link.json"
    link.symlink_to(path)
    assert fs.read_state(str(link), now=1005.0) is None
    (tmp_path / "bad.json").write_text("[1,2]")
    assert fs.read_state(str(tmp_path / "bad.json"), now=1.0) is None
    monkeypatch.setattr(fs, "MAX_FILE_BYTES", 10)
    assert fs.read_state(path, now=1005.0) is None


def test_writer_does_not_follow_a_planted_temp_symlink(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("keep")
    path = tmp_path / fs.FILE_NAME
    (tmp_path / f".{fs.FILE_NAME}.{os.getpid()}.tmp").symlink_to(victim)
    fs.write_state(str(path), True, RULESET)
    assert victim.read_text() == "keep" and json.loads(path.read_text())["success"] is True


@pytest.fixture
def health(monkeypatch):
    seen = {}
    monkeypatch.setattr(control_health, "report",
                        lambda name, state, detail="", heartbeat_interval=None: seen.__setitem__(name, (state, detail)))
    monkeypatch.setattr(control_health, "heartbeat", lambda name: None)
    return seen


def test_monitor_publishes_and_reports(tmp_path, health):
    path = str(tmp_path / fs.FILE_NAME)
    mon = fs.FirewallStateMonitor(path, read_ruleset=lambda: (True, LOCKED, ""))
    mon.run_once()
    assert fs.read_state(path) == (True, LOCKED)
    assert health["firewall"][0] == "active" and "LOCKDOWN ACTIVE" in health["firewall"][1]
    fs.FirewallStateMonitor(path, read_ruleset=lambda: (False, "", "nft: permission denied")).run_once()
    assert health["firewall"][0] == "failed" and fs.read_state(path) == (False, "")


def test_dashboard_prefers_the_published_state_and_sends_no_request(tmp_path, monkeypatch):
    import config
    import kaia_dashboard as d
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    fs.write_state(str(tmp_path / fs.FILE_NAME), True, RULESET)
    monkeypatch.setattr(d, "query_gate_diagnostics", lambda *a: pytest.fail("audited request sent"))
    d._ruleset_cache.update(at=None, success=False, stdout="")
    assert d.query_ruleset() == (True, RULESET)


def test_dashboard_falls_back_to_the_gate_when_nothing_fresh_is_published(tmp_path, monkeypatch):
    import config
    import kaia_dashboard as d
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    calls = []
    monkeypatch.setattr(d, "query_gate_diagnostics", lambda kind: (calls.append(kind), (True, LOCKED))[1])
    d._ruleset_cache.update(at=None, success=False, stdout="")
    assert d.query_ruleset() == (True, LOCKED) and calls == ["nft_list"]


def test_gate_runs_the_monitor():
    from pathlib import Path
    gate = Path("security/policy_gate.py").read_text()
    assert "FirewallStateMonitor(" in gate and 'HostExecutor.execute_diagnostics("nft_list", [])' in gate
    assert "firewall" in control_health.GATE_CONTROLS
