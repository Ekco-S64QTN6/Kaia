"""The dashboards must show a lockdown while the host is locked down.

They used to ask `systemctl is-active kaia-lockdown.service`. That unit is a
oneshot without RemainAfterExit, so it reports "inactive" the moment
kaia-lockdown.sh has loaded its drop-everything table. On 2026-10-30 the host
was locked down nine times and both dashboards said the lockdown was off. The
state now comes from the ruleset itself.
"""

import os
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from kaia_dashboard import KaiamonSnapshot, lockdown_state_from_ruleset
from core.web_bridge import serialize_snapshot

# `nft list ruleset` after kaia-lockdown.sh: flush ruleset, then only this table.
LOCKED_RULESET = """table inet kaia_lockdown {
\tchain input {
\t\ttype filter hook input priority filter; policy drop;
\t}

\tchain forward {
\t\ttype filter hook forward priority filter; policy drop;
\t}

\tchain output {
\t\ttype filter hook output priority filter; policy drop;
\t}
}
"""

# A healthy host: the stock nftables.conf table plus Kaia's block table.
HEALTHY_RULESET = """table inet filter {
\tchain input {
\t\ttype filter hook input priority filter; policy drop;
\t\tct state { established, related } accept comment "kaia_lockdown is not this table"
\t}
}
table inet kaia_block {
\tchain input {
\t\ttype filter hook input priority -10; policy accept;
\t\tip saddr 192.0.2.7 drop comment "kaia-block"
\t}
}
table inet kaia_lockdown_snapshot_note {
}
"""


def test_lockdown_table_in_ruleset_is_active():
    assert lockdown_state_from_ruleset(True, LOCKED_RULESET) == "active"


def test_healthy_ruleset_is_inactive_even_when_the_name_appears_elsewhere():
    assert lockdown_state_from_ruleset(True, HEALTHY_RULESET) == "inactive"


def test_unreadable_ruleset_is_unknown_not_off():
    assert lockdown_state_from_ruleset(False, "") == "unknown"
    assert lockdown_state_from_ruleset(False, LOCKED_RULESET) == "unknown"


def test_lockdown_table_name_matches_the_lockdown_script():
    script = Path("scripts/kaia-lockdown.sh").read_text()
    assert "add table inet kaia_lockdown\n" in script


def test_collector_no_longer_trusts_the_oneshot_unit_state():
    source = Path("kaia_dashboard.py").read_text()
    assert '"is-active", "kaia-lockdown.service"' not in source
    assert "lockdown_state_from_ruleset(success, stdout)" in source


def test_hud_posture_goes_critical_on_an_active_lockdown():
    snap = KaiamonSnapshot(
        system_security={"gate_running": True, "lockdown_active": True, "lockdown_state": "active"},
        containment={"sentinel_alerts": 0},
    )
    data = serialize_snapshot(snap)
    assert data["sentinel_state"] == "LockdownMode"
    assert data["threat_level"] == "CRITICAL"
    assert data["system_security"]["lockdown_state"] == "active"


def test_hud_renders_unknown_lockdown_distinctly_from_off():
    js = Path("core/web/static/js/app.js").read_text()
    assert "sec.lockdown_state" in js
    assert "Lockdown <b class=\"warn\">unknown</b>" in js
