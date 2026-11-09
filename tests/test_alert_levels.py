"""Every security event type has an explicit level and reason.

Before, the feed coloured events by disposition alone: an observed sudo exec,
a pacman upgrade and an unexpected root gain were all yellow "observed".
"""

import os
import re
from pathlib import Path

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

from security import alert_levels
from security.alert_levels import EVENT_LEVELS, LEVELS, count_by_level, level_of

ROOT = Path(__file__).resolve().parent.parent


def _emitted_event_types() -> set:
    """Event types the code can record, read from the source."""
    types = set()
    for path in list((ROOT / "security").glob("*.py")) + [ROOT / "core/web_bridge.py", ROOT / "main.py"]:
        src = path.read_text()
        types |= set(re.findall(r'event_type\s*=\s*"([a-z_]+)"', src))
        types |= set(re.findall(r'"(honeypot_(?:canary_read|file_access))"', src))
        types |= set(re.findall(r'\b(?:PRIV_EVENT|EVENT)_[A-Z_]+ = "([a-z_]+)"', src))
        for m in re.finditer(r"_emit\(\s*\"([a-z_]+)\"", src):
            types.add(m.group(1))
    sc = (ROOT / "security/supply_chain_monitor.py").read_text()
    ops = re.search(r'operation not in \(([^)]*)\)', sc).group(1)
    types |= {f"pacman_package_{op}" for op in re.findall(r'"([a-z]+)"', ops)}
    types |= set(re.findall(r'"(pacman_monitor_[a-z_]+)"', sc))
    # fanotify mask names stored in fim_events, not security event types.
    return types - {"create", "attrib", "modify", "close_write"}


def test_every_emitted_event_type_is_classified():
    emitted = _emitted_event_types()
    assert len(emitted) > 30
    missing = sorted(emitted - set(EVENT_LEVELS))
    assert not missing, f"classify these in security/alert_levels.py: {missing}"


def test_levels_and_reasons_are_well_formed():
    for event_type, (level, reason) in EVENT_LEVELS.items():
        assert level in LEVELS, event_type
        assert reason and len(reason) < 120, event_type


def test_unknown_types_are_visible_never_info():
    assert level_of("something_new")[0] == "medium"


def test_the_cases_that_matter():
    assert level_of("privilege_escalation_unexpected")[0] == "critical"
    assert level_of("privilege_setuid_exec")[0] == "info"
    assert level_of("privilege_escalation")[0] == "info"     # legacy setuid() attempts
    assert level_of("honeypot_canary_read")[0] == "critical"
    assert level_of("tamper_detected")[0] == "critical"
    assert level_of("package_file_altered")[0] == "critical"
    assert level_of("pacman_package_upgraded")[0] == "info"
    assert alert_levels.at_least("invalid_capability_token", "high")


def test_count_by_level_covers_every_level():
    counts = count_by_level({"tamper_detected": 2, "privilege_setuid_exec": 40, "new_thing": 1})
    assert counts == {"info": 40, "low": 0, "medium": 1, "high": 0, "critical": 2}


def test_levels_never_drive_a_response():
    src = (ROOT / "security/alert_levels.py").read_text()
    assert "trigger_lockdown" not in src and "import config" not in src and "security.db" not in src


# --- the feed ---

def test_feed_uses_levels_and_shows_readable_details():
    from kaia_dashboard import format_security_event
    level, msg = format_security_event("privilege_escalation_unexpected", "ebpf_telemetry",
                                       "exploit (PID 31337)", "uid 1000->0 euid 1000->0 suid 1000", "observed")
    assert level == "CRITICAL" and "uid 1000->0" in msg and "CRITICAL" in msg
    level, _ = format_security_event("privilege_setuid_exec", "ebpf_telemetry", "bash (PID 1)",
                                     "uid 1000->1000 euid 1000->0 suid 1000 exec=/usr/bin/sudo", "observed")
    assert level == "INFO"
    assert format_security_event("block_ip", "policy_gate", "203.0.113.4", "", "approved")[0] == "SUCCESS"
    level, msg = format_security_event("tamper_detected", "tamper_detector", "/opt/kaia/core/config.py",
                                       "0123456789abcdef0123456789abcdef", "blocked")
    assert level == "CRITICAL" and "hash:01234567" in msg and "0123456789abcdef0123" not in msg


def test_hud_shows_critical_and_high_counts():
    js = (ROOT / "core/web/static/js/app.js").read_text()
    assert "con.alert_counts_24h" in js
    assert "CRIT / HIGH (24H)" in (ROOT / "core/web/static/index.html").read_text()
    dash = (ROOT / "kaia_dashboard.py").read_text()
    assert '"alert_counts_24h": alert_counts' in dash
