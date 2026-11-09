"""How much attention each security event deserves, and why.

Events used to carry only a disposition: every ``observed`` event (a routine
sudo exec, a pacman upgrade, an unexpected root gain) looked the same in the
feed. Each event type now has an explicit level and a one-line reason, decided
here and nowhere else, so the HUD, the curses dashboard and any later consumer
agree.

Levels are derived when events are read, never stored: the security event
table is prefix-hashed by tamper detection, so its schema does not change.
An unknown type is ``medium``: a new event must be visible, never silent.
Levels only rank what operators see. Nothing uses them to act (axiom 5).
"""

from __future__ import annotations

from typing import Dict, Tuple

LEVELS = ("info", "low", "medium", "high", "critical")
RANK = {name: i for i, name in enumerate(LEVELS)}

EVENT_LEVELS: Dict[str, Tuple[str, str]] = {
    # Integrity and containment.
    "tamper_detected": ("critical", "a protected file differs from the trusted baseline"),
    "emergency_lockdown_triggered": ("critical", "the network was cut off"),
    # Decoys: nothing legitimate touches them.
    "honeypot_canary_read": ("critical", "a canary flag file was read"),
    "honeypot_file_access": ("critical", "a decoy file was opened"),
    "honeypot_port_trigger": ("critical", "a decoy port was contacted"),
    "honeypot_auto_block": ("high", "a decoy reader's peer was blocked"),
    # Privilege.
    "privilege_escalation_unexpected": ("critical", "root gained outside execve with no prior root identity"),
    "privilege_setuid_exec_unusual": ("high", "set-user-ID root exec from outside the system directories"),
    "privilege_setuid_exec": ("info", "routine set-user-ID exec (sudo, su, pkexec)"),
    "privilege_escalation": ("info", "legacy record (before 2026-11-09): a setuid() attempt, not a gain"),
    "privilege_config_critical": ("critical", "NOPASSWD, a second uid 0, a privileged group member or a preload"),
    "privilege_config_changed": ("medium", "a privilege-granting file changed"),
    # Installed content.
    "package_file_altered": ("critical", "a privileged packaged file differs from its package"),
    "privileged_file_unowned": ("critical", "a set-user/group-ID file no package owns"),
    "package_file_unverifiable": ("medium", "a privileged packaged file could not be checked"),
    "pacman_package_downgraded": ("medium", "a package was downgraded"),
    "pacman_package_installed": ("low", "a package was installed"),
    "pacman_package_removed": ("low", "a package was removed"),
    "pacman_package_reinstalled": ("low", "a package was reinstalled"),
    "pacman_package_upgraded": ("info", "a package was upgraded"),
    "pacman_monitor_baseline_established": ("info", "pacman.log monitoring started"),
    "pacman_monitor_cursor_reset": ("low", "pacman.log was rotated or truncated"),
    # File integrity monitoring.
    "fim_yara_match": ("critical", "a file matched a YARA rule"),
    "fim_daemon_crash": ("high", "the FIM daemon stopped"),
    "fim_event_processing_failure": ("medium", "a file event could not be processed"),
    "fim_audit_write_failure": ("medium", "a FIM audit record could not be written"),
    "fim_yara_scan_error": ("medium", "a YARA scan failed"),
    "fim_scan_skipped": ("low", "a file was too large or vanished before scanning"),
    "telemetry_script_sentinel_alert": ("high", "a script appeared or changed in the workspace"),
    # Policy Gate decisions: denials of what a model or caller asked for.
    "invalid_capability_token": ("high", "a request carried an invalid capability token"),
    "lattice_permission_violation": ("high", "a request exceeded the permission lattice"),
    "file_write_violation": ("high", "a write to a protected path was refused"),
    "unauthorized_file_write_attempt": ("high", "an unauthorised file write was refused"),
    "unauthorized_mitigation_attempt": ("high", "an unauthorised firewall change was refused"),
    "unauthorized_restart_attempt": ("high", "an unauthorised service restart was refused"),
    "unauthorized_service_restart_attempt": ("high", "an unauthorised service restart was refused"),
    "unauthorized_script_run_attempt": ("high", "an unauthorised script run was refused"),
    "unallowed_script_run_attempt": ("high", "a script outside the allowlist was refused"),
    "regex_filter_violation": ("high", "arguments with forbidden characters were refused"),
    "unknown_action_denied": ("high", "an unknown action was refused"),
    "restart_frequency_exceeded": ("high", "service restarts exceeded the rate limit"),
    "schema_validation_failure": ("medium", "a malformed request was refused"),
    "restart_healthy_service_blocked": ("medium", "a restart of a healthy service was refused"),
    "policy_gate_unavailable": ("high", "the Policy Gate could not be reached"),
    "service_restart_event": ("info", "an approved service restart"),
    "block_ip": ("info", "an IP block was applied"),
    "unblock_ip": ("info", "an IP block was lifted"),
}

UNKNOWN = ("medium", "unclassified event type")


def level_of(event_type: str) -> Tuple[str, str]:
    """(level, reason) for an event type; unknown types are never below medium."""
    return EVENT_LEVELS.get(str(event_type), UNKNOWN)


def at_least(event_type: str, level: str) -> bool:
    return RANK[level_of(event_type)[0]] >= RANK[level]


def count_by_level(type_counts: Dict[str, int]) -> Dict[str, int]:
    """{level: n} from {event_type: n}, every level present."""
    out = {name: 0 for name in LEVELS}
    for event_type, n in type_counts.items():
        out[level_of(event_type)[0]] += int(n)
    return out
