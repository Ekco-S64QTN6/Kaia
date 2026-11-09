"""Privilege telemetry reports gains of root, taken from commit_creds().

The old probes sat on sys_enter_setuid/sys_enter_setreuid and recorded the
*requested* uid: on the host that was 268 'privilege_escalation' events in 8
days, all of them sudo (already euid 0) and pacman dropping to the alpm user.
They missed setresuid() and every in-kernel credential change, so the one case
that matters, a kernel exploit installing root credentials, was never seen.

The kernel side cannot be loaded here; tests/test_ebpf_program_compiles.py
compiles it locally and DEPLOYMENT_CHECKLIST step 3 exercises it on the host.
"""

import logging
import os
import re

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "ci-test-secret")

import pytest

from security import ebpf_telemetry
from security.ebpf_telemetry import (
    PRIV_ALERT_EVENTS,
    PRIV_EVENT_SETUID_EXEC,
    PRIV_EVENT_SETUID_EXEC_UNUSUAL,
    PRIV_EVENT_UNEXPECTED,
    PRIV_KIND_NO_PRIOR_ROOT,
    PRIV_KIND_SETUID_EXEC,
    EBPFTelemetryEngine,
    classify_privilege_gain,
)

TEXT = ebpf_telemetry.bpf_text


def _function_body(name: str) -> str:
    start = TEXT.index(f"int {name}(")
    end = TEXT.index("\n}\n", start)
    return TEXT[start:end]


# --- kernel program (source level) ---

def test_setuid_syscall_entry_probes_are_gone():
    assert "sys_enter_setuid" not in TEXT
    assert "sys_enter_setreuid" not in TEXT


def test_kind_constants_match_the_kernel_program():
    assert f"#define PRIV_KIND_SETUID_EXEC {PRIV_KIND_SETUID_EXEC}" in TEXT
    assert f"#define PRIV_KIND_NO_PRIOR_ROOT {PRIV_KIND_NO_PRIOR_ROOT}" in TEXT


def test_event_struct_carries_old_and_new_identities():
    struct = TEXT[TEXT.index("struct privilege_event_t {"):TEXT.index("};", TEXT.index("struct privilege_event_t {"))]
    for field in ("pid", "kind", "old_uid", "old_euid", "old_suid", "new_uid", "new_euid"):
        assert re.search(rf"\bu32 {field};", struct), field
    assert "char comm[16];" in struct and "char filename[256];" in struct


def test_exec_is_recognised_only_by_the_exact_cred_being_installed():
    body = _function_body("kprobe__commit_creds")
    # A stale map entry must never classify a later, unrelated change as exec.
    assert "if (ec->cred == (u64)new)" in body
    assert "exec_creds.delete(&tid);" in body
    hook = _function_body("kprobe__security_bprm_committing_creds")
    assert "&bprm->cred" in hook and "exec_creds.update(&tid, &ec);" in hook


def test_only_gains_of_root_leave_the_kernel():
    body = _function_body("kprobe__commit_creds")
    # Already effectively root, or nothing root in the new cred: not a gain.
    assert "if (oe.val == 0 || (nu.val != 0 && ne.val != 0))" in body
    # Outside exec, regaining a root real/saved uid (sudo's seteuid dance) is not a gain.
    assert "if (ou.val == 0 || os.val == 0)" in body
    assert body.index("if (ou.val == 0 || os.val == 0)") < body.index("kind = PRIV_KIND_NO_PRIOR_ROOT;")
    # The old identity is the task's objective cred.
    assert "&task->real_cred" in body


# --- classification ---

@pytest.mark.parametrize("path", [
    "/usr/bin/sudo", "/usr/bin/su", "/usr/bin/pkexec", "/usr/bin/fusermount3",
    "/usr/lib/polkit-1/polkit-agent-helper-1", "/usr/lib/dbus-daemon-launch-helper",
])
def test_system_setuid_exec_is_routine(path):
    assert classify_privilege_gain(PRIV_KIND_SETUID_EXEC, path) == PRIV_EVENT_SETUID_EXEC


@pytest.mark.parametrize("path", [
    "/tmp/x", "/home/user/.cache/su", "./sudo", "sudo", "",
    "/usr/bin/../../tmp/x", "/usr/bin/./../../home/u/x",
    "/usr/bin/su\ndo", "/usr/bin/su do", "/usr/bin/" + "a" * 300,
])
def test_setuid_exec_elsewhere_or_from_odd_paths_is_flagged(path):
    assert classify_privilege_gain(PRIV_KIND_SETUID_EXEC, path) == PRIV_EVENT_SETUID_EXEC_UNUSUAL


def test_gain_without_exec_or_prior_root_is_unexpected_whatever_the_path():
    assert classify_privilege_gain(PRIV_KIND_NO_PRIOR_ROOT, "") == PRIV_EVENT_UNEXPECTED
    assert classify_privilege_gain(PRIV_KIND_NO_PRIOR_ROOT, "/usr/bin/sudo") == PRIV_EVENT_UNEXPECTED


def test_unknown_kind_is_never_routine():
    assert classify_privilege_gain(0, "/usr/bin/sudo") == PRIV_EVENT_UNEXPECTED
    assert classify_privilege_gain(99, "/usr/bin/sudo") == PRIV_EVENT_UNEXPECTED


# --- recording ---

@pytest.fixture
def recorded(monkeypatch):
    events = []
    import security.db
    monkeypatch.setattr(security.db, "log_security_event", lambda **kw: events.append(kw))
    return events


def test_sudo_exec_is_recorded_without_an_alert(recorded, caplog):
    engine = EBPFTelemetryEngine()
    with caplog.at_level(logging.WARNING, logger="security.ebpf_telemetry"):
        engine._record_privilege_event(
            pid=4242, comm="bash", kind=PRIV_KIND_SETUID_EXEC,
            old_ids=(1000, 1000, 1000), new_ids=(1000, 0), filename="/usr/bin/sudo",
        )
    assert not caplog.records
    (ev,) = recorded
    assert ev["event_type"] == PRIV_EVENT_SETUID_EXEC
    assert ev["actor"] == "bash (PID 4242)"
    assert ev["payload_hash"] == "uid 1000->1000 euid 1000->0 suid 1000 exec=/usr/bin/sudo"
    assert ev["disposition"] == "observed"
    (entry,) = engine.get_privilege_escalations()
    assert entry["type"] == PRIV_EVENT_SETUID_EXEC and entry["euid"] == 0


def test_root_from_nowhere_logs_critical_and_never_responds(recorded, caplog, monkeypatch):
    import security.policy_gate as gate
    monkeypatch.setattr(gate, "trigger_lockdown", lambda *a, **k: pytest.fail("alert-only"))
    engine = EBPFTelemetryEngine()
    with caplog.at_level(logging.CRITICAL, logger="security.ebpf_telemetry"):
        engine._record_privilege_event(
            pid=31337, comm="exploit", kind=PRIV_KIND_NO_PRIOR_ROOT,
            old_ids=(1000, 1000, 1000), new_ids=(0, 0), filename="",
        )
    assert any("Unexpected root gain" in r.getMessage() for r in caplog.records)
    (ev,) = recorded
    assert ev["event_type"] == PRIV_EVENT_UNEXPECTED
    assert ev["payload_hash"] == "uid 1000->0 euid 1000->0 suid 1000"
    assert ev["disposition"] == "observed"


def test_hostile_comm_and_path_are_sanitised_before_recording(recorded):
    engine = EBPFTelemetryEngine()
    engine._record_privilege_event(
        pid=7, comm="x;rm -rf /", kind=PRIV_KIND_SETUID_EXEC,
        old_ids=(1000, 1000, 1000), new_ids=(1000, 0), filename="/tmp/a b;$(id)",
    )
    (ev,) = recorded
    assert ev["event_type"] == PRIV_EVENT_SETUID_EXEC_UNUSUAL
    assert ";" not in ev["actor"] and " " not in ev["actor"].split(" (PID")[0]
    assert ";" not in ev["payload_hash"] and "$" not in ev["payload_hash"]


def test_dashboard_alert_types_match_the_engine():
    import kaia_dashboard
    assert tuple(kaia_dashboard.PRIVILEGE_ALERT_EVENT_TYPES) == tuple(PRIV_ALERT_EVENTS)
    source = open(kaia_dashboard.__file__).read()
    assert "WHERE type = 'privilege_escalation'" not in source
