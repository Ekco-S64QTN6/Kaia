import pytest

import security.policy_gate as policy_gate_module
from security.policy_gate import PolicyGate, generate_capability_token


@pytest.fixture
def gate(monkeypatch):
    gate = PolicyGate()
    monkeypatch.setattr(gate, "_log_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(policy_gate_module, "log_security_event", lambda *args, **kwargs: None)
    return gate


def test_lockdown_denies_missing_token(gate, monkeypatch):
    triggered = []
    monkeypatch.setattr(policy_gate_module, "trigger_lockdown", lambda reason: triggered.append(reason))

    result = gate.evaluate_and_execute({"action": "lockdown", "session_id": "test"})

    assert result["status"] == "denied"
    assert triggered == []


def test_lockdown_denies_wrong_scope_token(gate, monkeypatch):
    triggered = []
    monkeypatch.setattr(policy_gate_module, "trigger_lockdown", lambda reason: triggered.append(reason))
    token = generate_capability_token("restart_service", "sshd.service", duration_seconds=15)

    result = gate.evaluate_and_execute({
        "action": "lockdown",
        "capability_token": token,
        "session_id": "test",
    })

    assert result["status"] == "denied"
    assert triggered == []


def test_lockdown_accepts_short_lived_scoped_token(gate, monkeypatch):
    triggered = []
    monkeypatch.setattr(policy_gate_module, "trigger_lockdown", lambda reason: triggered.append(reason))
    token = generate_capability_token("diagnostics", "lockdown_confirm", duration_seconds=15)

    result = gate.evaluate_and_execute({
        "action": "lockdown",
        "capability_token": token,
        "session_id": "test",
    })

    assert result["status"] == "success"
    assert triggered == ["operator_command"]



def test_restart_healthy_service_denied_even_with_test_named_session(gate, monkeypatch):
    import sqlite3
    from security import telemetry_daemon
    from security.host_executor import HostExecutor

    monkeypatch.setattr(policy_gate_module.config, "GLOBAL_LATTICE_LEVEL", "bwrap")
    monkeypatch.setattr(policy_gate_module.config, "WORKSPACE_LATTICE_LEVEL", "none")
    monkeypatch.setattr(
        policy_gate_module.config, "GLOBAL_PERMISSIONS",
        {"diagnostics", "block_ip", "restart_service", "write_file", "run_script"},
    )
    monkeypatch.setattr(
        policy_gate_module.config, "WORKSPACE_PERMISSIONS",
        {"diagnostics", "block_ip", "restart_service", "write_file", "run_script"},
    )
    monkeypatch.setattr(
        policy_gate_module.config, "SERVICE_RESTART_ALLOWLIST",
        set(policy_gate_module.config.SERVICE_RESTART_ALLOWLIST) | {"nginx"},
    )

    class Cursor:
        def execute(self, *args, **kwargs):
            return None

        def fetchone(self):
            return (0,)

    class Connection:
        def cursor(self):
            return Cursor()

        def close(self):
            return None

    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: Connection())
    monkeypatch.setattr(
        telemetry_daemon, "get_systemd_unit_status",
        lambda name: {"active_state": "active", "sub_state": "running"},
    )
    executed = []
    monkeypatch.setattr(
        HostExecutor, "execute_service_control",
        lambda service: executed.append(service) or (True, "restarted", ""),
    )

    result = gate.evaluate_and_execute({
        "action": "restart_service",
        "service_name": "nginx",
        "justification": "Verify healthy services cannot be restarted by labeling the session as a test.",
        "capability_token": generate_capability_token("restart_service", "nginx"),
        "session_id": "test-exploit-session",
    })

    assert result["status"] == "denied"
    assert "healthy" in result["message"].lower()
    assert executed == []
