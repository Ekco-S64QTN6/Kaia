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
