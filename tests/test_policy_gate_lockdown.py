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

def test_restart_frequency_check_fails_closed_when_database_is_unavailable(tmp_path, monkeypatch):
    import config

    monkeypatch.setattr(
        config, "SECURITY_DB_PATH", str(tmp_path / "missing-parent" / "events.db")
    )
    gate = PolicyGate()

    reserved, limit_exceeded, message = gate._reserve_service_restart(
        "nginx", "unit-test", {"action": "restart_service"}
    )

    assert reserved is False
    assert limit_exceeded is False
    assert "denied" in message.lower()


def test_restart_frequency_reservation_is_atomic_under_concurrency(tmp_path, monkeypatch):
    import sqlite3
    from concurrent.futures import ThreadPoolExecutor

    import config
    from security.db import initialize_db

    db_path = tmp_path / "events.db"
    monkeypatch.setattr(config, "SECURITY_DB_PATH", str(db_path))
    monkeypatch.setattr(config, "RESTART_MAX_FREQUENCY_COUNT", 3)
    monkeypatch.setattr(config, "RESTART_MAX_FREQUENCY_WINDOW_SECONDS", 300)
    initialize_db()
    gate = PolicyGate()

    def reserve(index):
        return gate._reserve_service_restart(
            "nginx", f"session-{index}", {"action": "restart_service", "index": index}
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(8)))

    assert sum(1 for reserved, _, _ in results if reserved) == 3
    assert sum(1 for reserved, exceeded, _ in results if exceeded) == 5

    with sqlite3.connect(db_path) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM security_events WHERE type = 'service_restart_event'"
        ).fetchone()[0]
    assert count == 3

