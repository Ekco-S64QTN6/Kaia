import json
from datetime import UTC, datetime, timedelta

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
        rowcount = 1

        def execute(self, *args, **kwargs):
            return None

        def fetchone(self):
            return (0,)

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            return None

        def rollback(self):
            return None

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

def test_restart_service_denied_when_health_probe_is_missing_or_malformed(gate, monkeypatch):
    import config
    from security import telemetry_daemon
    from security.host_executor import HostExecutor

    monkeypatch.setattr(config, "GLOBAL_PERMISSIONS", {"restart_service"})
    monkeypatch.setattr(config, "WORKSPACE_PERMISSIONS", {"restart_service"})
    monkeypatch.setattr(
        config, "SERVICE_RESTART_ALLOWLIST",
        set(config.SERVICE_RESTART_ALLOWLIST) | {"nginx"},
    )
    monkeypatch.setattr(
        telemetry_daemon, "get_systemd_unit_status",
        lambda name: {},
    )
    monkeypatch.setattr(
        HostExecutor, "execute_service_control",
        lambda *args: pytest.fail("unknown health must not reach service restart"),
    )

    result = gate.evaluate_and_execute({
        "action": "restart_service",
        "service_name": "nginx",
        "justification": "test that missing health telemetry fails closed",
        "capability_token": generate_capability_token("restart_service", "nginx"),
        "session_id": "unit_test",
    })

    assert result["status"] == "denied"
    assert "cannot determine health" in result["message"].lower()



def _signed_temporal_token(issued_at, expires):
    token_data = {
        "capability": "diagnostics",
        "target": "lockdown_confirm",
        "issued_at": issued_at,
        "expires": expires,
        "issued_by": "operator",
    }
    token_data["signature"] = policy_gate_module.sign_token(
        token_data, policy_gate_module.config.CAPABILITY_TOKEN_SECRET
    )
    return json.dumps(token_data)


def _token_timestamp(value):
    return value.isoformat().replace("+00:00", "Z")


def test_capability_token_accepts_exact_maximum_lifetime():
    now = datetime.now(UTC)
    issued_at = now - timedelta(seconds=1)
    expires_at = issued_at + timedelta(
        seconds=policy_gate_module.MAX_CAPABILITY_TOKEN_LIFETIME_SECONDS
    )
    token = _signed_temporal_token(
        _token_timestamp(issued_at), _token_timestamp(expires_at)
    )

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is True
    assert reason is None


def test_capability_token_rejects_future_issued_at():
    issued_at = datetime.now(UTC) + timedelta(minutes=1)
    expires_at = issued_at + timedelta(seconds=15)
    token = _signed_temporal_token(
        _token_timestamp(issued_at), _token_timestamp(expires_at)
    )

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "future" in reason.lower()


@pytest.mark.parametrize(
    "issued_at,expires_at",
    [
        ("not-a-timestamp", "2026-10-30T12:00:00Z"),
        ("2026-10-30T12:00:00", "2026-10-30T12:00:15Z"),
    ],
)
def test_capability_token_rejects_malformed_or_timezone_naive_issued_at(
    issued_at, expires_at
):
    token = _signed_temporal_token(issued_at, expires_at)

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "issued_at" in reason.lower()


@pytest.mark.parametrize(
    "issued_at,expires_at",
    [
        ("2026-10-30T12:00:00Z", "not-a-timestamp"),
        ("2026-10-30T12:00:00Z", "2026-10-30T12:00:15"),
    ],
)
def test_capability_token_rejects_malformed_or_timezone_naive_expiration(
    issued_at, expires_at
):
    token = _signed_temporal_token(issued_at, expires_at)

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "expires" in reason.lower()


@pytest.mark.parametrize("offset_seconds", [0, -1])
def test_capability_token_rejects_non_positive_lifetime(offset_seconds):
    issued_at = datetime.now(UTC) - timedelta(seconds=2)
    expires_at = issued_at + timedelta(seconds=offset_seconds)
    token = _signed_temporal_token(
        _token_timestamp(issued_at), _token_timestamp(expires_at)
    )

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "lifetime" in reason.lower()


def test_capability_token_rejects_lifetime_over_one_hour():
    now = datetime.now(UTC)
    issued_at = now - timedelta(seconds=1)
    expires_at = issued_at + timedelta(
        seconds=policy_gate_module.MAX_CAPABILITY_TOKEN_LIFETIME_SECONDS + 1
    )
    token = _signed_temporal_token(
        _token_timestamp(issued_at), _token_timestamp(expires_at)
    )

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "maximum" in reason.lower()


@pytest.mark.parametrize(
    "duration",
    [
        -1,
        policy_gate_module.MAX_CAPABILITY_TOKEN_LIFETIME_SECONDS + 1,
        True,
        1.5,
    ],
)
def test_capability_token_generator_rejects_invalid_lifetime(duration):
    with pytest.raises(ValueError, match="duration|negative|maximum"):
        generate_capability_token("diagnostics", "lockdown_confirm", duration)


def test_capability_token_verifier_rejects_non_object_json():
    valid, reason = policy_gate_module.verify_capability_token(
        "[]", "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "JSON object" in reason


def test_capability_token_generator_default_remains_valid():
    token = generate_capability_token("diagnostics", "lockdown_confirm")
    token_data = json.loads(token)

    assert set(token_data) == {
        "capability", "target", "issued_at", "expires", "issued_by", "signature"
    }
    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )
    assert valid is True
    assert reason is None


def test_capability_token_rejects_missing_issued_at():
    now = datetime.now(UTC)
    token_data = {
        "capability": "diagnostics",
        "target": "lockdown_confirm",
        "expires": _token_timestamp(now + timedelta(seconds=15)),
        "issued_by": "operator",
    }
    token_data["signature"] = policy_gate_module.sign_token(
        token_data, policy_gate_module.config.CAPABILITY_TOKEN_SECRET
    )

    valid, reason = policy_gate_module.verify_capability_token(
        json.dumps(token_data), "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "issued_at" in reason.lower()


def test_capability_token_rejects_expired_positive_lifetime():
    issued_at = datetime.now(UTC) - timedelta(seconds=policy_gate_module.MAX_CAPABILITY_TOKEN_LIFETIME_SECONDS + 10)
    expires_at = issued_at + timedelta(seconds=policy_gate_module.MAX_CAPABILITY_TOKEN_LIFETIME_SECONDS)
    token = _signed_temporal_token(
        _token_timestamp(issued_at), _token_timestamp(expires_at)
    )

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "expired" in reason.lower()

def _signed_token_with_overrides(**overrides):
    now = datetime.now(UTC)
    token_data = {
        "capability": "diagnostics",
        "target": "lockdown_confirm",
        "issued_at": _token_timestamp(now),
        "expires": _token_timestamp(now + timedelta(seconds=15)),
        "issued_by": "operator",
    }
    has_signature_override = "signature" in overrides
    signature_override = overrides.pop("signature", None)
    token_data.update(overrides)
    token_data["signature"] = policy_gate_module.sign_token(
        token_data, policy_gate_module.config.CAPABILITY_TOKEN_SECRET
    )
    if has_signature_override:
        token_data["signature"] = signature_override
    return json.dumps(token_data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("capability", None),
        ("capability", 7),
        ("capability", []),
        ("capability", {}),
        ("target", None),
        ("target", 7),
        ("target", []),
        ("target", {}),
    ],
)
def test_capability_token_rejects_non_string_scope_fields(field, value):
    token = _signed_token_with_overrides(**{field: value})

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "must be strings" in reason


@pytest.mark.parametrize(
    "signature",
    [None, 7, [], {}, "A" * 64, "0" * 64, "not-a-signature"],
)
def test_capability_token_rejects_malformed_or_invalid_signatures(signature):
    token = _signed_token_with_overrides(signature=signature)

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert "signature mismatch" in reason.lower()


@pytest.mark.parametrize(
    "field,value",
    [
        ("issued_at", None),
        ("issued_at", 1_798_000_000),
        ("issued_at", []),
        ("expires", None),
        ("expires", 1_798_000_015),
        ("expires", []),
    ],
)
def test_capability_token_rejects_non_string_timestamp_fields(field, value):
    token = _signed_token_with_overrides(**{field: value})

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    assert field in reason

def test_capability_token_verifier_rejects_excessive_json_nesting():
    token = "[" * 10_000 + "0" + "]" * 10_000

    valid, reason = policy_gate_module.verify_capability_token(
        token, "diagnostics", "lockdown_confirm"
    )

    assert valid is False
    # Python 3.12 raises RecursionError at this depth ("JSON parsing failed");
    # 3.14's parser returns the list, which is then refused as a non-object.
    # Either way the token is rejected before any field is trusted.
    assert "JSON parsing failed" in reason or "must be a JSON object" in reason

