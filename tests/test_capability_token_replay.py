import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import config
import security.db as security_db
import security.policy_gate as policy_gate_module
from security.db import consume_capability_token
from security.policy_gate import PolicyGate, generate_capability_token


def _use_temp_database(tmp_path, monkeypatch):
    db_path = tmp_path / "security_events.db"
    monkeypatch.setattr(config, "SECURITY_DB_PATH", str(db_path))
    return db_path


def test_capability_token_consumption_persists_only_signature_fingerprint(
    tmp_path, monkeypatch
):
    db_path = _use_temp_database(tmp_path, monkeypatch)
    signature = "a" * 64
    expires_at = datetime.now(UTC) + timedelta(seconds=30)

    consumed, reason = consume_capability_token(signature, expires_at)

    assert consumed is True
    assert reason is None
    fingerprint = hashlib.sha256(signature.encode("ascii")).hexdigest()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT token_fingerprint FROM consumed_capability_tokens"
        ).fetchall()
    assert rows == [(fingerprint,)]
    assert signature not in repr(rows)


def test_capability_token_cannot_be_consumed_again_after_database_reopen(
    tmp_path, monkeypatch
):
    _use_temp_database(tmp_path, monkeypatch)
    signature = "b" * 64
    expires_at = datetime.now(UTC) + timedelta(seconds=30)

    assert consume_capability_token(signature, expires_at) == (True, None)
    # Each call opens a fresh SQLite connection, modelling a Policy Gate restart.
    consumed, reason = consume_capability_token(signature, expires_at)

    assert consumed is False
    assert reason == "Token has already been used."


def test_capability_token_consumption_is_atomic_across_threads(
    tmp_path, monkeypatch
):
    _use_temp_database(tmp_path, monkeypatch)
    signature = "c" * 64
    expires_at = datetime.now(UTC) + timedelta(seconds=30)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(
                lambda _: consume_capability_token(signature, expires_at),
                range(8),
            )
        )

    assert sum(consumed for consumed, _ in results) == 1
    assert all(
        reason == "Token has already been used."
        for consumed, reason in results
        if not consumed
    )


def test_expired_consumption_fingerprints_are_retained(tmp_path, monkeypatch):
    db_path = _use_temp_database(tmp_path, monkeypatch)
    security_db.initialize_db()
    old_signature = "d" * 64
    old_fingerprint = hashlib.sha256(old_signature.encode("ascii")).hexdigest()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO consumed_capability_tokens
                (token_fingerprint, expires_at_epoch, consumed_at_epoch)
            VALUES (?, ?, ?)
            """,
            (old_fingerprint, 1.0, 0.5),
        )

    consumed, reason = consume_capability_token(
        "e" * 64, datetime.now(UTC) + timedelta(seconds=30)
    )

    assert consumed is True
    assert reason is None
    with sqlite3.connect(db_path) as conn:
        fingerprints = conn.execute(
            "SELECT token_fingerprint FROM consumed_capability_tokens"
        ).fetchall()
    assert old_fingerprint in {row[0] for row in fingerprints}


def test_capability_token_database_failure_denies_consumption(monkeypatch):
    def fail_connect(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic database outage")

    monkeypatch.setattr(security_db.sqlite3, "connect", fail_connect)

    consumed, reason = consume_capability_token(
        "f" * 64, datetime.now(UTC) + timedelta(seconds=30)
    )

    assert consumed is False
    assert "replay protection" in reason.lower()
    assert "denied" in reason.lower()


def test_verifier_consumes_signature_once_even_with_json_reformatting(
    tmp_path, monkeypatch
):
    _use_temp_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "CAPABILITY_TOKEN_SECRET", "replay-test-secret")
    token = generate_capability_token(
        "diagnostics", "ss", duration_seconds=30
    )
    reformatted_token = json.dumps(json.loads(token), indent=2)

    first = policy_gate_module.verify_capability_token(
        token, "diagnostics", "ss", consume_token=True
    )
    second = policy_gate_module.verify_capability_token(
        reformatted_token, "diagnostics", "ss", consume_token=True
    )

    assert first == (True, None)
    assert second[0] is False
    assert second[1] == "Token has already been used."


def test_wrong_scope_does_not_consume_a_valid_token(tmp_path, monkeypatch):
    _use_temp_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "CAPABILITY_TOKEN_SECRET", "replay-test-secret")
    token = generate_capability_token(
        "block_ip", "203.0.113.42", duration_seconds=30
    )

    wrong_scope = policy_gate_module.verify_capability_token(
        token, "unblock_ip", "203.0.113.42", consume_token=True
    )
    correct_scope = policy_gate_module.verify_capability_token(
        token, "block_ip", "203.0.113.42", consume_token=True
    )

    assert wrong_scope[0] is False
    assert "capability mismatch" in wrong_scope[1].lower()
    assert correct_scope == (True, None)


def test_policy_gate_executes_a_token_authorized_mitigation_only_once(
    tmp_path, monkeypatch
):
    _use_temp_database(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "CAPABILITY_TOKEN_SECRET", "replay-test-secret")
    gate = PolicyGate()
    monkeypatch.setattr(gate, "_log_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        policy_gate_module, "log_security_event", lambda *args, **kwargs: None
    )
    executions = []

    def execute_mitigation(*args):
        executions.append(args)
        return True, "blocked", ""

    monkeypatch.setattr(
        policy_gate_module.HostExecutor, "execute_mitigation", execute_mitigation
    )
    payload = {
        "action": "block_ip",
        "target_ip": "203.0.113.42",
        "protocol": "all",
        "justification": "replay regression",
        "capability_token": generate_capability_token(
            "block_ip", "203.0.113.42", duration_seconds=30
        ),
        "session_id": "replay-regression",
    }

    first = gate.evaluate_and_execute(payload)
    second = gate.evaluate_and_execute(payload)

    assert first["status"] == "success"
    assert second["status"] == "denied"
    assert "already been used" in second["message"].lower()
    assert len(executions) == 1
