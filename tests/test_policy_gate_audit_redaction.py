"""Regression tests ensuring audit ledgers never store replayable capability tokens."""

import hashlib
import json
import os

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test-audit-redaction-secret")

import config
from security.policy_gate import PolicyGate, generate_capability_token


def test_audit_record_redacts_raw_token_and_keeps_nonreplayable_metadata(tmp_path, monkeypatch):
    ledger = tmp_path / "audit.jsonl"
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", str(ledger))
    token = generate_capability_token("diagnostics", "nft_list", duration_seconds=30)
    token_data = json.loads(token)

    PolicyGate()._log_audit(
        {
            "action": "diagnostics",
            "query_type": "nft_list",
            "justification": "test audit redaction",
            "capability_token": token,
            "session_id": "unit_test",
        },
        "approved",
    )

    line = ledger.read_text(encoding="utf-8").strip()
    record = json.loads(line)

    assert token not in line
    assert token_data["signature"] not in line
    assert record["request"]["capability_token"] == "[REDACTED]"
    assert record["capability_token"] is None

    metadata = json.loads(record["capability_token_metadata"])
    assert "signature" not in metadata
    assert metadata["capability"] == "diagnostics"
    assert metadata["target"] == "nft_list"
    assert metadata["expires"] == token_data["expires"]
    assert metadata["fingerprint"] == hashlib.sha256(token.encode("utf-8")).hexdigest()


def test_audit_record_with_no_token_has_no_metadata(tmp_path, monkeypatch):
    ledger = tmp_path / "audit.jsonl"
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", str(ledger))

    PolicyGate()._log_audit(
        {"action": "diagnostics", "query_type": "nft_list", "session_id": "unit_test"},
        "denied",
    )

    record = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert record["capability_token"] is None
    assert record["capability_token_metadata"] is None
