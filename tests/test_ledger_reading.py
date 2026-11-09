"""HUD and dashboard read the audit ledger in bounded, incremental steps.

The ledger only grows (18 MB, 50k rows on the host by 2026-11). The web HUD read
and parsed all of it on every 5 Hz snapshot to show the newest 50 rows, and the
dashboard re-parsed all of it every 5 s to count tokens. Worse, the token cards
only understood the pre-redaction row format, so no token issued since the gate
stopped storing raw tokens could appear in the HUD's token widget. Fixtures here
are rows written by the gate's own _log_audit.
"""

import hashlib
import json
import os
from datetime import datetime, timedelta, UTC

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test-ledger-reading-secret")

import config
import kaia_dashboard
from kaia_dashboard import LedgerTokenCounter, ledger_token, read_ledger_tail
from security.policy_gate import PolicyGate, generate_capability_token


def _gate_row(ledger, monkeypatch, result="approved", duration=60, token=None):
    monkeypatch.setattr(config, "AUDIT_LOG_PATH", str(ledger))
    token = token or generate_capability_token("diagnostics", "nft_list", duration_seconds=duration)
    PolicyGate()._log_audit(
        {"action": "diagnostics", "query_type": "nft_list", "justification": "ledger reading test",
         "capability_token": token, "session_id": "unit_test"},
        result,
    )
    return token


def _legacy_row(signature="legacy-hmac-signature", expires="2026-10-10T12:00:00Z", result="approved"):
    tok = json.dumps({"capability": "run_script", "target": "free-space.sh",
                      "issued_at": "2026-10-10T11:00:00Z", "expires": expires, "signature": signature})
    return json.dumps({"timestamp": "2026-10-10T11:00:01Z", "result": result,
                       "capability_token": tok, "request": {"capability_token": tok}}) + "\n"


# --- token identity ---

def test_redacted_gate_rows_are_understood(tmp_path, monkeypatch):
    ledger = tmp_path / "audit.jsonl"
    token = _gate_row(ledger, monkeypatch)
    (record,) = read_ledger_tail(str(ledger))
    parsed = ledger_token(record)
    assert parsed is not None, "a row the gate writes today must yield a token card"
    assert parsed["id"] == hashlib.sha256(token.encode()).hexdigest()
    assert parsed["capability"] == "diagnostics" and parsed["target"] == "nft_list"
    assert "signature" not in parsed


def test_legacy_rows_are_identified_by_a_digest_never_the_signature():
    parsed = ledger_token(json.loads(_legacy_row()))
    assert parsed["id"] == hashlib.sha256(b"legacy-hmac-signature").hexdigest()
    assert "legacy-hmac-signature" not in json.dumps(parsed)


def test_rows_without_tokens_or_with_garbage_yield_nothing():
    assert ledger_token({"request": {"capability_token": "[REDACTED]"}, "capability_token": None}) is None
    assert ledger_token({"capability_token_metadata": "[1, 2]"}) is None
    assert ledger_token({"capability_token": "{\"capability\": \"x\"}"}) is None
    assert ledger_token({"request": "not a dict"}) is None


# --- bounded tail ---

def test_tail_reads_a_bounded_window_and_drops_the_partial_first_line(tmp_path):
    ledger = tmp_path / "audit.jsonl"
    with open(ledger, "w") as f:
        for i in range(5000):
            f.write(json.dumps({"n": i, "pad": "x" * 100}) + "\n")
    records = read_ledger_tail(str(ledger), max_records=50, max_bytes=4096)
    assert [r["n"] for r in records][-1] == 4999
    assert len(records) <= 50 and all(isinstance(r, dict) for r in records)
    assert records == sorted(records, key=lambda r: r["n"])


def test_tail_refuses_symlinks_and_skips_oversized_or_broken_lines(tmp_path):
    ledger = tmp_path / "audit.jsonl"
    ledger.write_text("{bad json\n" + json.dumps({"big": "y" * (70 * 1024)}) + "\n" + json.dumps({"ok": 1}) + "\n")
    assert read_ledger_tail(str(ledger)) == [{"ok": 1}]
    link = tmp_path / "link.jsonl"
    link.symlink_to(ledger)
    assert read_ledger_tail(str(link)) == []
    assert read_ledger_tail(str(tmp_path / "missing.jsonl")) == []


# --- incremental counter ---

def test_counter_reads_only_appended_rows_and_counts_each_token_once(tmp_path, monkeypatch):
    ledger = tmp_path / "audit.jsonl"
    counter = LedgerTokenCounter()
    token = _gate_row(ledger, monkeypatch, duration=600)
    assert counter.counts(str(ledger)) == (1, 0, 0)
    _gate_row(ledger, monkeypatch, token=token)          # the same token again: still one
    _gate_row(ledger, monkeypatch, result="denied")      # a new, rejected token
    assert counter.counts(str(ledger)) == (1, 0, 1)
    later = datetime.now(UTC) + timedelta(hours=2)
    assert counter.counts(str(ledger), now=later) == (0, 1, 1)


def test_counter_waits_for_a_complete_line(tmp_path):
    ledger = tmp_path / "audit.jsonl"
    row = _legacy_row()
    ledger.write_text(row[:40])
    counter = LedgerTokenCounter()
    assert counter.counts(str(ledger)) == (0, 0, 0)
    with open(ledger, "a") as f:
        f.write(row[40:])
    assert counter.counts(str(ledger)) == (0, 1, 0)


def test_counter_restarts_on_rotation_or_truncation(tmp_path):
    ledger = tmp_path / "audit.jsonl"
    ledger.write_text(_legacy_row("a") + _legacy_row("b", result="denied"))
    counter = LedgerTokenCounter()
    assert counter.counts(str(ledger)) == (0, 1, 1)
    rotated = tmp_path / "new.jsonl"
    rotated.write_text(_legacy_row("c"))
    os.replace(rotated, ledger)                          # new inode
    assert counter.counts(str(ledger)) == (0, 1, 0)
    with open(ledger, "w") as f:                         # same inode, truncated
        f.write("")
    assert counter.counts(str(ledger)) == (0, 0, 0)


def test_counter_matches_a_full_reparse(tmp_path, monkeypatch):
    ledger = tmp_path / "audit.jsonl"
    for i in range(30):
        _gate_row(ledger, monkeypatch, result="denied" if i % 3 == 0 else "approved", duration=30 + i)
    incremental = LedgerTokenCounter()
    for _ in range(3):
        incremental.counts(str(ledger))
    assert incremental.counts(str(ledger)) == LedgerTokenCounter().counts(str(ledger)) == (20, 0, 10)


# --- HUD ---

def test_hud_token_cards_show_current_tokens_without_signatures(tmp_path, monkeypatch):
    from core.web_bridge import serialize_snapshot
    ledger = tmp_path / "audit.jsonl"
    token = _gate_row(ledger, monkeypatch, duration=300)
    signature = json.loads(token)["signature"]
    snap = kaia_dashboard.KaiamonSnapshot(system_security={"gate_running": True}, containment={})
    data = serialize_snapshot(snap)
    (card,) = data["system_security"]["token_cards"]
    assert card["fingerprint"] == hashlib.sha256(token.encode()).hexdigest()
    assert card["capability"] == "diagnostics" and not card["is_expired"] and card["ttl_seconds"] > 0
    assert signature not in json.dumps(data)


def test_no_reader_loads_the_whole_ledger():
    bridge = open("core/web_bridge.py").read()
    assert "content = f.read()" not in bridge
    dash = open("kaia_dashboard.py").read()
    assert "return self._token_counter.counts(config.AUDIT_LOG_PATH)" in dash
    js = open("core/web/static/js/tokens.js").read()
    assert "token.signature" not in js
