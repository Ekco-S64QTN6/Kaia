"""Starting history fresh archives (never deletes) and keeps used-token records."""

import importlib.util
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("kaia_reset_history", ROOT / "scripts/kaia_reset_history.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _state(tmp_path):
    db = tmp_path / "security_events.db"
    conn = sqlite3.connect(db)
    for stmt in _mod().SCHEMA:
        conn.execute(stmt)
    conn.execute("INSERT INTO security_events VALUES ('e1','t','honeypot_file_access','h','python','','observed','s')")
    conn.execute("INSERT INTO consumed_capability_tokens VALUES ('f'||'0'*63, 9e9, 1.0)")
    conn.commit(); conn.close()
    (tmp_path / "audit_ledger.json").write_text(json.dumps({"result": "approved", "request": {"action": "diagnostics"}}) + "\n")
    (tmp_path / "exec_baseline.json").write_text("{}")
    (tmp_path / "privilege_watch.json").write_text("{}")
    return tmp_path


def test_history_is_archived_and_fresh_files_keep_the_token_registry(tmp_path):
    state = _state(tmp_path)
    r = _mod().reset(str(state), now=0, chown=False)
    archive = Path(r["archive"])
    assert archive.name == "19700101T000000Z" and (archive.stat().st_mode & 0o777) == 0o700
    assert set(r["moved"]) >= {"security_events.db", "audit_ledger.json", "exec_baseline.json"}
    old = sqlite3.connect(archive / "security_events.db")
    assert old.execute("SELECT COUNT(*) FROM security_events").fetchone()[0] == 1       # nothing destroyed
    new = sqlite3.connect(state / "security_events.db")
    assert new.execute("SELECT COUNT(*) FROM security_events").fetchone()[0] == 0
    assert new.execute("SELECT COUNT(*) FROM consumed_capability_tokens").fetchone()[0] == 1
    assert r["tokens_kept"] == 1
    assert (state / "audit_ledger.json").read_text() == ""
    assert (state / "privilege_watch.json").exists()                                    # real state kept
    assert ((state / "audit_ledger.json").stat().st_mode & 0o777) == 0o640


def test_symlinks_in_the_state_directory_are_refused(tmp_path):
    state = _state(tmp_path)
    (state / "conn_baseline.json").symlink_to("/etc/passwd")
    try:
        _mod().reset(str(state), now=1, chown=False)
    except RuntimeError as e:
        assert "symlink" in str(e)
    else:
        raise AssertionError("expected refusal")


def test_active_blocks_prevent_a_reset(tmp_path):
    ledger = tmp_path / "audit_ledger.json"
    ledger.write_text(json.dumps({"result": "approved", "request": {"action": "block_ip", "target_ip": "203.0.113.4"}}) + "\n")
    assert [b["target_ip"] for b in _mod().active_blocks(str(ledger))] == ["203.0.113.4"]
