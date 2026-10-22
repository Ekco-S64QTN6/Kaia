import os
import sys
import pathlib

os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))

import time
import pytest
import sqlite3
import tempfile
import shutil
import unittest.mock as mock

import config
from security.schemas import IocRuleRequest
from security.rule_engine import RuleEngine
from security.fim_daemon import FIMDaemon
from security.ebpf_telemetry import EBPFTelemetryEngine
from security.network_discovery import PassiveDiscoveryEngine
from security.honeypot import HoneypotCoordinator, get_remote_ip_for_pid
from security.policy_gate import trigger_lockdown


def test_ioc_rule_schema():
    # Valid request
    req = IocRuleRequest(
        rule_name="detect_eicar",
        author="Test Author",
        threat_description="Testing rule parser matching EICAR signature",
        target_ioc_indicator="EICAR_TEST_STRING",
        mitre_framework_id="T1059"
    )
    assert req.rule_name == "detect_eicar"
    assert req.mitre_framework_id == "T1059"

    # Invalid names
    with pytest.raises(ValueError):
        IocRuleRequest(
            rule_name="detect-eicar-invalid", # hyphen is not allowed by regex ^[a-zA-Z0-9_]+$
            threat_description="Testing invalid characters",
            target_ioc_indicator="test"
        )


def test_rule_compilation():
    engine = RuleEngine.get_instance()
    # Test escaping and formatting
    rule = engine.compile_rule_text(
        rule_name="test_rule",
        author="Author",
        threat_description="Desc text minimum of 10 chars",
        target_ioc_indicator="bad_indicator{}[/]",
        mitre_framework_id="T1027"
    )
    assert "rule test_rule" in rule
    assert "bad_indicator" in rule
    # IOC and metadata are quoted literals; metacharacters must not escape into rule syntax.
    assert '$ioc = "bad_indicator{}[/]"' in rule
    assert "condition:\n        any of them" in rule
    assert "rule bad_indicator" not in rule


def test_yara_rule_validation(monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from security import rule_engine

    engine = RuleEngine.get_instance()
    calls = []

    def fake_sandbox(argv, **kwargs):
        calls.append((argv, kwargs))
        rule_path = Path(argv[-2])
        source = rule_path.read_text(encoding="utf-8")
        stdout = "/tmp/kaia_benign_corpus/benign_0.txt\n" if "Lorem ipsum" in source else ""
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(rule_engine.subprocess, "run", fake_sandbox)
    valid = """
rule test_valid_syntax {
    meta:
        author = "Tester"
        description = "This is a valid rule description"
    strings:
        $s1 = "benign_non_matching_pattern_xyz"
    condition:
        any of them
}
"""
    is_ok, err = engine.validate_rule(valid)
    assert is_ok, err

    false_positive = """
rule test_false_positive {
    meta:
        author = "Tester"
        description = "This will match lorem ipsum"
    strings:
        $s1 = "Lorem ipsum"
    condition:
        any of them
}
"""
    is_ok, err = engine.validate_rule(false_positive)
    assert not is_ok
    assert "False positive" in err
    argv, kwargs = calls[0]
    assert argv[0] == "/usr/bin/systemd-run"
    assert "/usr/bin/yara" in argv
    assert "--wait" in argv and "--pipe" in argv and "--collect" in argv
    assert "-p" in argv and "PrivateNetwork=yes" in argv
    assert kwargs["shell"] is False
    assert "KAIA_CAPABILITY_TOKEN_SECRET" not in kwargs["env"]


def test_yara_validation_fails_closed_when_sandbox_runner_is_missing(monkeypatch):
    from security import rule_engine

    engine = RuleEngine.get_instance()
    monkeypatch.setattr(
        rule_engine.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("systemd-run unavailable")),
    )
    ok, error = engine.validate_rule('rule test_missing_runner { condition: true }')
    assert not ok
    assert "refusing to activate" in error.lower()


def test_rule_engine_rejects_path_traversal_and_duplicate_names(tmp_path, monkeypatch):
    from security import rule_engine

    engine = RuleEngine.get_instance()
    engine.rules_dir = str(tmp_path)
    monkeypatch.setattr(rule_engine, "HAS_YARA", True)

    ok, error = engine.add_rule("../escape", "Tester", "A valid threat description", "indicator")
    assert not ok and "rule name" in error.lower()
    assert not (tmp_path.parent / "escape.yar").exists()

    (tmp_path / "existing_rule.yar").write_text('rule existing_rule { condition: true }', encoding="utf-8")
    ok, error = engine.add_rule("existing_rule", "Tester", "A valid threat description", "indicator")
    assert not ok and "already exists" in error.lower()


def test_rule_engine_refuses_to_store_rule_without_python_yara(tmp_path, monkeypatch):
    from security import rule_engine

    engine = RuleEngine.get_instance()
    engine.rules_dir = str(tmp_path)
    monkeypatch.setattr(rule_engine, "HAS_YARA", False)

    ok, error = engine.add_rule("new_rule", "Tester", "A valid threat description", "indicator")
    assert not ok and "yara-python is required" in error.lower()
    assert not (tmp_path / "new_rule.yar").exists()


def test_fim_daemon_db_query(tmp_path):
    # Mock DB path
    fim = FIMDaemon()
    fim.db_path = str(tmp_path / "fim_test.db")
    
    # Initialize DB
    conn = sqlite3.connect(fim.db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS fim_events (
            timestamp TEXT,
            event_type TEXT,
            pid INTEGER,
            comm TEXT,
            path TEXT,
            yara_matches TEXT,
            sha256 TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO fim_events (timestamp, event_type, pid, comm, path, yara_matches, sha256)
        VALUES ('2026-06-27T12:00:00Z', 'modify', 1234, 'python', '/home/user/test.py', 'test_rule', 'abcdef')
    """)
    conn.commit()
    conn.close()

    alerts = fim.get_recent_alerts(5)
    assert len(alerts) == 1
    assert alerts[0]["path"] == "/home/user/test.py"
    assert alerts[0]["yara_matches"] == "test_rule"


def test_ebpf_telemetry_engine_fallbacks():
    # If BCC is missing, starting the engine should return False gracefully
    engine = EBPFTelemetryEngine.get_instance()
    # Mocking HAS_BCC to False
    with mock.patch("security.ebpf_telemetry.HAS_BCC", False):
        res = engine.start()
        assert res is False


def test_passive_discovery_decoding():
    engine = PassiveDiscoveryEngine.get_instance()
    
    # Test NetBIOS name decoding
    # L2 netbios encoded name of "KAIAPC" padded with spaces (total 16 bytes, i.e. 32 encoded bytes)
    # K: K -> 0x4b -> nibbles 4 and 11 -> E and L
    # Let's test a simpler manual check
    encoded = b"EKELEEEBECECEEEEEEEEEEEEEEEEEEEE" # Decodes to netbios representation
    decoded = engine._decode_netbios_name(encoded)
    assert isinstance(decoded, str)

    # Test DNS name decoding from simple payload
    # format: \x03www\x06google\x03com\x00
    dns_payload = b"\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00" + b"\x03www\x06google\x03com\x00"
    dns_name = engine._parse_dns_name(dns_payload, offset=12)
    assert dns_name == "www.google.com"


def test_lockdown_trigger(tmp_path):
    # Mock config base dir to temp path so it doesn't write to system folders
    lockdown_script = tmp_path / "kaia-lockdown.sh"
    lockdown_script.write_text("#!/bin/bash\necho 'Lockdown running'")
    
    with mock.patch("config.WORKSPACE_DIR", str(tmp_path)), \
         mock.patch("subprocess.run") as mock_run:
        # Mock subprocess.run return code to simulate systemctl failure to trigger fallback
        mock_run.return_value = mock.Mock(returncode=1)
        trigger_lockdown("test_tamper")
        
        # Verify fallback shell script execution
        mock_run.assert_any_call(["/bin/bash", str(tmp_path / "scripts" / "kaia-lockdown.sh")], check=False)


def test_tamper_detector_sqlite_hash(tmp_path):
    from security.tamper_detection import TamperDetector
    import sqlite3
    
    db_path = str(tmp_path / "security_events.db")
    
    # Initialize mock database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            event_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            source TEXT NOT NULL,
            actor TEXT NOT NULL,
            payload_hash TEXT,
            disposition TEXT NOT NULL,
            session_id TEXT NOT NULL
        )
    """)
    cursor.execute("""
        INSERT INTO security_events (event_id, timestamp, type, source, actor, payload_hash, disposition, session_id)
        VALUES ('evt_1', '2026-07-09T08:00:00Z', 'block_ip', 'policy_gate', '127.0.0.1', 'abc', 'approved', 'session_1')
    """)
    conn.commit()
    conn.close()

    detector = TamperDetector()
    detector.sqlite_append_only_files = [db_path]
    detector.append_only_files = []
    
    # 1. Verify row count
    row_count = detector._get_sqlite_row_count(db_path)
    assert row_count == 1
    
    # 2. Verify hash is generated
    h1 = detector._compute_sqlite_prefix_hash(db_path, row_count)
    assert len(h1) == 64  # SHA-256 hex digest length
    
    # 3. Simulate normal database append (new row)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO security_events (event_id, timestamp, type, source, actor, payload_hash, disposition, session_id)
        VALUES ('evt_2', '2026-07-09T08:05:00Z', 'restart_service', 'policy_gate', 'nginx', 'def', 'approved', 'session_2')
    """)
    conn.commit()
    conn.close()
    
    # 4. Content hash of the original first row should not change
    h2 = detector._compute_sqlite_prefix_hash(db_path, row_count)
    assert h1 == h2  # Hash of first row remains identical despite new row append
    
    # 5. Modify the baseline row (tampering simulation)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("UPDATE security_events SET disposition = 'denied' WHERE rowid = 1")
    conn.commit()
    conn.close()
    
    # 6. Verify hash mismatch detects modification
    h3 = detector._compute_sqlite_prefix_hash(db_path, row_count)
    assert h1 != h3  # Tamper detected!


def test_yara_literal_escaping_preserves_data_without_injecting_rule_syntax():
    from security.rule_engine import RuleEngine

    raw = 'quote" slash\\ newline\n rule fake_injected { condition: true }'
    escaped = RuleEngine._escape_yara_literal(raw)
    assert '\\\"' in escaped
    assert "\\\\ " in escaped or escaped.endswith("\\\\")
    assert "\\n" in escaped
    rule = RuleEngine().compile_rule_text(
        "literal_escape_test",
        "Unit test",
        "Valid description with enough characters",
        raw,
        "T1234",
    )
    assert '$ioc = "' + escaped + '"' in rule
    assert "rule fake_injected {" not in rule
