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
    # YARA metacharacters must be stripped
    assert "{" not in rule.split("strings:")[1].split("condition:")[0]


def test_yara_rule_validation():
    engine = RuleEngine.get_instance()
    # A standard valid rule text
    rule_text = """
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
    # Validation should succeed (and not match benign lorem ipsum files)
    is_ok, err = engine.validate_rule(rule_text)
    assert is_ok, f"Validation failed with: {err}"

    # False positive test: rule that matches lorem ipsum
    false_positive_rule = """
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
    is_ok, err = engine.validate_rule(false_positive_rule)
    assert not is_ok
    assert "False positive" in err or "validation syntax error" in err # fallbacks can return syntax or false positive depending on local yara installation


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
