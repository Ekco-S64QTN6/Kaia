import pytest
import time
import os
import json
import socket
import hashlib
import sqlite3
import pathlib
import sys

# Set paths
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))

os.environ["KAIA_CAPABILITY_TOKEN_SECRET"] = "test_signing_secret_key_2026"

import config
from security.policy_gate import PolicyGate, generate_capability_token
import security.db

def send_framed_request(socket_path, flat_payload):
    import uuid
    request_id = str(uuid.uuid4())
    nested_payload = {
        "request_id": request_id,
        "action": flat_payload.get("action"),
        "payload": {k: v for k, v in flat_payload.items() if k not in ["action", "capability_token"]},
        "capability_token": flat_payload.get("capability_token")
    }

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    payload_bytes = json.dumps(nested_payload).encode('utf-8')
    header = len(payload_bytes).to_bytes(4, byteorder='big')
    client.sendall(header + payload_bytes)
    
    header_resp = bytearray()
    while len(header_resp) < 4:
        packet = client.recv(4 - len(header_resp))
        if not packet:
            raise RuntimeError("Connection closed while reading header")
        header_resp.extend(packet)
    length = int.from_bytes(header_resp, byteorder='big')
    
    payload_resp = bytearray()
    while len(payload_resp) < length:
        packet = client.recv(length - len(payload_resp))
        if not packet:
            raise RuntimeError("Connection closed while reading payload")
        payload_resp.extend(packet)
    client.close()
    return json.loads(payload_resp.decode('utf-8'))

@pytest.fixture(scope="module")
def gate_server():
    # Save original configuration
    orig_g_lvl = config.GLOBAL_LATTICE_LEVEL
    orig_w_lvl = config.WORKSPACE_LATTICE_LEVEL
    orig_g_perms = set(config.GLOBAL_PERMISSIONS)
    orig_w_perms = set(config.WORKSPACE_PERMISSIONS)
    
    # Configure for tests
    config.GLOBAL_LATTICE_LEVEL = "bwrap"
    config.WORKSPACE_LATTICE_LEVEL = "none"
    config.GLOBAL_PERMISSIONS = {"diagnostics", "block_ip", "restart_service", "write_file", "run_script"}
    config.WORKSPACE_PERMISSIONS = {"diagnostics", "block_ip", "restart_service", "write_file", "run_script"}

    security.db.initialize_db()
    gate = PolicyGate()
    gate.socket_path = config.POLICY_GATE_SOCKET_FALLBACK
    gate.start()
    time.sleep(1.0)
    yield gate
    gate.stop()
    if os.path.exists(gate.socket_path):
        try:
            os.remove(gate.socket_path)
        except Exception:
            pass
            
    # Restore configuration
    config.GLOBAL_LATTICE_LEVEL = orig_g_lvl
    config.WORKSPACE_LATTICE_LEVEL = orig_w_lvl
    config.GLOBAL_PERMISSIONS = orig_g_perms
    config.WORKSPACE_PERMISSIONS = orig_w_perms

def check_db_blocked():
    conn = sqlite3.connect(config.SECURITY_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT type, disposition FROM security_events ORDER BY rowid DESC LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row

def test_path_traversal(gate_server):
    token = generate_capability_token("write_file", "../../etc/passwd")
    payload = {
        "action": "write_file",
        "filepath": "../../etc/passwd",
        "content": "malicious",
        "justification": "Testing path traversal",
        "capability_token": token,
        "session_id": "test_neg_1"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    assert "violation" in resp.get("executor_response", {}).get("message", "").lower()
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_blocklisted_py(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, "test.py")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "print(1)",
        "justification": "Testing .py blocklist",
        "capability_token": token,
        "session_id": "test_neg_2"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_blocklisted_sh(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, "test.sh")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "echo 1",
        "justification": "Testing .sh blocklist",
        "capability_token": token,
        "session_id": "test_neg_3"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_hidden_file(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, ".env")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "KEY=val",
        "justification": "Testing hidden file blocklist",
        "capability_token": token,
        "session_id": "test_neg_4"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_protected_dir_core(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, "core", "anything.txt")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "text",
        "justification": "Testing core dir blocklist",
        "capability_token": token,
        "session_id": "test_neg_5"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_protected_dir_security(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, "security", "anything.txt")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "text",
        "justification": "Testing security dir blocklist",
        "capability_token": token,
        "session_id": "test_neg_6"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_expired_token(gate_server):
    # duration_seconds=0 causes it to expire immediately or after 1 sec sleep
    token = generate_capability_token("restart_service", "nginx", duration_seconds=0)
    time.sleep(1.0)
    payload = {
        "action": "restart_service",
        "service_name": "nginx",
        "justification": "operator request for expired token",
        "capability_token": token,
        "session_id": "test_neg_7"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    # Check that error msg mentions verification failure or expired
    assert "token" in resp.get("executor_response", {}).get("message", "").lower()
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_malformed_token(gate_server):
    payload = {
        "action": "restart_service",
        "service_name": "nginx",
        "justification": "operator request for malformed token",
        "capability_token": "garbage_token_string",
        "session_id": "test_neg_8"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    assert "parsing" in resp.get("executor_response", {}).get("message", "").lower() or "signature" in resp.get("executor_response", {}).get("message", "").lower()
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_shell_injection(gate_server):
    payload = {
        "action": "diagnostics",
        "query_type": "ss",
        "args": ["-t; rm -rf /tmp/test"],
        "justification": "operator query with shell injection",
        "session_id": "test_neg_9"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    assert "unsafe shell character" in resp.get("executor_response", {}).get("message", "").lower()
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_unauthorized_service(gate_server):
    token = generate_capability_token("restart_service", "apache2")
    payload = {
        "action": "restart_service",
        "service_name": "apache2",
        "justification": "operator request for unauthorized service",
        "capability_token": token,
        "session_id": "test_neg_10"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    assert "allowed services list" in resp.get("executor_response", {}).get("message", "").lower()
    
    row = check_db_blocked()
    assert row is not None
    assert row[1] == "blocked"

def test_blocklisted_git(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, ".git", "config")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "malicious",
        "justification": "Testing .git blocklist",
        "capability_token": token,
        "session_id": "test_neg_git"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    row = check_db_blocked()
    assert row is not None and row[1] == "blocked"

def test_protected_dir_storage(gate_server):
    path = os.path.join(config.WORKSPACE_DIR, "storage", "anything.txt")
    token = generate_capability_token("write_file", path)
    payload = {
        "action": "write_file",
        "filepath": path,
        "content": "text",
        "justification": "Testing storage dir blocklist",
        "capability_token": token,
        "session_id": "test_neg_storage"
    }
    resp = send_framed_request(gate_server.socket_path, payload)
    assert resp.get("approved") is False
    row = check_db_blocked()
    assert row is not None and row[1] == "blocked"

def test_sandbox_env_masked(gate_server):
    """Verify .env is masked to /dev/null inside the Bubblewrap sandbox (INV-011)."""
    script_path = os.path.expanduser("~/free-space.sh")
    env_path = os.path.join(config.WORKSPACE_DIR, ".env")
    try:
        with open(script_path, "w") as f:
            f.write(f"#!/bin/bash\ncat {env_path} 2>/dev/null\necho EXIT_CODE_$?\n")
        os.chmod(script_path, 0o755)
    except Exception as e:
        pytest.skip(f"Cannot create test script: {e}")

    token = generate_capability_token("run_script", "free-space.sh")
    payload = {
        "action": "run_script",
        "script_name": "free-space.sh",
        "justification": "Sandbox masking test",
        "capability_token": token,
        "session_id": "test_sandbox_mask"
    }
    try:
        resp = send_framed_request(gate_server.socket_path, payload)
        stdout = resp.get("executor_response", {}).get("stdout", "")
        # .env content must not appear — output should only be EXIT_CODE_0 (cat succeeded but file is empty)
        # Check that no key=value patterns appear (would indicate .env leaked)
        import re
        assert not re.search(r"[A-Z_]+=\S+", stdout), f".env content leaked into sandbox stdout: {stdout}"
    finally:
        if os.path.exists(script_path):
            os.remove(script_path)

def test_executor_allowlist_enforced_directly():
    """HostExecutor must enforce its own allowlist even when called without PolicyGate."""
    from security.host_executor import HostExecutor
    success, stdout, stderr = HostExecutor.execute_service_control("apache2")
    assert not success
    assert "allowlist" in stderr.lower() or "not in" in stderr.lower()


# ============================================================
# Tamper detection: self-protection, latching, and lockdown scope
#
# Regression tests for a live incident on 2026-09-08: editing
# security/host_executor.py while the Policy Gate was running caused
# tamper detection to fire kaia-lockdown 24 times in 12 minutes --
# one per 30s check interval -- each firing `nft flush ruleset` and
# taking the host off the network. The detector was correct to fire;
# it was wrong to fire repeatedly, and it was not watching itself.
# ============================================================

def _detector_with_stubs(monkeypatch):
    """A TamperDetector whose side effects (DB write, lockdown) are captured."""
    from security import tamper_detection as td
    import security.policy_gate as pg

    lockdowns = []
    monkeypatch.setattr(td, "log_security_event", lambda **kw: None)
    monkeypatch.setattr(pg, "trigger_lockdown", lambda reason: lockdowns.append(reason))
    return td.TamperDetector(), lockdowns


def test_tamper_detector_watches_itself(monkeypatch):
    """The watchdog must protect its own code, its logger, and its response script.

    Without this, the cheapest bypass in the system is to neutralise
    tamper_detection.py first -- after which nothing else on the watchlist is
    ever checked again.
    """
    detector, _ = _detector_with_stubs(monkeypatch)
    watched = {os.path.abspath(p) for p in detector.immutable_files}
    for required in (
        os.path.join(config.WORKSPACE_DIR, "security", "tamper_detection.py"),
        os.path.join(config.WORKSPACE_DIR, "security", "db.py"),
        os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-lockdown.sh"),
    ):
        assert os.path.abspath(required) in watched, f"{required} is not tamper-protected"


def test_tamper_alert_latches_per_state(monkeypatch):
    """One modification must produce exactly one lockdown, not one per interval."""
    detector, lockdowns = _detector_with_stubs(monkeypatch)
    target = os.path.join(config.WORKSPACE_DIR, "security", "host_executor.py")

    for _ in range(5):
        detector._trigger_tamper_alert(target, "Baseline hash mismatch", "HASH_A")
    assert len(lockdowns) == 1, f"expected 1 lockdown for one state, got {len(lockdowns)}"


def test_tamper_alert_refires_on_new_state(monkeypatch):
    """A genuinely different modification must alert again, not be swallowed."""
    detector, lockdowns = _detector_with_stubs(monkeypatch)
    target = os.path.join(config.WORKSPACE_DIR, "security", "host_executor.py")

    detector._trigger_tamper_alert(target, "Baseline hash mismatch", "HASH_A")
    detector._trigger_tamper_alert(target, "Baseline hash mismatch", "HASH_B")
    assert len(lockdowns) == 2, "a distinct second modification must re-trigger"


def test_env_tamper_triggers_lockdown(monkeypatch):
    """.env holds KAIA_CAPABILITY_TOKEN_SECRET -- tampering with it is the most
    severe case, and previously did not trigger lockdown because the check was a
    substring test for 'security/'."""
    detector, lockdowns = _detector_with_stubs(monkeypatch)
    detector._trigger_tamper_alert(
        os.path.join(config.WORKSPACE_DIR, ".env"), "Baseline hash mismatch", "HASH_ENV"
    )
    assert len(lockdowns) == 1, ".env tampering must trigger emergency lockdown"


# ============================================================
# HostExecutor: defence in depth on script execution
# ============================================================

def test_executor_script_allowlist_enforced_directly():
    """execute_script must enforce SCRIPT_ALLOWLIST even without the Policy Gate,
    matching the standard test_executor_allowlist_enforced_directly sets for
    execute_service_control."""
    from security.host_executor import HostExecutor
    success, _, stderr = HostExecutor.execute_script("definitely_not_allowlisted.sh")
    assert not success
    assert "allowlist" in stderr.lower()


@pytest.mark.parametrize("evil", [
    "/etc/passwd",
    "../../../tmp/evil.sh",
    "subdir/evil.sh",
])
def test_executor_script_rejects_path_escape(evil):
    """os.path.join('~', name) silently discards '~' for absolute names and lets
    '../' climb out of $HOME, so the executor must require a bare filename."""
    from security.host_executor import HostExecutor
    success, _, stderr = HostExecutor.execute_script(evil)
    assert not success
    assert "allowlist" in stderr.lower() or "bare filename" in stderr.lower() \
        or "escapes" in stderr.lower()


# ============================================================
# execute_mitigation must not write into UFW's table
# ============================================================

def test_mitigation_uses_dedicated_table_not_ufw(monkeypatch):
    """IP blocks must land in Kaia's own nftables table.

    Writing to "ip filter" put the rule inside UFW's ruleset, where the next
    `ufw reload` discarded it silently -- while the audit ledger still recorded
    the block as applied. A block that can vanish without a signal is worse than
    no block at all.
    """
    from security.host_executor import HostExecutor

    issued = []
    monkeypatch.setattr(HostExecutor, "_run_cmd",
                        staticmethod(lambda cmd: (issued.append(cmd), (True, "", ""))[1]))

    ok, _, _ = HostExecutor.execute_mitigation("203.0.113.42", "tcp", 4444)
    assert ok

    flat = [" ".join(c) for c in issued]
    assert any(f"add table inet {config.NFT_BLOCK_TABLE}" in c for c in flat), \
        "must create its own table"
    rule = [c for c in flat if "add rule" in c]
    assert rule, "no rule was added"
    assert f"inet {config.NFT_BLOCK_TABLE}" in rule[0], "rule must target Kaia's table"
    assert "ip filter" not in rule[0], "rule must NOT be written into ufw's table"
    assert "203.0.113.42" in rule[0] and "drop" in rule[0]


def test_mitigation_handles_ipv6_selector(monkeypatch):
    """IPv6 targets need the ip6 selector; 'ip saddr' with a v6 address is a
    syntax error nft rejects at rule-add time."""
    from security.host_executor import HostExecutor

    issued = []
    monkeypatch.setattr(HostExecutor, "_run_cmd",
                        staticmethod(lambda cmd: (issued.append(cmd), (True, "", ""))[1]))

    HostExecutor.execute_mitigation("2001:db8::1", "all")
    rule = [" ".join(c) for c in issued if "add rule" in " ".join(c)]
    assert rule and "ip6 saddr" in rule[0], f"expected ip6 selector, got: {rule}"


def test_service_allowlist_has_single_source():
    """policy_gate and host_executor must share one allowlist, not two copies
    that can drift apart."""
    import inspect
    from security import host_executor, policy_gate
    for mod in (host_executor, policy_gate):
        src = inspect.getsource(mod)
        assert "ALLOWED_SERVICES = [" not in src, \
            f"{mod.__name__} redefines the service allowlist locally"
        assert "config.SERVICE_RESTART_ALLOWLIST" in src
