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
