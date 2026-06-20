import time
import json
import socket
import sys
import hashlib
import os
os.environ.setdefault("KAIA_CAPABILITY_TOKEN_SECRET", "test_signing_secret_key_2026")

# Set relative directory paths
import pathlib
root_dir = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "core"))

import config
from security.policy_gate import PolicyGate, generate_capability_token, verify_capability_token
import security.db

def send_framed_request(socket_path, flat_payload):
    import uuid
    # Nest the flat request payload into the required IPC Request Schema
    request_id = str(uuid.uuid4())
    nested_payload = {
        "request_id": request_id,
        "action": flat_payload.get("action"),
        "payload": {k: v for k, v in flat_payload.items() if k not in ["action", "capability_token"]},
        "capability_token": flat_payload.get("capability_token")
    }

    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.connect(socket_path)
    
    # Send length-prefixed frame
    payload_bytes = json.dumps(nested_payload).encode('utf-8')
    header = len(payload_bytes).to_bytes(4, byteorder='big')
    client.sendall(header + payload_bytes)
    
    # Receive length-prefixed response
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
    nested_response = json.loads(payload_resp.decode('utf-8'))
    
    # Unwrap nested response
    if not nested_response.get("approved", False):
        exec_resp = nested_response.get("executor_response", {})
        msg = exec_resp.get("message") or "Action denied by Policy Gate."
        return {"status": "denied", "message": msg}
    else:
        return nested_response["executor_response"]

def test_security_flow():
    print("=== Running Security Subsystem Tests ===")
    
    # 1. Init DB
    security.db.initialize_db()
    
    # 2. Token Generation & Verification Test
    print("\n[Test 1] Testing capability token generation & signature...")
    token_str = generate_capability_token("restart_service", "nginx", duration_seconds=10)
    print(f"Generated token: {token_str}")
    
    # Validate token
    ok, err = verify_capability_token(token_str, "restart_service", "nginx")
    if ok:
        print("PASS: Valid token verified successfully.")
    else:
        print(f"FAIL: Token verification failed: {err}")
        return False

    # Validate token with mismatch target
    ok, err = verify_capability_token(token_str, "restart_service", "postgresql")
    if not ok:
        print(f"PASS: Mismatch target token rejected as expected: {err}")
    else:
        print("FAIL: Mismatch target token was accepted!")
        return False

    # Validate token with mismatch capability
    ok, err = verify_capability_token(token_str, "block_ip", "nginx")
    if not ok:
        print(f"PASS: Mismatch capability token rejected as expected: {err}")
    else:
        print("FAIL: Mismatch capability token was accepted!")
        return False

    # 3. Unix Socket Policy Gate Daemon Test
    print("\n[Test 2] Starting Policy Gate Unix Socket Server...")
    gate = PolicyGate()
    # Force socket to tmp for test safety if running in a restricted container
    gate.socket_path = config.POLICY_GATE_SOCKET_FALLBACK
    gate.start()
    
    time.sleep(1.0) # Wait for thread start
    
    # Check if socket file exists
    if os.path.exists(gate.socket_path):
        print(f"PASS: Socket server bound and listening at {gate.socket_path}")
    else:
        print("FAIL: Socket file was not created!")
        gate.stop()
        return False

    # 4. IPC Diagnostics Test
    print("\n[Test 3] Testing Diagnostics IPC payload...")
    payload = {
        "action": "diagnostics",
        "query_type": "ss",
        "args": ["-t"],
        "justification": "Querying TCP socket status",
        "session_id": "sess_test_123"
    }
    
    try:
        response = send_framed_request(gate.socket_path, payload)
        if response.get("status") == "success":
            print("PASS: Diagnostics request executed successfully via HostExecutor.")
            print(f"Diagnostics stdout snippet: {response.get('stdout', '')[:100]}...")
        else:
            print(f"FAIL: Diagnostics query denied: {response.get('message')}")
            gate.stop()
            return False
    except Exception as e:
        print(f"FAIL: IPC Diagnostics connection error: {e}")
        gate.stop()
        return False

    # 5. Service Control Verification (Allowed vs Denied)
    print("\n[Test 4] Testing Service Control Authorization (nginx vs apache2)...")
    
    # Nginx (Allowed in HostExecutor, token matches target)
    token_nginx = generate_capability_token("restart_service", "nginx")
    payload_nginx = {
        "action": "restart_service",
        "service_name": "nginx",
        "justification": "Testing allowed service restart",
        "capability_token": token_nginx,
        "session_id": "sess_test_123"
    }
    
    # Apache2 (Blocked - not in HostExecutor allowlist)
    token_apache = generate_capability_token("restart_service", "apache2")
    payload_apache = {
        "action": "restart_service",
        "service_name": "apache2",
        "justification": "Testing denied service restart",
        "capability_token": token_apache,
        "session_id": "sess_test_123"
    }

    try:
        # Send Nginx restart (allowed in validator schema but we verify execution attempt)
        resp_nginx = send_framed_request(gate.socket_path, payload_nginx)
        
        print(f"Nginx response: {resp_nginx}")
        # Note: if running as non-root on a system where sudo systemctl requires password,
        # HostExecutor will return an error status with stderr containing sudo/auth errors,
        # but the request itself is *approved* and executed by the gate (not denied).
        if resp_nginx.get("status") in ["success", "error"]:
            print("PASS: Allowed service restart passed gate evaluation (forwarded to HostExecutor).")
        else:
            print(f"FAIL: Allowed service restart was blocked by the gate: {resp_nginx.get('message')}")
            gate.stop()
            return False

        # Send Apache2 restart (denied)
        resp_apache = send_framed_request(gate.socket_path, payload_apache)
        
        print(f"Apache2 response: {resp_apache}")
        if resp_apache.get("status") == "denied":
            print(f"PASS: Apache2 service restart denied as expected: {resp_apache.get('message')}")
        else:
            print("FAIL: Unauthorized service restart was not denied!")
            gate.stop()
            return False

    except Exception as e:
        print(f"FAIL: IPC Service Control connection error: {e}")
        gate.stop()
        return False

    # 5. Script Sandboxing Test
    print("\n[Test 5] Testing Script Execution & Bubblewrap Sandboxing...")
    
    # Create allowed script at ~/free-space.sh
    dummy_script_path = os.path.expanduser("~/free-space.sh")
    try:
        with open(dummy_script_path, "w") as f:
            f.write("#!/bin/bash\necho 'SANDBOX_OK'\n")
        os.chmod(dummy_script_path, 0o755)
    except Exception as e:
        print(f"FAIL: Could not create dummy script: {e}")
        gate.stop()
        return False

    token_script = generate_capability_token("run_script", "free-space.sh")
    payload_script = {
        "action": "run_script",
        "script_name": "free-space.sh",
        "justification": "Testing sandboxed script execution",
        "capability_token": token_script,
        "session_id": "sess_test_123"
    }

    try:
        resp_script = send_framed_request(gate.socket_path, payload_script)
        
        print(f"Script response: {resp_script}")
        if resp_script.get("status") == "success" and "SANDBOX_OK" in resp_script.get("stdout", ""):
            print("PASS: Script executed successfully inside Bubblewrap sandbox.")
        else:
            print(f"FAIL: Sandboxed script run failed or was denied: {resp_script.get('message', '')} | stderr: {resp_script.get('stderr', '')}")
            gate.stop()
            return False
    except Exception as e:
        print(f"FAIL: IPC Script run connection error: {e}")
        gate.stop()
        return False
    finally:
        if os.path.exists(dummy_script_path):
            os.remove(dummy_script_path)

    # 5.5 Testing State Modification Protections (Write Limits)
    print("\n[Test 5.5] Testing State Modification Path Protections...")
    from security.host_executor import HostExecutor
    
    # Test writing to .env (must fail)
    success, stdout, stderr = HostExecutor.execute_state_modification(".env", "DB_PASS=compromised")
    if not success and "violation" in stderr:
        print("PASS: Writing to hidden .env file blocked successfully.")
    else:
        print(f"FAIL: Writing to .env was not blocked. Success: {success}, Err: {stderr}")
        gate.stop()
        return False

    # Test writing to python file (must fail)
    success, stdout, stderr = HostExecutor.execute_state_modification("main.py", "print('hacked')")
    if not success and "violation" in stderr:
        print("PASS: Writing to Python script main.py blocked successfully.")
    else:
        print(f"FAIL: Writing to main.py was not blocked. Success: {success}, Err: {stderr}")
        gate.stop()
        return False

    # Test writing to core directory (must fail by directory check, using a non-python file)
    success, stdout, stderr = HostExecutor.execute_state_modification("core/notes.txt", "VERSION='hacked'")
    if not success and "violation" in stderr:
        print("PASS: Writing to core/ package files blocked successfully via directory validation.")
    else:
        print(f"FAIL: Writing to core/notes.txt was not blocked. Success: {success}, Err: {stderr}")
        gate.stop()
        return False

    # Test writing to a legitimate text file (must succeed)
    test_text_path = os.path.join(config.WORKSPACE_DIR, "test_write_ok.txt")
    success, stdout, stderr = HostExecutor.execute_state_modification(test_text_path, "legitimate content")
    if success:
        print("PASS: Writing to legitimate text file in workspace succeeded.")
        if os.path.exists(test_text_path):
            os.remove(test_text_path)
    else:
        print(f"FAIL: Writing to legitimate text file failed: {stderr}")
        gate.stop()
        return False

    # 5.6 Testing Bubblewrap Sandbox Masking (Leakage Protection)
    print("\n[Test 5.6] Testing Bubblewrap Sandbox Masking...")
    # Create a test script that tries to cat .env and list storage
    leak_script_path = os.path.expanduser("~/free-space.sh")
    try:
        env_path = os.path.join(config.WORKSPACE_DIR, ".env")
        storage_path = os.path.join(config.WORKSPACE_DIR, "storage")
        with open(leak_script_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("echo \"=== ENV CONTENT ===\"\n")
            f.write(f"cat {env_path} 2>/dev/null\n")
            f.write("echo \"=== STORAGE CONTENT ===\"\n")
            f.write(f"ls -A {storage_path} 2>/dev/null\n")
        os.chmod(leak_script_path, 0o755)
    except Exception as e:
        print(f"FAIL: Could not create leak script: {e}")
        gate.stop()
        return False

    token_leak = generate_capability_token("run_script", "free-space.sh")
    payload_leak = {
        "action": "run_script",
        "script_name": "free-space.sh",
        "justification": "Testing security sanitization of bubblewrap",
        "capability_token": token_leak,
        "session_id": "sess_test_123"
    }

    try:
        resp_leak = send_framed_request(gate.socket_path, payload_leak)
        
        stdout = resp_leak.get("stdout", "")
        # Verify .env was empty (due to /dev/null binding) and storage is empty (due to tmpfs)
        env_leak_success = "=== ENV CONTENT ===" in stdout and len(stdout.split("=== ENV CONTENT ===")[1].split("=== STORAGE CONTENT ===")[0].strip()) == 0
        storage_leak_success = "=== STORAGE CONTENT ===" in stdout and len(stdout.split("=== STORAGE CONTENT ===")[1].strip()) == 0
        
        if env_leak_success:
            print("PASS: Sandbox could not read host .env file (masked to /dev/null).")
        else:
            print(f"FAIL: Sandbox read .env content! Output:\n{stdout}")
            gate.stop()
            return False

        if storage_leak_success:
            print("PASS: Sandbox could not list files in host storage/ directory (masked via tmpfs).")
        else:
            print(f"FAIL: Sandbox listed storage/ content! Output:\n{stdout}")
            gate.stop()
            return False
            
    except Exception as e:
        print(f"FAIL: IPC Leak run connection error: {e}")
        gate.stop()
        return False
    finally:
        if os.path.exists(leak_script_path):
            os.remove(leak_script_path)


    # 6. Fail-Closed Test
    print("\n[Test 6] Testing Fail-Closed Socket Behavior...")
    gate.stop() # Shut down the socket server daemon
    time.sleep(1.0)
    
    # Try connecting. It must fail.
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(gate.socket_path)
        print("FAIL: Socket connection succeeded when daemon was supposed to be stopped!")
        return False
    except Exception as e:
        print(f"PASS: Socket connection failed as expected (Gate down): {e}")
        
        # Test client side logging of the failed state
        session_id = "sess_fail_closed_test"
        payload_hash = hashlib.sha256(b"test_payload").hexdigest()
        event_id = security.db.log_security_event(
            event_type="policy_gate_unavailable",
            source="client",
            actor="kaiacord",
            payload_hash=payload_hash,
            disposition="blocked",
            session_id=session_id
        )
        print(f"PASS: Fail-closed incident successfully written to security_events.db. Event ID: {event_id}")

    # Check security_events.db entries
    events = security.db.query_security_events(limit=5)
    print(f"\nRecent security events in DB:")
    for ev in events:
        print(f" - {ev['timestamp']} | Event: {ev['type']} | Disposition: {ev['disposition']}")

    print("\n=== ALL SECURITY TESTS PASSED! ===")
    return True

if __name__ == "__main__":
    success = test_security_flow()
    sys.exit(0 if success else 1)
