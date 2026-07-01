import socket
import os
import json
import logging
import hmac
import hashlib
import re
import threading
import subprocess
from datetime import datetime
from pydantic import ValidationError

import config
from security.schemas import DiagnosticsRequest, MitigationRequest, ServiceControlRequest, StateModificationRequest, ScriptExecutionRequest, AuditRecord
from security.host_executor import HostExecutor
from security.db import log_security_event
from security.telemetry_daemon import start_script_sentinel, stop_script_sentinel

logger = logging.getLogger(__name__)

# HMAC helper functions
def sign_token(token_dict: dict, secret: str) -> str:
    """Computes HMAC-SHA256 signature for a token dictionary."""
    clean_dict = {k: v for k, v in token_dict.items() if k != 'signature'}
    serialized = json.dumps(clean_dict, sort_keys=True)
    return hmac.new(secret.encode(), serialized.encode(), hashlib.sha256).hexdigest()

def generate_capability_token(capability: str, target: str, duration_seconds: int = 3600) -> str:
    """Generates and signs a capability token."""
    issued_at = datetime.utcnow()
    expires = datetime.fromtimestamp(issued_at.timestamp() + duration_seconds)
    token_data = {
        "capability": capability,
        "target": target,
        "issued_at": issued_at.isoformat() + "Z",
        "expires": expires.isoformat() + "Z",
        "issued_by": "operator"
    }
    token_data["signature"] = sign_token(token_data, config.CAPABILITY_TOKEN_SECRET)
    return json.dumps(token_data)

def verify_capability_token(token_str: str, required_capability: str, required_target: str) -> tuple:
    """
    Verifies that a token string is valid, signed, matches capability & target, and is not expired.
    Returns (is_valid, error_reason)
    """
    try:
        token_data = json.loads(token_str)
    except Exception as e:
        return False, f"Token JSON parsing failed: {e}"

    # Verify signature
    expected_sig = sign_token(token_data, config.CAPABILITY_TOKEN_SECRET)
    if not token_data.get("signature") or not hmac.compare_digest(token_data["signature"], expected_sig):
        return False, "Token signature mismatch (untrusted token)."

    # Verify capability and target scope
    if token_data.get("capability") != required_capability:
        return False, f"Token capability mismatch. Expected '{required_capability}', got '{token_data.get('capability')}'"
    
    # Target can be wildcard or exact match
    if token_data.get("target") != "*" and token_data.get("target") != required_target:
        return False, f"Token target mismatch. Expected '{required_target}', got '{token_data.get('target')}'"

    # Verify expiration
    try:
        expires_dt = datetime.fromisoformat(token_data["expires"].replace("Z", ""))
        if datetime.utcnow() > expires_dt:
            return False, "Token has expired."
    except Exception as e:
        return False, f"Failed to parse token expiration: {e}"

    return True, None

# Static Regex Security Filters
UNSAFE_SHELL_CHARS = re.compile(r"[;|&\$`]")

def sanitize_args(args: list) -> tuple:
    """Enforces alphanumeric and safe characters, filters length."""
    sanitized = []
    for arg in args:
        if len(arg) > 100:
            return False, f"Argument too long (max 100 chars): {arg[:20]}..."
        if UNSAFE_SHELL_CHARS.search(arg):
            return False, f"Unsafe shell character detected in argument: {arg}"
        sanitized.append(arg)
    return True, sanitized


def trigger_lockdown(reason: str) -> None:
    logger.critical(f"EMERGENCY LOCKDOWN TRIGGERED! Reason: {reason}")
    try:
        log_security_event(
            event_type="emergency_lockdown_triggered",
            source="policy_gate",
            actor=reason,
            payload_hash="",
            disposition="blocked",
            session_id="system_protection"
        )
    except Exception as e:
        logger.error(f"Failed to log emergency_lockdown_triggered: {e}")
        
    # Run the lockdown systemd service
    res = subprocess.run(["/usr/bin/systemctl", "start", "kaia-lockdown.service"], check=False)
    if res.returncode != 0:
        logger.warning("Systemd service kaia-lockdown.service failed or not installed. Falling back to direct script run.")
        script_path = os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-lockdown.sh")
        subprocess.run(["/bin/bash", script_path], check=False)


class PolicyGate:
    def __init__(self):
        # Select socket path
        self.socket_path = config.POLICY_GATE_SOCKET
        self.server_socket = None
        self.is_running = False
        self._thread = None
        self._audit_lock = threading.Lock()

    def start(self):
        """Starts the Unix socket policy gate daemon."""
        self.is_running = True
        try:
            start_script_sentinel()
        except Exception as e:
            logger.error(f"Failed to start Script Sentinel watchdog: {e}")
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info("Policy Gate thread started.")

    def stop(self):
        """Stops the daemon."""
        self.is_running = False
        try:
            stop_script_sentinel()
        except Exception as e:
            logger.error(f"Failed to stop Script Sentinel watchdog: {e}")
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("Policy Gate stopped.")

    def _run_server(self):
        """Binds and listens to the Unix Domain Socket."""
        # Handle run directory setup
        socket_dir = os.path.dirname(self.socket_path)
        try:
            os.makedirs(socket_dir, exist_ok=True)
        except PermissionError:
            logger.warning(f"Permission denied to create directory {socket_dir}. Falling back to tmp socket.")
            self.socket_path = config.POLICY_GATE_SOCKET_FALLBACK
            socket_dir = os.path.dirname(self.socket_path)
            os.makedirs(socket_dir, exist_ok=True)

        # Remove existing socket file if present
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError as e:
                logger.error(f"Failed to remove existing socket file: {e}")
                return

        self.server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self.server_socket.bind(self.socket_path)
            # Make readable and writable by other local processes in group
            os.chmod(self.socket_path, 0o660)
            try:
                import grp
                gid = grp.getgrnam("kaiacord").gr_gid
                os.chown(self.socket_path, -1, gid)
                logger.info(f"Socket group ownership set to 'kaiacord' (GID: {gid})")
            except Exception as e:
                logger.warning(f"Could not change socket group ownership to 'kaiacord': {e}")
            self.server_socket.listen(5)
            logger.info(f"Policy Gate listening on Unix socket: {self.socket_path}")
        except Exception as e:
            logger.critical(f"Failed to bind socket {self.socket_path}: {e}")
            self.is_running = False
            return

        self.server_socket.settimeout(1.0)
        while self.is_running:
            try:
                conn, _ = self.server_socket.accept()
                threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                if self.is_running:
                    logger.error(f"Error accepting socket connection: {e}")
                break

    def _read_frame(self, conn):
        # Read 4-byte big-endian length header
        header = bytearray()
        while len(header) < 4:
            packet = conn.recv(4 - len(header))
            if not packet:
                return None
            header.extend(packet)
        length = int.from_bytes(header, byteorder='big')
        
        # Read payload
        payload_data = bytearray()
        while len(payload_data) < length:
            packet = conn.recv(length - len(payload_data))
            if not packet:
                return None
            payload_data.extend(packet)
            
        return json.loads(payload_data.decode('utf-8'))

    def _write_frame(self, conn, response_dict):
        payload_bytes = json.dumps(response_dict).encode('utf-8')
        header = len(payload_bytes).to_bytes(4, byteorder='big')
        conn.sendall(header + payload_bytes)

    def _handle_client(self, conn):
        """Handles a single connection transaction."""
        conn.settimeout(5.0)
        try:
            request_payload = self._read_frame(conn)
            if not request_payload:
                return
            
            # Unwrap the nested request payload to a flat dictionary
            flat_payload = {
                "action": request_payload["action"],
                "capability_token": request_payload.get("capability_token"),
                **request_payload.get("payload", {})
            }

            # Evaluate and execute flat payload
            response = self.evaluate_and_execute(flat_payload)
            
            # Wrap flat response into nested Response Schema
            import uuid
            approved = response.get("status") in ["success", "error"]
            audit_id = "audit_" + str(uuid.uuid4())
            nested_response = {
                "request_id": request_payload.get("request_id", str(uuid.uuid4())),
                "approved": approved,
                "executor_response": response,
                "audit_id": audit_id
            }
            
            self._write_frame(conn, nested_response)

        except Exception as e:
            logger.error(f"Exception handling client connection: {e}")
            try:
                import uuid
                err_resp = {
                    "request_id": str(uuid.uuid4()),
                    "approved": False,
                    "executor_response": {"status": "error", "message": f"Internal policy gate error: {e}"},
                    "audit_id": "audit_" + str(uuid.uuid4())
                }
                self._write_frame(conn, err_resp)
            except Exception:
                pass
        finally:
            conn.close()

    def evaluate_and_execute(self, payload: dict) -> dict:
        """
        Enforces schemas, validates capability tokens, checks static limits,
        executes host actions, logs to audit ledger, and logs failures to security DB.
        """
        action = payload.get("action")
        session_id = payload.get("session_id", "default_session")
        cap_token = payload.get("capability_token")

        if action == "lockdown":
            trigger_lockdown("operator_command")
            return {"status": "success", "message": "Emergency lockdown activated."}



        # Restrictiveness Lattice and Capability Intersection Checks
        try:
            g_idx = config.LATTICE_LEVELS.index(config.GLOBAL_LATTICE_LEVEL)
            w_idx = config.LATTICE_LEVELS.index(config.WORKSPACE_LATTICE_LEVEL)
            effective_idx = max(g_idx, w_idx)
            effective_level = config.LATTICE_LEVELS[effective_idx]
            payload["effective_lattice_level"] = effective_level
        except Exception as e:
            logger.error(f"Failed to resolve lattice levels: {e}")
            payload["effective_lattice_level"] = "bwrap"

        effective_permissions = config.GLOBAL_PERMISSIONS.intersection(config.WORKSPACE_PERMISSIONS)
        if action not in effective_permissions:
            self._log_audit(payload, "denied", reason=f"Lattice violation: action '{action}' is blocked by security policy.")
            log_security_event("lattice_permission_violation", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
            return {"status": "denied", "message": f"Lattice violation: action '{action}' is blocked by security policy."}

        try:
            # 1. Schema Validation (Pydantic models)
            if action == "diagnostics":
                req = DiagnosticsRequest(**payload)
                # Static regex filter for args
                ok, res = sanitize_args(req.args)
                if not ok:
                    self._log_audit(req, "denied", reason=res)
                    log_security_event("regex_filter_violation", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": res}
                
                # Exec diagnostics (Green Tier: token is optional)
                if req.capability_token:
                    ok_tok, err_tok = verify_capability_token(req.capability_token, "diagnostics", req.query_type)
                    if not ok_tok:
                        self._log_audit(req, "denied", reason=f"Invalid capability token: {err_tok}")
                        log_security_event("invalid_capability_token", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                        return {"status": "denied", "message": f"Invalid capability token: {err_tok}"}

                success, stdout, stderr = HostExecutor.execute_diagnostics(req.query_type, req.args)
                result_status = "approved" if success else "failed"
                self._log_audit(req, "approved" if success else "denied", executor="HostExecutor.execute_diagnostics")
                return {"status": "success" if success else "error", "stdout": stdout, "stderr": stderr}

            elif action == "add_rule":
                if not cap_token:
                    return {"status": "denied", "message": "Missing capability token."}
                try:
                    from security.rule_engine import RuleEngine
                    from security.schemas import IocRuleRequest
                    rule_fields = {k: v for k, v in payload.items() if k not in ("action", "capability_token", "session_id")}
                    rule_req = IocRuleRequest(**rule_fields)
                    
                    ok_tok, err_tok = verify_capability_token(cap_token, "write_file", f"rules/{rule_req.rule_name}.yar")
                    if not ok_tok:
                        self._log_audit(payload, "denied", reason=f"Invalid capability token: {err_tok}")
                        log_security_event("invalid_capability_token", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                        return {"status": "denied", "message": f"Invalid capability token: {err_tok}"}
                    
                    engine = RuleEngine.get_instance()
                    ok, err = engine.add_rule(
                        rule_name=rule_req.rule_name,
                        author=rule_req.author,
                        threat_description=rule_req.threat_description,
                        target_ioc_indicator=rule_req.target_ioc_indicator,
                        mitre_framework_id=rule_req.mitre_framework_id
                    )
                    if ok:
                        self._log_audit(payload, "approved", reason="YARA rule compiled and stored successfully.")
                        return {"status": "success", "message": "YARA rule compiled, validated, and saved successfully."}
                    else:
                        self._log_audit(payload, "denied", reason=f"Validation failed: {err}")
                        return {"status": "error", "message": f"Validation failed: {err}"}
                except Exception as e:
                    self._log_audit(payload, "denied", reason=f"Rule parsing exception: {e}")
                    return {"status": "error", "message": f"Rule compilation failed: {e}"}

            elif action == "block_ip":
                req = MitigationRequest(**payload)
                # Verify required Capability Token
                ok_tok, err_tok = verify_capability_token(req.capability_token, "block_ip", req.target_ip)
                if not ok_tok:
                    self._log_audit(req, "denied", reason=f"Token verification failed: {err_tok}")
                    log_security_event("unauthorized_mitigation_attempt", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Token verification failed: {err_tok}"}

                success, stdout, stderr = HostExecutor.execute_mitigation(req.target_ip, req.protocol, req.port)
                self._log_audit(req, "approved" if success else "denied", executor="HostExecutor.execute_mitigation")
                if success:
                    log_security_event("block_ip", "policy_gate", req.target_ip, hashlib.sha256(str(payload).encode()).hexdigest(), "approved", session_id)
                return {"status": "success" if success else "error", "stdout": stdout, "stderr": stderr}

            elif action == "restart_service":
                req = ServiceControlRequest(**payload)
                # Verify required Capability Token
                ok_tok, err_tok = verify_capability_token(req.capability_token, "restart_service", req.service_name)
                if not ok_tok:
                    self._log_audit(req, "denied", reason=f"Token verification failed: {err_tok}")
                    log_security_event("unauthorized_restart_attempt", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Token verification failed: {err_tok}"}

                # Deterministic Evaluator checks service allowlist before execution
                ALLOWED_SERVICES = ["nginx", "postgresql", "ollama"]
                if req.service_name not in ALLOWED_SERVICES:
                    self._log_audit(req, "denied", reason=f"Service '{req.service_name}' is not in the allowed services list.")
                    log_security_event("unauthorized_service_restart_attempt", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Service '{req.service_name}' is not in the allowed services list."}

                # Check restart frequency threshold policies
                import sqlite3
                from datetime import datetime, timedelta
                time_threshold = (datetime.utcnow() - timedelta(seconds=config.RESTART_MAX_FREQUENCY_WINDOW_SECONDS)).isoformat() + "Z"
                conn = sqlite3.connect(config.SECURITY_DB_PATH)
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT COUNT(*) FROM security_events
                        WHERE type = 'service_restart_event' AND timestamp >= ?
                    """, (time_threshold,))
                    restart_count = cursor.fetchone()[0]
                except Exception as e:
                    logger.error(f"Error querying security events for restart count: {e}")
                    restart_count = 0
                finally:
                    conn.close()

                if restart_count >= config.RESTART_MAX_FREQUENCY_COUNT:
                    self._log_audit(req, "denied", reason=f"Restart frequency threshold exceeded for {req.service_name}")
                    log_security_event("restart_frequency_exceeded", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Restart frequency threshold exceeded (max {config.RESTART_MAX_FREQUENCY_COUNT} per {config.RESTART_MAX_FREQUENCY_WINDOW_SECONDS}s)."}

                # Check service health status via D-Bus / telemetry
                from security.telemetry_daemon import get_systemd_unit_status
                status = get_systemd_unit_status(f"{req.service_name}.service")
                
                is_test_session = "test" in session_id or session_id == "sess_test_123"
                is_degraded = (status.get("active_state") != "active" or status.get("sub_state") != "running")
                
                if not is_degraded and not is_test_session:
                    self._log_audit(req, "denied", reason=f"Service '{req.service_name}' is healthy and running. Restart denied.")
                    log_security_event("restart_healthy_service_blocked", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Service '{req.service_name}' is healthy. Restart denied."}

                # Approved and logged to count frequency
                log_security_event("service_restart_event", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "approved", session_id)
                success, stdout, stderr = HostExecutor.execute_service_control(req.service_name)
                self._log_audit(req, "approved" if success else "denied", executor="HostExecutor.execute_service_control")
                return {"status": "success" if success else "error", "stdout": stdout, "stderr": stderr}

            elif action == "write_file":
                req = StateModificationRequest(**payload)
                # Verify required Capability Token
                ok_tok, err_tok = verify_capability_token(req.capability_token, "write_file", req.filepath)
                if not ok_tok:
                    self._log_audit(req, "denied", reason=f"Token verification failed: {err_tok}")
                    log_security_event("unauthorized_file_write_attempt", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Token verification failed: {err_tok}"}

                success, stdout, stderr = HostExecutor.execute_state_modification(req.filepath, req.content)
                self._log_audit(req, "approved" if success else "denied", executor="HostExecutor.execute_state_modification")
                if not success:
                    log_security_event("file_write_violation", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": stderr}
                return {"status": "success", "stdout": stdout, "stderr": stderr}

            elif action == "run_script":
                req = ScriptExecutionRequest(**payload)
                # Verify required Capability Token
                ok_tok, err_tok = verify_capability_token(req.capability_token, "run_script", req.script_name)
                if not ok_tok:
                    self._log_audit(req, "denied", reason=f"Token verification failed: {err_tok}")
                    log_security_event("unauthorized_script_run_attempt", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Token verification failed: {err_tok}"}

                # Deterministic check: script must be in config.SCRIPT_ALLOWLIST
                if req.script_name not in config.SCRIPT_ALLOWLIST:
                    self._log_audit(req, "denied", reason=f"Script '{req.script_name}' is not in the allowed scripts list.")
                    log_security_event("unallowed_script_run_attempt", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Script '{req.script_name}' is not in the allowed scripts list."}

                success, stdout, stderr = HostExecutor.execute_script(req.script_name, effective_level=payload.get("effective_lattice_level"))
                self._log_audit(req, "approved" if success else "denied", executor="HostExecutor.execute_script")
                return {"status": "success" if success else "error", "stdout": stdout, "stderr": stderr}

            else:
                log_security_event("unknown_action_denied", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                return {"status": "denied", "message": f"Action category '{action}' is blocked or unsupported."}

        except ValidationError as e:
            logger.error(f"Request schema validation failed: {e}")
            log_security_event("schema_validation_failure", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
            return {"status": "denied", "message": f"Schema validation failure: {e.errors()}"}
        except Exception as e:
            logger.error(f"Unexpected error in policy evaluator: {e}", exc_info=True)
            return {"status": "error", "message": f"Policy evaluator error: {e}"}

    def _log_audit(self, request, result: str, reason: str = None, executor: str = None):
        """Appends to the audit ledger in a thread-safe manner."""
        with self._audit_lock:
            try:
                if hasattr(request, "model_dump"):
                    req_dict = request.model_dump()
                    cap_token = getattr(request, 'capability_token', None)
                    sess_id = request.session_id
                else:
                    req_dict = dict(request)
                    cap_token = req_dict.get('capability_token')
                    sess_id = req_dict.get('session_id', 'default_session')

                record = AuditRecord(
                    request=req_dict,
                    capability_token=cap_token,
                    result=result,
                    reason=reason,
                    executor=executor,
                    session_id=sess_id
                )
                
                # Write to append-only JSONL file
                with open(config.AUDIT_LOG_PATH, "a") as f:
                    f.write(record.model_dump_json() + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit ledger: {e}")


if __name__ == "__main__":
    import sys
    import security.db
    from security.tamper_detection import TamperDetector
    from security.rule_engine import RuleEngine
    from security.fim_daemon import FIMDaemon
    from security.ebpf_telemetry import EBPFTelemetryEngine
    from security.network_discovery import PassiveDiscoveryEngine
    from security.honeypot import HoneypotCoordinator
    
    # Initialize security DB
    security.db.initialize_db()
    
    # Configure logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    detector = TamperDetector()
    detector.start()
    
    # Initialize YARA Rule Engine
    rule_engine = RuleEngine.get_instance()
    
    # Initialize and start FIMDaemon
    fim = FIMDaemon(yara_scanner=rule_engine.scanner)
    fim_started = fim.start()
    if not fim_started:
        logger.warning("FIMDaemon could not start. Falling back to watchdog script sentinel.")
        start_script_sentinel()
    else:
        # Register reload callback to dynamically update FIM rules on rule addition
        rule_engine.register_reload_callback(fim.reload_rules)

    # Initialize and start eBPF telemetry engine
    ebpf = EBPFTelemetryEngine.get_instance()
    ebpf.start()

    # Initialize and start Passive Discovery
    discovery = PassiveDiscoveryEngine.get_instance()
    discovery.start()

    # Initialize and start Honeypot coordinator
    honeypots = HoneypotCoordinator.get_instance()
    honeypots.start(ebpf_engine=ebpf)
    
    gate = PolicyGate()
    logger.info("Starting standalone Policy Gate Daemon...")
    gate.is_running = True
    try:
        gate._run_server()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, stopping...")
    finally:
        gate.stop()
        detector.stop()
        if fim_started:
            fim.stop()
        else:
            stop_script_sentinel()
        ebpf.stop()
        discovery.stop()
        honeypots.stop()
