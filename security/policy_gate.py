import socket
import os
import json
import logging
import hmac
import hashlib
import re
import threading
from datetime import datetime
from pydantic import ValidationError

import config
from security.schemas import DiagnosticsRequest, MitigationRequest, ServiceControlRequest, StateModificationRequest, ScriptExecutionRequest, AuditRecord
from security.host_executor import HostExecutor
from security.db import log_security_event

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
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        logger.info("Policy Gate thread started.")

    def stop(self):
        """Stops the daemon."""
        self.is_running = False
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

    def _handle_client(self, conn):
        """Handles a single connection transaction."""
        conn.settimeout(5.0)
        try:
            data = conn.recv(65536).decode("utf-8")
            if not data:
                return
            
            try:
                request_payload = json.loads(data)
            except json.JSONDecodeError as e:
                self._send_response(conn, {"status": "error", "message": f"Malformed JSON request: {e}"})
                return

            response = self.evaluate_and_execute(request_payload)
            self._send_response(conn, response)

        except Exception as e:
            logger.error(f"Exception handling client connection: {e}")
            try:
                self._send_response(conn, {"status": "error", "message": f"Internal policy gate error: {e}"})
            except Exception:
                pass
        finally:
            conn.close()

    def _send_response(self, conn, response_dict):
        conn.sendall(json.dumps(response_dict).encode("utf-8"))

    def evaluate_and_execute(self, payload: dict) -> dict:
        """
        Enforces schemas, validates capability tokens, checks static limits,
        executes host actions, logs to audit ledger, and logs failures to security DB.
        """
        action = payload.get("action")
        session_id = payload.get("session_id", "default_session")
        cap_token = payload.get("capability_token")

        if not action:
            return {"status": "error", "message": "Missing 'action' field in request payload."}

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
                    ok_tok, err_tok = verify_capability_token(req.capability_token, "view_logs", req.query_type)
                    if not ok_tok:
                        self._log_audit(req, "denied", reason=f"Invalid capability token: {err_tok}")
                        log_security_event("invalid_capability_token", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                        return {"status": "denied", "message": f"Invalid capability token: {err_tok}"}

                success, stdout, stderr = HostExecutor.execute_diagnostics(req.query_type, req.args)
                result_status = "approved" if success else "failed"
                self._log_audit(req, "approved" if success else "denied", executor="HostExecutor.execute_diagnostics")
                return {"status": "success" if success else "error", "stdout": stdout, "stderr": stderr}

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
                ALLOWED_SERVICES = ["nginx", "postgresql", "ollama", "chroma"]
                if req.service_name not in ALLOWED_SERVICES:
                    self._log_audit(req, "denied", reason=f"Service '{req.service_name}' is not in the allowed services list.")
                    log_security_event("unauthorized_service_restart_attempt", "policy_gate", "kaiacord", hashlib.sha256(str(payload).encode()).hexdigest(), "blocked", session_id)
                    return {"status": "denied", "message": f"Service '{req.service_name}' is not in the allowed services list."}

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
                return {"status": "success" if success else "error", "stdout": stdout, "stderr": stderr}

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

                success, stdout, stderr = HostExecutor.execute_script(req.script_name)
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
                record = AuditRecord(
                    request=request.model_dump(),
                    capability_token=getattr(request, 'capability_token', None),
                    result=result,
                    reason=reason,
                    executor=executor,
                    session_id=request.session_id
                )
                
                # Write to append-only JSONL file
                with open(config.AUDIT_LOG_PATH, "a") as f:
                    f.write(record.model_dump_json() + "\n")
            except Exception as e:
                logger.error(f"Failed to write audit ledger: {e}")
