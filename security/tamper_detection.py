import os
import sys
import time
import hashlib
import logging
import threading
import config
from security.db import log_security_event

logger = logging.getLogger(__name__)

class TamperDetector:
    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None
        self._baselines = {}  # filepath -> (hash_val, size_or_type)
        
        # 1. Core configs (immutable)
        self.immutable_files = [
            os.path.join(config.WORKSPACE_DIR, ".env"),
            os.path.join(config.WORKSPACE_DIR, "core", "config.py"),
            os.path.join(config.WORKSPACE_DIR, "security", "schemas.py"),
            os.path.join(config.WORKSPACE_DIR, "kaia_dashboard.py"),
            os.path.join(config.WORKSPACE_DIR, "main.py"),
            os.path.join(config.WORKSPACE_DIR, "security", "policy_gate.py"),
            os.path.join(config.WORKSPACE_DIR, "security", "host_executor.py"),
        ]
        
        # Systemd service unit (if exists)
        sysd_service = "/etc/systemd/system/kaia-policy-gate.service"
        if os.path.exists(sysd_service):
            self.immutable_files.append(sysd_service)
            
        # 2. Append-only security files
        self.append_only_files = [
            config.SECURITY_DB_PATH,
            config.AUDIT_LOG_PATH,
        ]

    def _compute_sha256(self, filepath: str, limit: int = None) -> str:
        h = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                if limit is not None:
                    h.update(f.read(limit))
                else:
                    h.update(f.read())
            return h.hexdigest()
        except Exception as e:
            logger.error(f"Error computing hash for {filepath}: {e}")
            return ""

    def start(self):
        """Establish baselines and start audit thread."""
        logger.info("Initializing tamper detection baseline hashes...")
        
        # Startup sanity check: verify critical files exist
        for filepath in self.immutable_files:
            # .env might be missing in some test setups, check others
            if filepath.endswith(".env") and not os.path.exists(filepath):
                continue
            if not os.path.exists(filepath):
                print(f"FATAL: Critical system file missing: {filepath}", file=sys.stderr)
                sys.exit(1)

        # Baseline immutable files
        for filepath in self.immutable_files:
            if os.path.exists(filepath):
                h = self._compute_sha256(filepath)
                self._baselines[filepath] = (h, "immutable")

        # Baseline append-only files (using prefix-hash technique to avoid false alarms on normal appends)
        for filepath in self.append_only_files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                h = self._compute_sha256(filepath, limit=size)
                self._baselines[filepath] = (h, size)

        self._thread = threading.Thread(target=self._run, daemon=True, name="tamper-detector")
        self._thread.start()
        logger.info("Tamper detection thread started successfully.")

    def stop(self):
        self._stop_event.set()

    def _run(self):
        while not self._stop_event.wait(30.0):
            self.check_integrity()

    def check_integrity(self):
        for filepath, (baseline_hash, file_type) in list(self._baselines.items()):
            if not os.path.exists(filepath):
                self._trigger_tamper_alert(filepath, "File deleted / missing")
                continue
                
            if file_type == "immutable":
                current_hash = self._compute_sha256(filepath)
                if current_hash != baseline_hash:
                    self._trigger_tamper_alert(filepath, "Baseline hash mismatch")
            else:
                # Append-only: read up to the original size
                size_limit = file_type
                current_hash = self._compute_sha256(filepath, limit=size_limit)
                if current_hash != baseline_hash:
                    self._trigger_tamper_alert(filepath, "Historical prefix hash mismatch (file altered or truncated)")

    def _trigger_tamper_alert(self, filepath: str, reason: str):
        msg = f"CRITICAL TAMPER DETECTED: {filepath} | Reason: {reason}"
        print(msg, file=sys.stderr)
        logger.critical(msg)
        
        # Log critical event to security DB
        try:
            log_security_event(
                event_type="tamper_detected",
                source="tamper_detector",
                actor=filepath,
                payload_hash=self._compute_sha256(filepath)[:32] if os.path.exists(filepath) else "deleted",
                disposition="blocked",
                session_id="system_protection"
            )
        except Exception as e:
            logger.error(f"Failed to log tamper event: {e}")

        # Trigger emergency lockdown if core file
        if "security/" in filepath or "core/config.py" in filepath:
            try:
                from security.policy_gate import trigger_lockdown
                trigger_lockdown(f"Tamper detected on core file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to trigger lockdown for tamper: {e}")
