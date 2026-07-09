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
            config.AUDIT_LOG_PATH,
        ]
        
        # 3. SQLite append-only files
        self.sqlite_append_only_files = [
            config.SECURITY_DB_PATH,
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

    def _compute_sqlite_prefix_hash(self, db_path: str, row_count: int) -> str:
        """
        Hashes the logical content (not raw bytes) of the first `row_count` rows
        by rowid. Stable across WAL checkpoints, header counter changes, and
        VACUUM, since it only reflects committed row content.
        """
        import sqlite3
        h = hashlib.sha256()
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, timestamp, type, source, actor, payload_hash, disposition, session_id
                FROM security_events
                ORDER BY rowid ASC
                LIMIT ?
            """, (row_count,))
            for row in cursor.fetchall():
                h.update("|".join(str(c) for c in row).encode("utf-8"))
        except Exception as e:
            logger.error(f"Error computing SQLite content hash for {db_path}: {e}")
            return ""
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass
        return h.hexdigest()

    def _get_sqlite_row_count(self, db_path: str) -> int:
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM security_events")
            return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Error counting rows in {db_path}: {e}")
            return 0
        finally:
            try:
                if conn:
                    conn.close()
            except Exception:
                pass

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

        # Baseline SQLite append-only files (using logical content hashing to avoid false alarms on WAL/header updates)
        for filepath in self.sqlite_append_only_files:
            if os.path.exists(filepath):
                row_count = self._get_sqlite_row_count(filepath)
                h = self._compute_sqlite_prefix_hash(filepath, row_count)
                self._baselines[filepath] = (h, ("sqlite", row_count))

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
            elif isinstance(file_type, tuple) and len(file_type) == 2 and file_type[0] == "sqlite":
                row_count = file_type[1]
                current_hash = self._compute_sqlite_prefix_hash(filepath, row_count)
                if current_hash != baseline_hash:
                    self._trigger_tamper_alert(filepath, "SQLite historical row content mismatch (rows altered or deleted)")
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
