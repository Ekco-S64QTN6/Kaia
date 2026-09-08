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
        
        # Files whose modification has already been alerted on, so one incident
        # produces one lockdown instead of one every check interval.
        # filepath -> the observed state that fired (hash, or "missing").
        self._fired = {}

        # 1. Core configs (immutable)
        self.immutable_files = [
            os.path.join(config.WORKSPACE_DIR, ".env"),
            os.path.join(config.WORKSPACE_DIR, "core", "config.py"),
            os.path.join(config.WORKSPACE_DIR, "security", "schemas.py"),
            os.path.join(config.WORKSPACE_DIR, "kaia_dashboard.py"),
            os.path.join(config.WORKSPACE_DIR, "main.py"),
            os.path.join(config.WORKSPACE_DIR, "security", "policy_gate.py"),
            os.path.join(config.WORKSPACE_DIR, "security", "host_executor.py"),
            # Self-protection. Without these three the cheapest bypass in the
            # whole system is to edit the watchdog first: neutralise
            # tamper_detection.py and nothing else here is ever checked again;
            # neutralise db.py and the alert is never recorded; neutralise
            # kaia-lockdown.sh and the response never fires.
            os.path.join(config.WORKSPACE_DIR, "security", "tamper_detection.py"),
            os.path.join(config.WORKSPACE_DIR, "security", "db.py"),
            os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-lockdown.sh"),
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
                self._trigger_tamper_alert(filepath, "File deleted / missing", "missing")
                continue

            if file_type == "immutable":
                current_hash = self._compute_sha256(filepath)
                reason = "Baseline hash mismatch"
            elif isinstance(file_type, tuple) and len(file_type) == 2 and file_type[0] == "sqlite":
                current_hash = self._compute_sqlite_prefix_hash(filepath, file_type[1])
                reason = "SQLite historical row content mismatch (rows altered or deleted)"
            else:
                # Append-only: read up to the original size
                current_hash = self._compute_sha256(filepath, limit=file_type)
                reason = "Historical prefix hash mismatch (file altered or truncated)"

            if current_hash != baseline_hash:
                self._trigger_tamper_alert(filepath, reason, current_hash)
            elif filepath in self._fired:
                # Back to baseline: re-arm so a later, distinct modification
                # alerts again instead of being swallowed by the latch.
                self._fired.pop(filepath, None)
                logger.warning(
                    f"Tamper state cleared, file matches baseline again: {filepath}. "
                    "Detector re-armed for this path."
                )

    def _trigger_tamper_alert(self, filepath: str, reason: str, observed_state: str = ""):
        # Latch. The detector re-checks every 30s and the hash does not return to
        # baseline on its own, so without this a single modification triggers a
        # fresh lockdown forever -- re-flushing nftables every interval while an
        # operator is trying to investigate. Alert loudly once per distinct file
        # state; keep observing quietly after that.
        if self._fired.get(filepath) == observed_state:
            logger.warning(
                f"Tamper still present (already alerted, lockdown not re-triggered): "
                f"{filepath} | Reason: {reason}"
            )
            return
        self._fired[filepath] = observed_state

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

        # Trigger emergency lockdown for files that gate or grant authority.
        # Previously a substring test that missed .env entirely -- the file that
        # holds KAIA_CAPABILITY_TOKEN_SECRET, i.e. the ability to mint any
        # capability token. Tampering with it is the most severe case, not an
        # exempt one. Matched on resolved paths rather than substrings.
        lockdown_paths = {
            os.path.join(config.WORKSPACE_DIR, ".env"),
            os.path.join(config.WORKSPACE_DIR, "core", "config.py"),
            os.path.join(config.WORKSPACE_DIR, "scripts", "kaia-lockdown.sh"),
            "/etc/systemd/system/kaia-policy-gate.service",
        }
        in_security_pkg = os.path.dirname(os.path.abspath(filepath)) == os.path.join(
            os.path.abspath(config.WORKSPACE_DIR), "security"
        )
        if in_security_pkg or os.path.abspath(filepath) in {os.path.abspath(p) for p in lockdown_paths}:
            try:
                from security.policy_gate import trigger_lockdown
                trigger_lockdown(f"Tamper detected on core file: {filepath}")
            except Exception as e:
                logger.error(f"Failed to trigger lockdown for tamper: {e}")
